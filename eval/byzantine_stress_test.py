#!/usr/bin/env python3
"""
Byzantine Resilience Stress Test - Suite 3

Attacks plain FedAvg (no defense) with 1 malicious client sending
adversarial gradients (random noise scaled 10× true gradient norm).
Then tests three Byzantine-resilient aggregation methods:
  --byzantine norm   : discard gradients whose norm deviates >3σ from mean
  --byzantine krum  : multi-client geometric median approximation
  --byzantine trimmed_mean : discard top/bottom α fractions before averaging

Runs 5 configs per Suite 3:
  3a  baseline (no attack, no defense)
  3b  attack only (1 malicious, no defense)
  3c  attack + norm defense
  3d  attack + krum defense
  3e  attack + trimmed_mean defense

Each run: 3 rounds, 3 clients, Pythia-70m, wikitext-2.
Output: eval/results/suite3_*.json
"""

import gc
import json
import math
import os
import sys
import time
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "eval" / "results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "EleutherAI/pythia-70m"
NUM_CLIENTS = 5
NUM_ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH_SIZE = 4
MAX_SEQ_LEN = 128
LR = 8e-4  # LoRA init: need higher LR to break through random-A noise; previous 3e-4 diverged
LORA_RANK = 4
LORA_ALPHA = 8.0
LORA_DROPOUT = 0.05
LISA_BOTTOM = 2
LISA_TOP = 2
LISA_MIDDLE = 2
MAX_TRAIN_BATCHES_PER_CLIENT = 20
MAX_TEST_BATCHES = 20
SEED = 42

# Byzantine defaults
BYZANTINE_METHOD = "none"       # none | norm | krum | trimmed_mean
MALICIOUS_CLIENTS = 0          # number of malicious clients (0 = no attack)
ATTACK_TYPE = "label_flip"     # label_flip = noise scaled 10x true grad norm
TRIMMED_MEAN_TRIM = 0.1        # fraction to trim from each tail (10%)


# ---------------------------------------------------------------------------
# Byzantine defense: norm-based outlier rejection
# ---------------------------------------------------------------------------

def byzantine_norm_filter(deltas: List[Dict[str, torch.Tensor]],
                           weights: List[float],
                           sigma_threshold: float = 3.0) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
    """
    Compute L2 norm of each client's aggregated delta.
    Uses Median Absolute Deviation (MAD) for robust outlier detection.
    MAD is robust to Byzantine outliers unlike mean/std which are dominated by them.
    """
    norms = []
    for delta in deltas:
        n = math.sqrt(sum(v.float().pow(2).sum().item() for v in delta.values()))
        norms.append(n)

    n = len(norms)
    norms_arr = np.array(norms)
    sorted_norms = np.sort(norms_arr)
    median_n = sorted_norms[n // 2] if n % 2 == 1 else (sorted_norms[n // 2 - 1] + sorted_norms[n // 2]) / 2

    # Median Absolute Deviation (MAD) - robust to outliers
    abs_devs = np.abs(norms_arr - median_n)
    sorted_devs = np.sort(abs_devs)
    mid_d = n // 2
    mad = sorted_devs[mid_d] if n % 2 == 1 else (sorted_devs[mid_d - 1] + sorted_devs[mid_d]) / 2

    # Modified z-score threshold
    # Constant 0.6745 maps MAD to std under normal distribution
    # Modified z-score = 0.6745 * |x - median| / MAD
    if mad < 1e-10:
        # All norms essentially identical
        threshold = median_n
        outlier_mask = np.zeros(n, dtype=bool)
    else:
        modified_z = 0.6745 * abs_devs / mad
        outlier_mask = modified_z > sigma_threshold
        # Compute threshold in original norm space for logging
        threshold = median_n + sigma_threshold * (mad / 0.6745)

    # Log diagnostics
    if mad < 1e-10:
        mad_display = "identical"
        z_scores = ["0.0"] * n
    else:
        mad_display = f"{mad:.4f}"
        z_scores = [f"{0.6745 * abs_devs[i] / mad:.2f}" for i in range(n)]
    logger.info(f"    [norm-filter] norms={[f'{x:.4f}' for x in norms]}")
    logger.info(f"    [norm-filter] median={median_n:.4f} MAD={mad_display}")
    logger.info(f"    [norm-filter] threshold={threshold:.4f} z_scores={z_scores}")

    keep_indices = [i for i in range(n) if not outlier_mask[i]]

    dropped = set(range(n)) - set(keep_indices)
    if dropped:
        dropped_norms = [norms[i] for i in dropped]
        logger.info(f"    [norm-filter] DROPPED clients: {dropped} (norms={dropped_norms})")

    return [deltas[i] for i in keep_indices], [weights[i] for i in keep_indices]


# ---------------------------------------------------------------------------
# Byzantine defense: Multi-Krum
# ---------------------------------------------------------------------------

def byzantine_krum(deltas: List[Dict[str, torch.Tensor]],
                   weights: List[float],
                   n_malicious: int = 1) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Compute the Multi-Krum aggregation.
    Score each client by summed squared distances to its f-nearest neighbours.
    Keep the n - f - 1 clients with lowest scores, then average.

    Bug fixes from original:
    - Guard was n <= 2*f+1 (wrong: allows n=2f+1 which is too few).
      Correct minimum: n >= 2f+3. We fall back when n < 2f+3.
    - k (neighbors) and n_keep could be 0 or negative for tiny n; clamp with max().
    """
    n = len(deltas)
    f = n_malicious

    # Correct viability check: need n >= 2f+3 for Krum to work properly.
    # If fewer clients, fall back to simple averaging (which is all we can do).
    if n < 2 * f + 3:
        logger.warning(f"    [krum] Not enough clients ({n}) for f={f} malicious "
                       f"(need n>=2f+3={2*f+3}). Falling back to weighted mean.")
        acc: Dict[str, torch.Tensor] = {}
        for delta, w in zip(deltas, weights):
            for k, v in delta.items():
                acc[k] = acc.get(k, torch.zeros_like(v)) + v.float() * w
        total_w = sum(weights)
        for k in acc:
            acc[k] /= total_w
        return acc, 1.0

    # Compute pairwise squared distances between clients based on flattened delta vectors
    def flatten_delta(d):
        vals = []
        for v in sorted(d.keys()):
            vals.append(d[v].float().flatten())
        return torch.cat(vals)

    flat_deltas = [flatten_delta(d) for d in deltas]

    # Compute pairwise Euclidean distances and Krum scores
    # score_i = sum of squared distances to k nearest neighbors (excluding self)
    k = max(1, n - f - 2)  # number of nearest neighbors to consider
    n_keep = max(1, n - f - 1)  # number of clients to keep

    scores = []
    all_dists = []  # for debug logging
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            d = (flat_deltas[i] - flat_deltas[j]).pow(2).sum().item()
            dists.append((j, d))
        dists.sort(key=lambda x: x[1])
        # Sum squared distances to k nearest neighbours
        score = sum(d for _, d in dists[:k])
        scores.append(score)
        all_dists.append(dists[:k])

    logger.info(f"    [krum] n={n} f={f} k={k} n_keep={n_keep}")
    logger.info(f"    [krum] scores={[f'{s:.4f}' for s in scores]}")

    # Keep n - f - 1 clients with lowest scores
    sorted_indices = sorted(range(n), key=lambda i: scores[i])
    keep_indices = sorted_indices[:n_keep]
    selected_client = keep_indices[0]  # primary selected client for logging

    logger.info(f"    [krum] keeping clients: {keep_indices} "
                f"(dropped {set(range(n)) - set(keep_indices)})")
    logger.info(f"    [krum] primary selected client: {selected_client}")

    # Weighted average of kept deltas
    total_w = sum(weights[i] for i in keep_indices)
    acc: Dict[str, torch.Tensor] = {}
    for idx in keep_indices:
        for k, v in deltas[idx].items():
            acc[k] = acc.get(k, torch.zeros_like(v.float())) + v.float() * (weights[idx] / total_w)

    return acc, 1.0


# ---------------------------------------------------------------------------
# Byzantine defense: Trimmed Mean
# ---------------------------------------------------------------------------

def byzantine_trimmed_mean(deltas: List[Dict[str, torch.Tensor]],
                            weights: List[float],
                            trim_fraction: float = TRIMMED_MEAN_TRIM) -> Dict[str, torch.Tensor]:
    """
    For each parameter tensor, sort element-wise across clients, trim the top/bottom
    trim_fraction, then average the remaining.
    """
    n = len(deltas)
    n_trim = max(1, int(n * trim_fraction))
    n_keep = n - 2 * n_trim
    if n_keep < 1:
        logger.warning(f"    [trimmed_mean] n={n} too small for trim={trim_fraction}. Using mean.")
        n_trim = 0
        n_keep = n

    logger.info(f"    [trimmed_mean] n={n} trim={n_trim} each side -> keep={n_keep}")

    # Collect all parameter keys
    all_keys = set()
    for d in deltas:
        all_keys.update(d.keys())

    result: Dict[str, torch.Tensor] = {}
    for key in sorted(all_keys):
        tensors = [d[key].float() for d in deltas if key in d]
        if len(tensors) < n:
            # Pad with zeros for missing
            max_shape = max(t.shape for t in tensors)
            padded = []
            for t in tensors:
                if t.shape != max_shape:
                    pad_t = torch.zeros(max_shape)
                    pad_t[:t.numel()] = t.flatten()
                    padded.append(pad_t)
                else:
                    padded.append(t.flatten())
            tensors = padded

        # Stack -> sort along dim=0 -> trim -> mean
        stacked = torch.stack(tensors, dim=0)  # [n, *dims]
        flat_shape = [stacked.shape[0], -1]
        flat = stacked.flatten(1)  # [n, total_params]

        # Sort and trim
        sorted_flat, _ = flat.sort(dim=0)
        trimmed = sorted_flat[n_trim:n - n_trim, :]  # [n_keep, total_params]

        if trimmed.numel() == 0:
            result[key] = flat.mean(dim=0).view_as(tensors[0])
        else:
            result[key] = trimmed.mean(dim=0)

        result[key] = result[key].view_as(tensors[0])

    return result


# ---------------------------------------------------------------------------
# Adversarial gradient injection
# ---------------------------------------------------------------------------

def inject_adversarial_delta(delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Label-flip attack: replace the true gradient with random noise scaled to
    10× the true gradient's L2 norm.
    """
    # Compute true gradient norm
    true_norm = math.sqrt(sum(v.float().pow(2).sum().item() for v in delta.values()))

    # Generate adversarial delta: random noise with 10× the true norm
    adv_delta = {}
    for k, v in delta.items():
        v_float = v.float()
        noise = torch.randn_like(v_float)
        noise_norm = math.sqrt(noise.pow(2).sum().item()) + 1e-8
        scaled_noise = noise * (10.0 * true_norm / noise_norm)
        adv_delta[k] = scaled_noise

    adv_norm = math.sqrt(sum(v.pow(2).sum().item() for v in adv_delta.values()))
    logger.info(f"    [attack] true_norm={true_norm:.6f} adv_norm={adv_norm:.6f} scale=10x")
    return adv_delta


# ---------------------------------------------------------------------------
# LoRA helpers (same as fedavg_vs_lisafedavg.py)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Module, rank: int = 4, alpha: float = 8.0,
                 dropout: float = 0.05):
        super().__init__()
        self.weight_data = linear.weight.data.clone().float()
        self.bias_data = linear.bias.data.clone().float() if linear.bias is not None else None
        self.out_features, self.in_features = self.weight_data.shape
        self.is_conv1d = isinstance(linear, nn.Conv1d)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # Option 3 fix: lora_B = zero (standard LoRA init).
        # Standard LoRA: BA must start at 0 so the model IS the original pretrained model.
        # lora_A random std=0.01, lora_B = 0. With B=0, BA=0 and no perturbation.
        # lora_B then grows from gradients during training.
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        with torch.no_grad():
            original = nn.functional.linear(x_f32, self.weight_data, self.bias_data)
        lora_input = self.lora_dropout(x_f32)
        lora = nn.functional.linear(lora_input, self.lora_A)
        lora = nn.functional.linear(lora, self.lora_B)
        result = original + lora * self.scaling
        return result.to(orig_dtype)

    def trainable_params(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]


class LoraAppliedModel:
    TARGET_MODULES = [
        "query_key_value", "dense",
        "dense_h_to_4h", "dense_4h_to_h",
        "c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "fc1", "fc2", "c_fc",
    ]

    def __init__(self, model: nn.Module, rank: int = 4, alpha: float = 8.0,
                 dropout: float = 0.05):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_layers: Dict[str, LoRALinear] = {}

    def apply_lora(self) -> int:
        count = 0
        for full_name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv1d)):
                continue
            name_parts = full_name.split(".")
            if not any(tm in name_parts[-1] for tm in self.TARGET_MODULES):
                continue
            lora = LoRALinear(module, rank=self.rank, alpha=self.alpha, dropout=self.dropout)
            self.lora_layers[full_name] = lora
            parts = full_name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr = parts
                try:
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, attr, lora)
                    count += 1
                except KeyError:
                    pass
        logger.info(f"  LoRA applied to {count} layers (rank={self.rank})")
        return count

    def freeze_all(self):
        """Freeze all model parameters (including LoRA layers)."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_lora_layers(self, layer_indices: List[int]):
        """
        Unfreeze BOTH lora_A and lora_B for selected layers.
        FIX: Both A and B must be trainable for meaningful gradient updates.
        With B starting at zero, A alone produces near-zero output and the
        model cannot learn. B must also receive gradients to grow from zero.
        """
        patterns = []
        for idx in layer_indices:
            patterns.extend([f"gpt_neox.layers.{idx}.", f".h.{idx}."])
        for full_name, lora_layer in self.lora_layers.items():
            for pat in patterns:
                if pat in full_name:
                    for p in lora_layer.trainable_params():
                        p.requires_grad = True
                    break

    def unfreeze_lora_A_only(self, layer_indices: Optional[List[int]] = None):
        """
        DEPRECATED / FOR DEBUG ONLY.
        Unfreezing only lora_A is incorrect for actual training because
        lora_B stays frozen at zero, making lora_A @ lora_B = 0 regardless of A.
        Use unfreeze_lora_layers() which unfreezes both A and B.
        This method is kept only for experimental comparison.
        """
        if layer_indices is None:
            for lora_layer in self.lora_layers.values():
                lora_layer.lora_A.requires_grad = True
                lora_layer.lora_B.requires_grad = False
        else:
            patterns = []
            for idx in layer_indices:
                patterns.extend([f"gpt_neox.layers.{idx}.", f".h.{idx}."])
            for full_name, lora_layer in self.lora_layers.items():
                for pat in patterns:
                    if pat in full_name:
                        lora_layer.lora_A.requires_grad = True
                        lora_layer.lora_B.requires_grad = False
                        break

    def get_trainable_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_wikitext(tokenizer, max_seq: int = MAX_SEQ_LEN) -> Tuple[List[str], List[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return _synthetic_data()
    try:
        ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
        test_ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
    except Exception:
        return _synthetic_data()
    train_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
    test_texts = [t for t in test_ds["text"] if t.strip() and len(t.strip()) > 20]
    logger.info(f"  wikitext: {len(train_texts)} train, {len(test_texts)} test lines")
    return train_texts, test_texts


def _synthetic_data(n_train: int = 600, n_test: int = 100) -> Tuple[List[str], List[str]]:
    domains = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning neural network training data",
        "federated learning distributed privacy client server",
        "language model transformer attention mechanism",
        "optimization gradient descent learning rate",
    ]
    words = " ".join(domains).split()
    rng = random.Random(SEED)
    train_texts, test_texts = [], []
    for _ in range(n_train):
        sel = rng.sample(words, min(30, len(words)))
        rng.shuffle(sel)
        train_texts.append(" ".join(sel * 3)[:150])
    for _ in range(n_test):
        sel = rng.sample(words, min(25, len(words)))
        rng.shuffle(sel)
        test_texts.append(" ".join(sel * 3)[:150])
    logger.info(f"  synthetic: {len(train_texts)} train, {len(test_texts)} test lines")
    return train_texts, test_texts


def tokenize_texts(tokenizer, texts: List[str], max_seq: int) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        max_length=max_seq,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["input_ids"].clone(),
    }


def partition_data(texts: List[str], n: int, seed: int = SEED) -> List[List[str]]:
    rng = random.Random(seed)
    shuffled = texts.copy()
    rng.shuffle(shuffled)
    size = len(shuffled) // n
    partitions = []
    for i in range(n):
        start = i * size
        end = start + size if i < n - 1 else len(shuffled)
        partitions.append(shuffled[start:end])
    return partitions


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(model: nn.Module, test_enc: Dict[str, torch.Tensor],
                       batch_size: int = BATCH_SIZE) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    n_batches = min((len(test_enc["input_ids"]) + batch_size - 1) // batch_size, MAX_TEST_BATCHES)
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(test_enc["input_ids"]))
        ids = test_enc["input_ids"][start:end].clone().clamp(0, model.config.vocab_size - 1)
        mask = test_enc["attention_mask"][start:end]
        labs = test_enc["labels"][start:end].clone().clamp(0, model.config.vocab_size - 1)
        outputs = model(input_ids=ids, attention_mask=mask)
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labs.view(-1))
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1)) if total_tokens > 0 else float("inf")


# ---------------------------------------------------------------------------
# LISA layer selection
# ---------------------------------------------------------------------------

def lisa_select_layers(num_layers: int, bottom: int = LISA_BOTTOM,
                       top: int = LISA_TOP, middle: int = LISA_MIDDLE,
                       seed: Optional[int] = None) -> List[int]:
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    bottom_set = list(range(min(bottom, num_layers)))
    top_set = list(range(max(0, num_layers - top), num_layers))
    middle_pool = list(range(bottom, max(bottom, num_layers - top)))
    middle_set = rng.sample(middle_pool, min(middle, len(middle_pool))) if middle_pool else []
    return sorted(set(bottom_set + top_set + middle_set))


# ---------------------------------------------------------------------------
# State snapshots
# ---------------------------------------------------------------------------

def snapshot_lora_state(wrapper: LoraAppliedModel) -> Dict[str, torch.Tensor]:
    state = {}
    for full_name, lora_layer in wrapper.lora_layers.items():
        state[f"{full_name}.lora_A"] = lora_layer.lora_A.data.clone().cpu()
        state[f"{full_name}.lora_B"] = lora_layer.lora_B.data.clone().cpu()
    return state


def restore_lora_state(wrapper: LoraAppliedModel, state: Dict[str, torch.Tensor]):
    for full_name, lora_layer in wrapper.lora_layers.items():
        lora_layer.lora_A.data.copy_(state[f"{full_name}.lora_A"].clone())
        lora_layer.lora_B.data.copy_(state[f"{full_name}.lora_B"].clone())


def compute_deltas(before: Dict[str, torch.Tensor],
                   after: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: after[k] - before[k] for k in before}


# ---------------------------------------------------------------------------
# Client training
# ---------------------------------------------------------------------------

def train_client(model: nn.Module, tokenizer, wrapper: LoraAppliedModel,
                 client_texts: List[str], round_num: int, client_id: str,
                 selected_layers: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Train LoRA layers locally on client texts.
    FIX: unfreeze_lora_layers() now unfreezes BOTH lora_A and lora_B.
    Previously unfreeze_lora_layers only unfroze A (via a different code path
    in unfreeze_lora_A_only which was called by mistake). With B frozen at zero,
    lora_A @ lora_B = 0 and no learning occurred.
    """
    wrapper.freeze_all()
    if selected_layers is None:
        for lora_layer in wrapper.lora_layers.values():
            for p in lora_layer.trainable_params():
                p.requires_grad = True
    else:
        wrapper.unfreeze_lora_layers(selected_layers)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    enc = tokenize_texts(tokenizer, client_texts, MAX_SEQ_LEN)
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    model.train()

    losses = []
    n_batches = min((len(enc["input_ids"]) + BATCH_SIZE - 1) // BATCH_SIZE, MAX_TRAIN_BATCHES_PER_CLIENT)
    _debug_first = True

    for _ in range(LOCAL_EPOCHS):
        indices = torch.randperm(len(enc["input_ids"])).tolist()
        for i in range(n_batches):
            idx = indices[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            ids = enc["input_ids"][idx].clone().clamp(0, tokenizer.vocab_size - 1)
            mask = enc["attention_mask"][idx]
            labs = enc["labels"][idx].clone().clamp(0, tokenizer.vocab_size - 1)
            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask, labels=labs)
            loss = outputs.loss
            if torch.isnan(loss):
                continue
            loss.backward()
            if _debug_first and round_num == 1:
                grad_norms = {f"{fn}.lora_B": ll.lora_B.grad.norm().item()
                              for fn, ll in wrapper.lora_layers.items()
                              if ll.lora_B.grad is not None and ll.lora_B.grad.norm().item() > 1e-10}
                logger.info(f"  [DEBUG] {client_id} lora_B grads: {len(grad_norms)} non-zero {list(grad_norms.keys())[:3]}")
                _debug_first = False
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "client_id": client_id,
        "round": round_num,
        "avg_train_loss": avg_loss,
        "selected_layers": selected_layers,
    }


# ---------------------------------------------------------------------------
# Aggregation with Byzantine defense
# ---------------------------------------------------------------------------

def aggregate_deltas(
    deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
    wrapper: LoraAppliedModel,
    byzantine_method: str = "none",
    n_malicious: int = 0,
    round_num: int = 0,
) -> None:
    """Apply weighted gradient deltas using specified Byzantine defense."""
    if not deltas or not weights:
        return

    # Build flat dict for Byzantine methods
    deltas_copy = [{k: v.float() for k, v in d.items()} for d in deltas]

    if byzantine_method == "none":
        # Standard weighted average
        acc: Dict[str, torch.Tensor] = {}
        for delta, w in zip(deltas_copy, weights):
            for k, v in delta.items():
                acc[k] = acc.get(k, torch.zeros_like(v)) + v * w
        total_w = sum(weights)
        for k in acc:
            acc[k] /= total_w

    elif byzantine_method == "norm":
        # Filter outliers, then weighted average
        filtered_deltas, filtered_weights = byzantine_norm_filter(deltas_copy, weights)
        if not filtered_deltas:
            logger.warning("  [norm] All clients filtered! Skipping aggregation.")
            return
        acc = {}
        for delta, w in zip(filtered_deltas, filtered_weights):
            for k, v in delta.items():
                acc[k] = acc.get(k, torch.zeros_like(v)) + v * w
        total_w = sum(filtered_weights)
        for k in acc:
            acc[k] /= total_w

    elif byzantine_method == "krum":
        # Multi-Krum: returns already-averaged dict
        acc, _ = byzantine_krum(deltas_copy, weights, n_malicious=n_malicious)

    elif byzantine_method == "trimmed_mean":
        # Trimmed mean per parameter element
        acc = byzantine_trimmed_mean(deltas_copy, weights, trim_fraction=TRIMMED_MEAN_TRIM)

    else:
        raise ValueError(f"Unknown Byzantine method: {byzantine_method}")

    # FIX: Sanitize accumulated deltas before computing norm or applying.
    # NaN/Inf can arise from dtype mismatches (float32 delta + bfloat16 model).
    for k in acc:
        acc[k] = acc[k].nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)
        # Clamp to prevent extreme values from destabilizing the model
        acc[k] = torch.clamp(acc[k], min=-10.0, max=10.0)

    # Apply accumulated delta to server model.
    # FIX: Use SERVER_LR=1.0 (the natural choice) UNLESS delta_norm is
    # very large (indicating a potential Byzantine attack or numerical issue).
    # The old formula: SERVER_LR = min(0.1, 0.01/delta_norm) was backwards —
    # it made the update TOO SMALL when deltas were small (normal training)
    # and could make SERVER_LR=0.1 even for moderately-sized legitimate updates.
    delta_norm = math.sqrt(sum(v.float().pow(2).sum().item() for v in acc.values()))
    # Cap SERVER_LR at 1.0 to prevent runaway updates; in practice 1.0 works
    # because deltas are the parameter updates (not raw gradients).
    SERVER_LR = min(1.0, 0.1 / math.sqrt(max(delta_norm, 1e-8))) if delta_norm > 1e-6 else 1.0
    # DEBUG: log aggregation magnitude
    _dbg_norms = [v.float().norm().item() for v in acc.values()]
    logger.info(f"  [AGG] acc norm avg={sum(_dbg_norms)/len(_dbg_norms):.6f} max={max(_dbg_norms):.6f} SERVER_LR={SERVER_LR:.6f} (delta_norm={delta_norm:.6f})")
    for full_name, lora_layer in wrapper.lora_layers.items():
        for suffix in ["lora_A", "lora_B"]:
            key = f"{full_name}.{suffix}"
            if key in acc:
                with torch.no_grad():
                    _before_norm = getattr(lora_layer, suffix).data.norm().item()
                    getattr(lora_layer, suffix).add_(acc[key] * SERVER_LR)
                    _after_norm = getattr(lora_layer, suffix).data.norm().item()
                    if round_num == 1 and full_name == list(wrapper.lora_layers.keys())[0]:
                        logger.info(f"  [AGG]   {suffix} {full_name[:40]}: {_before_norm:.6f} -> {_after_norm:.6f} (delta={_after_norm-_before_norm:+.6f})")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_byzantine_experiment(
    byzantine_method: str = "none",
    n_malicious: int = 0,
    run_name: str = "",
    exp_tag: str = "suite3",
) -> Dict[str, Any]:
    """
    Run a federated experiment with optional Byzantine attack and defense.
    """
    attack_label = f"label_flip_10x" if n_malicious > 0 else "none"
    label = run_name or f"{exp_tag}_{byzantine_method}_mal{n_malicious}_{attack_label}"
    dist_label = "IID (random)"

    logger.info("=" * 70)
    logger.info(f"BYZANTINE STRESS TEST - {label}")
    logger.info(f"  Byzantine method: {byzantine_method}")
    logger.info(f"  Malicious clients: {n_malicious}")
    logger.info(f"  Attack type: {attack_label}")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Clients: {NUM_CLIENTS}, Rounds: {NUM_ROUNDS}")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    logger.info("\n[SETUP] Loading model and tokenizer...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, trust_remote_code=True, torch_dtype=torch.float32,
    )
    num_layers = config.num_hidden_layers
    logger.info(f"  Loaded {MODEL_ID} in {time.time()-t0:.1f}s")
    logger.info(f"  Layers: {num_layers}, Hidden: {config.hidden_size}, Vocab: {config.vocab_size}")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    logger.info("\n[SETUP] Loading wikitext dataset...")
    t0 = time.time()
    train_texts, test_texts = load_wikitext(tokenizer)
    logger.info(f"  Loaded in {time.time()-t0:.1f}s")

    client_partitions = partition_data(train_texts, NUM_CLIENTS)
    logger.info(f"  Partitions: {[len(p) for p in client_partitions]} texts each")
    test_enc = tokenize_texts(tokenizer, test_texts, MAX_SEQ_LEN)
    test_enc = {k: v.clone() for k, v in test_enc.items()}

    # -------------------------------------------------------------------------
    # Apply LoRA
    # -------------------------------------------------------------------------
    logger.info("\n[SETUP] Applying LoRA...")
    wrapper = LoraAppliedModel(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    wrapper.apply_lora()

    # DEBUG: check ppl BEFORE any training
    logger.info("\n[DEBUG] ppl BEFORE training (after LoRA init)...")
    ppl_before = compute_perplexity(model, test_enc)
    logger.info(f"  ppl = {ppl_before:.2e}")

    # -------------------------------------------------------------------------
    # Federated loop
    # -------------------------------------------------------------------------
    rounds_log = []
    comm_tensors_per_round = []

    for r in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- Round {r}/{NUM_ROUNDS} ---")
        t0 = time.time()

        round_base = snapshot_lora_state(wrapper)
        deltas = []
        weights = []
        round_losses = []
        malicious_injected = []

        for cid_idx, texts in enumerate(client_partitions):
            client_id = f"client-{cid_idx+1}"

            # LISA layer selection (always use LISA to match Suite 3 spec)
            seed = r * 100 + cid_idx * 17 + 42
            selected = lisa_select_layers(num_layers, seed=seed)

            # Train client
            result = train_client(model, tokenizer, wrapper, texts, r, client_id, selected_layers=selected)
            round_losses.append(result["avg_train_loss"])
            weights.append(len(texts))

            state_after = snapshot_lora_state(wrapper)
            delta = compute_deltas(round_base, state_after)

            # --- Byzantine attack injection ---
            is_malicious = cid_idx < n_malicious
            if is_malicious:
                delta_adv = inject_adversarial_delta(delta)
                malicious_injected.append(client_id)
                delta = delta_adv

            deltas.append(delta)

            # Reset to round base for next client
            restore_lora_state(wrapper, round_base)

        if malicious_injected:
            logger.info(f"  Malicious clients injected: {malicious_injected}")

        # -------------------------------------------------------------------------
        # Aggregate with Byzantine defense
        # -------------------------------------------------------------------------
        aggregate_deltas(
            deltas, weights, wrapper,
            byzantine_method=byzantine_method,
            n_malicious=n_malicious,
            round_num=r,
        )

        # Evaluate perplexity
        ppl = compute_perplexity(model, test_enc)
        avg_loss = sum(round_losses) / len(round_losses)
        comm_tensors = len(wrapper.lora_layers) * 2 * len(client_partitions)
        elapsed = time.time() - t0

        logger.info(
            f"  Round {r}: ppl={ppl:.2f} | loss={avg_loss:.4f} | "
            f"comm={comm_tensors} tensors | {elapsed:.1f}s"
        )

        rounds_log.append({
            "round": r,
            "avg_test_perplexity": ppl,
            "avg_train_loss": avg_loss,
            "total_comm_tensors": comm_tensors,
            "malicious_injected": malicious_injected,
        })
        comm_tensors_per_round.append(comm_tensors)

        gc.collect()

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    final_ppl = rounds_log[-1]["avg_test_perplexity"] if rounds_log else float("inf")

    results = {
        "experiment": "Byzantine Resilience Stress Test (Suite 3)",
        "run_label": label,
        "byzantine_method": byzantine_method,
        "n_malicious": n_malicious,
        "attack_type": attack_label,
        "model": MODEL_ID,
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "lisa_bottom": LISA_BOTTOM,
        "lisa_top": LISA_TOP,
        "lisa_middle": LISA_MIDDLE,
        "lora_rank": LORA_RANK,
        "rounds": rounds_log,
        "final_perplexity": final_ppl,
        "perplexity_curve": [r["avg_test_perplexity"] for r in rounds_log],
        "comm_tensors_per_round": comm_tensors_per_round,
        "total_comm_tensors": sum(comm_tensors_per_round),
    }

    out_path = EVAL_DIR / f"{label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {out_path}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Byzantine Resilience Stress Test")
    parser.add_argument("--byzantine", type=str, default="none",
                        choices=["none", "norm", "krum", "trimmed_mean"],
                        help="Byzantine defense method")
    parser.add_argument("--malicious-clients", type=int, default=0,
                        help="Number of malicious clients (0 = no attack)")
    parser.add_argument("--tag", type=str, default="suite3",
                        help="Experiment tag for output filename")
    parser.add_argument("--name", type=str, default="",
                        help="Custom run name (auto-generated if empty)")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    attack_label = f"label_flip_10x" if args.malicious_clients > 0 else "none"
    run_name = args.name or f"{args.tag}_{args.byzantine}_mal{args.malicious_clients}_{attack_label}"

    results = run_byzantine_experiment(
        byzantine_method=args.byzantine,
        n_malicious=args.malicious_clients,
        run_name=run_name,
        exp_tag=args.tag,
    )

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULT")
    logger.info("=" * 70)
    logger.info(f"  Method: {args.byzantine} | Malicious: {args.malicious_clients}")
    logger.info(f"  Final perplexity: {results['final_perplexity']:.2f}")
    logger.info(f"  Perplexity curve: {[f'{p:.2f}' for p in results['perplexity_curve']]}")
    logger.info(f"  Total comm tensors: {results['total_comm_tensors']}")
