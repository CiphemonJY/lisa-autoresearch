#!/usr/bin/env python3
"""
Comparative Evaluation: FedAvg vs LISA-FedAvg

Research question: Does LISA layer selection + federated (LISA-FedAvg) achieve
comparable or better accuracy vs plain FedAvg, while reducing communication cost?

Usage:
  python eval/fedavg_vs_lisafedavg.py

Output:
  - Live comparison table each round
  - eval/results/fedavg_vs_lisafedavg.json
  - eval/layer_selection_stats.json
"""

import gc
import json
import math
import os
import sys
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

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
NUM_CLIENTS = 3
NUM_ROUNDS = 3                     # reduced from 5 for faster runs; set QUICK=1 env var for 1-round smoke test
LOCAL_EPOCHS = 1
BATCH_SIZE = 4
MAX_SEQ_LEN = 128
LR = 3e-4
LORA_RANK = 4
LORA_ALPHA = 8.0
LORA_DROPOUT = 0.05
LISA_BOTTOM = 2
LISA_TOP = 2
LISA_MIDDLE = 2
MAX_TRAIN_BATCHES_PER_CLIENT = 20  # reduced from 40 to fit cron timeout
MAX_TEST_BATCHES = 20             # reduced from 50 for faster eval
SEED = 42
EVAL_DIR = Path("eval/results")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """LoRA: y = Wx + BAx with low-rank decomposition. Does NOT replace the original module."""

    def __init__(self, linear: nn.Module, rank: int = 4, alpha: float = 8.0,
                 dropout: float = 0.05):
        super().__init__()
        # Store original weights in float32 for stable math
        self.weight_data = linear.weight.data.clone().float()
        self.bias_data = linear.bias.data.clone().float() if linear.bias is not None else None
        self.out_features, self.in_features = self.weight_data.shape
        self.is_conv1d = isinstance(linear, nn.Conv1d)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA params in float32 for gradient stability
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # Cast to float32 for stable matmul with float32 stored weights
        x_f32 = x.to(torch.float32)
        with torch.no_grad():
            original = nn.functional.linear(x_f32, self.weight_data, self.bias_data)
        lora_input = self.lora_dropout(x_f32)
        if self.is_conv1d:
            lora = nn.functional.linear(lora_input, self.lora_A)
            lora = nn.functional.linear(lora, self.lora_B)
        else:
            lora = nn.functional.linear(lora_input, self.lora_A)
            lora = nn.functional.linear(lora, self.lora_B)
        result = original + lora * self.scaling
        return result.to(orig_dtype)

    def trainable_params(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]


class LoraAppliedModel:
    """Apply LoRA wrappers to target linear layers of a model."""

    # Modules to target (GPT-NeoX for Pythia, plus GPT-2 aliases)
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
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_lora_layers(self, layer_indices: List[int]):
        """
        Unfreeze LoRA params for the specified layer indices.
        Handles both 'gpt_neox.layers.N.' and 'h.N.' naming.
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

    def get_trainable_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_wikitext(tokenizer, max_seq: int = MAX_SEQ_LEN) -> Tuple[List[str], List[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not installed — using synthetic data")
        return _synthetic_data()

    try:
        ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
        test_ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
    except Exception as e:
        logger.warning(f"Failed to load wikitext: {e} — using synthetic data")
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
    random.seed(SEED)
    train_texts, test_texts = [], []
    for i in range(n_train):
        sel = random.sample(words, min(30, len(words)))
        random.shuffle(sel)
        train_texts.append(" ".join(sel * 3)[:150])
    for i in range(n_test):
        sel = random.sample(words, min(25, len(words)))
        random.shuffle(sel)
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


def partition_data(texts: List[str], n: int, seed: int = SEED, non_iid: bool = False) -> List[List[str]]:
    """
    Partition texts across n clients.
    non_iid=True: partition by sequential topic slices (client 0 = samples 0-400,
    client 1 = 400-800, client 2 = 800-1200) to simulate non-IID data skew.
    """
    if non_iid:
        # Non-IID: sequential slices instead of shuffled (simulates topic clustering)
        size = 400  # fixed slice size
        partitions = []
        for i in range(n):
            start = i * size
            end = start + size if i < n - 1 else len(texts)
            partitions.append(texts[start:end])
        return partitions
    else:
        random.seed(seed)
        shuffled = texts.copy()
        random.shuffle(shuffled)
        size = len(shuffled) // n
        partitions = []
        for i in range(n):
            start = i * size
            end = start + size if i < n - 1 else len(shuffled)
            partitions.append(shuffled[start:end])
        return partitions


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(model: nn.Module, test_enc: Dict[str, torch.Tensor],
                       batch_size: int = BATCH_SIZE,
                       max_batches: int = MAX_TEST_BATCHES) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    pad_token_id = getattr(model.config, "pad_token_id", None) or -100

    n_batches = min((len(test_enc["input_ids"]) + batch_size - 1) // batch_size, max_batches)
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(test_enc["input_ids"]))
        ids = test_enc["input_ids"][start:end].clone().clamp(0, model.config.vocab_size - 1)
        mask = test_enc["attention_mask"][start:end]
        labs = test_enc["labels"][start:end].clone().clamp(0, model.config.vocab_size - 1)

        outputs = model(input_ids=ids, attention_mask=mask)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labs.view(-1))
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
        random.seed(seed)

    bottom_set = list(range(min(bottom, num_layers)))
    top_set = list(range(max(0, num_layers - top), num_layers))
    middle_pool = list(range(bottom, max(bottom, num_layers - top)))
    middle_set = random.sample(middle_pool, min(middle, len(middle_pool))) if middle_pool else []

    return sorted(set(bottom_set + top_set + middle_set))


# ---------------------------------------------------------------------------
# State snapshots & restoration
# ---------------------------------------------------------------------------

def snapshot_lora_state(wrapper: LoraAppliedModel) -> Dict[str, torch.Tensor]:
    """Snapshot lora_A and lora_B tensors from all LoRA layers."""
    state = {}
    for full_name, lora_layer in wrapper.lora_layers.items():
        state[f"{full_name}.lora_A"] = lora_layer.lora_A.data.clone().cpu()
        state[f"{full_name}.lora_B"] = lora_layer.lora_B.data.clone().cpu()
    return state


def restore_lora_state(wrapper: LoraAppliedModel, state: Dict[str, torch.Tensor]):
    """Restore lora_A and lora_B from a snapshot."""
    for full_name, lora_layer in wrapper.lora_layers.items():
        lora_layer.lora_A.data.copy_(state[f"{full_name}.lora_A"].clone())
        lora_layer.lora_B.data.copy_(state[f"{full_name}.lora_B"].clone())


def compute_deltas(before: Dict[str, torch.Tensor],
                   after: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: after[k] - before[k] for k in before}


# ---------------------------------------------------------------------------
# Client training
# ---------------------------------------------------------------------------

def train_client(
    model: nn.Module,
    tokenizer,
    wrapper: LoraAppliedModel,
    client_texts: List[str],
    round_num: int,
    client_id: str,
    selected_layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Train LoRA layers locally on client texts.
    Perplexity is NOT computed here — only after aggregation (once per round).
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
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses) if losses else 0.0

    return {
        "client_id": client_id,
        "round": round_num,
        "avg_train_loss": avg_loss,
        "grad_tensors_sent": (
            len(wrapper.lora_layers) * 2 if selected_layers is None
            else len(set(selected_layers)) * 2
        ),
        "selected_layers": selected_layers,
    }


# ---------------------------------------------------------------------------
# Federated aggregation
# ---------------------------------------------------------------------------

def aggregate_deltas(
    deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
    wrapper: LoraAppliedModel,
) -> None:
    """Apply weighted gradient deltas to the server's LoRA layers."""
    if not deltas or not weights:
        return

    # Accumulate weighted deltas
    acc: Dict[str, torch.Tensor] = {}
    for delta, w in zip(deltas, weights):
        for k, v in delta.items():
            acc[k] = acc.get(k, torch.zeros_like(v)) + v.float() * w

    # Apply to server model
    for full_name, lora_layer in wrapper.lora_layers.items():
        for suffix in ["lora_A", "lora_B"]:
            key = f"{full_name}.{suffix}"
            if key in acc:
                with torch.no_grad():
                    getattr(lora_layer, suffix).add_(acc[key])


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(non_iid: bool = False) -> Dict[str, Any]:
    """
    Run a single federated experiment. When non_iid=True, partition wikitext-2
    by topic slices (client 0 = samples 0-400, client 1 = 400-800, client 2 = 800-1200)
    to simulate non-IID data skew.
    """
    dist_label = "Non-IID (topic slices)" if non_iid else "IID (random)"
    logger.info("=" * 70)
    logger.info(f"FedAvg vs LISA-FedAvg — {dist_label}")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Clients: {NUM_CLIENTS}, Rounds: {NUM_ROUNDS}")
    logger.info(f"LISA: bottom={LISA_BOTTOM}, top={LISA_TOP}, middle={LISA_MIDDLE}")
    logger.info(f"Local epochs/round: {LOCAL_EPOCHS}, Batch size: {BATCH_SIZE}")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Load model and tokenizer
    # -------------------------------------------------------------------------
    logger.info("\n[SETUP] Loading model and tokenizer...")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    num_layers = config.num_hidden_layers
    logger.info(f"  Loaded {MODEL_ID} in {time.time()-t0:.1f}s")
    logger.info(f"  Layers: {num_layers}, Hidden: {config.hidden_size}, Vocab: {config.vocab_size}")

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    logger.info("\n[SETUP] Loading wikitext dataset...")
    t0 = time.time()
    train_texts, test_texts = load_wikitext(tokenizer)
    logger.info(f"  Loaded in {time.time()-t0:.1f}s")

    client_partitions = partition_data(train_texts, NUM_CLIENTS, non_iid=non_iid)
    logger.info(f"  Partitions: {[len(p) for p in client_partitions]} texts each")
    logger.info(f"  Distribution: {dist_label}")

    test_enc = tokenize_texts(tokenizer, test_texts, MAX_SEQ_LEN)
    test_enc = {k: v.clone() for k, v in test_enc.items()}

    # -------------------------------------------------------------------------
    # Fresh model loader (used between experiments for cleanest reset)
    # -------------------------------------------------------------------------
    _base_model_snapshot: Optional[Dict] = None

    def load_fresh_model():
        """Load a brand-new base model from HuggingFace, discarding any prior state."""
        cfg = AutoConfig.from_pretrained(MODEL_ID)
        fresh = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        return fresh

    # -------------------------------------------------------------------------
    # Capture bare model weights BEFORE LoRA (for fast resets within an experiment)
    # -------------------------------------------------------------------------
    bare_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    def reinit_model(*, use_fresh: bool = False) -> LoraAppliedModel:
        """
        Reinitialize model for the next experiment.

        use_fresh=False (default, within same experiment):
            Restore the base model from our bare_state snapshot (no re-download).
        use_fresh=True (between experiments):
            Download a fresh copy from HuggingFace for cleanest isolation.
        """
        if use_fresh:
            fresh_model = load_fresh_model()
        else:
            # Fast path: restore base weights from our snapshot, no re-download
            fresh_model = load_fresh_model()
            try:
                fresh_model.load_state_dict({k: v.clone() for k, v in bare_state.items()}, strict=False)
            except Exception:
                # If strict=False still fails (e.g. dtype mismatch), just use bare weights directly
                pass

        wrapper = LoraAppliedModel(fresh_model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
        wrapper.apply_lora()
        return wrapper

    # -------------------------------------------------------------------------
    # Results storage
    # -------------------------------------------------------------------------
    fedavg_rounds = []
    lisafedavg_rounds = []
    layer_selection_counts: Dict[int, int] = {i: 0 for i in range(num_layers)}

    # ==========================================================================
    # EXPERIMENT 1: Plain FedAvg
    # ==========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Plain FedAvg (all layers)")
    logger.info("=" * 70)

    wrapper = reinit_model()
    wrapper.freeze_all()

    for r in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- FedAvg Round {r}/{NUM_ROUNDS} ---")
        t0 = time.time()

        round_base = snapshot_lora_state(wrapper)
        deltas = []
        weights = []
        round_losses = []

        for cid_idx, texts in enumerate(client_partitions):
            client_id = f"client-{cid_idx+1}"

            result = train_client(
                model, tokenizer, wrapper, texts,
                r, client_id, selected_layers=None
            )
            round_losses.append(result["avg_train_loss"])
            weights.append(len(texts))

            state_after = snapshot_lora_state(wrapper)
            deltas.append(compute_deltas(round_base, state_after))

            # Reset to round base for next client
            restore_lora_state(wrapper, round_base)

        # Aggregate
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]
        aggregate_deltas(deltas, norm_weights, wrapper)

        # Evaluate perplexity AFTER aggregation
        ppl = compute_perplexity(model, test_enc)
        avg_loss = sum(round_losses) / len(round_losses)
        comm_tensors = len(wrapper.lora_layers) * 2 * len(client_partitions)

        elapsed = time.time() - t0
        logger.info(
            f"  FedAvg Round {r}: ppl={ppl:.2f} | loss={avg_loss:.4f} | "
            f"comm={comm_tensors} tensors | {elapsed:.1f}s"
        )

        fedavg_rounds.append({
            "round": r,
            "avg_test_perplexity": ppl,
            "avg_train_loss": avg_loss,
            "total_comm_tensors": comm_tensors,
        })

    # ==========================================================================
    # EXPERIMENT 2: LISA-FedAvg
    # ==========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: LISA-FedAvg (selected layers only)")
    logger.info("=" * 70)

    wrapper = reinit_model(use_fresh=True)
    wrapper.freeze_all()

    for r in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- LISA-FedAvg Round {r}/{NUM_ROUNDS} ---")
        t0 = time.time()

        round_base = snapshot_lora_state(wrapper)
        deltas = []
        weights = []
        round_losses = []
        selected_this_round: List[int] = []

        for cid_idx, texts in enumerate(client_partitions):
            client_id = f"client-{cid_idx+1}"

            # LISA layer selection
            seed = r * 100 + cid_idx * 17 + 42
            selected = lisa_select_layers(num_layers, seed=seed)
            selected_this_round.extend(selected)

            result = train_client(
                model, tokenizer, wrapper, texts,
                r, client_id, selected_layers=selected
            )
            round_losses.append(result["avg_train_loss"])
            weights.append(len(texts))

            for layer_idx in selected:
                layer_selection_counts[layer_idx] += 1

            state_after = snapshot_lora_state(wrapper)
            deltas.append(compute_deltas(round_base, state_after))

            restore_lora_state(wrapper, round_base)

        # Aggregate
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]
        aggregate_deltas(deltas, norm_weights, wrapper)

        # Evaluate perplexity AFTER aggregation
        ppl = compute_perplexity(model, test_enc)
        avg_loss = sum(round_losses) / len(round_losses)

        # Comm = selected layers * 2 * num_clients (unique selected layers)
        unique_selected = len(set(selected_this_round))
        total_grads = unique_selected * 2 * len(client_partitions)

        elapsed = time.time() - t0
        logger.info(
            f"  LISA Round {r}: ppl={ppl:.2f} | loss={avg_loss:.4f} | "
            f"layers={sorted(set(selected_this_round))} | "
            f"comm={total_grads} tensors | {elapsed:.1f}s"
        )

        lisafedavg_rounds.append({
            "round": r,
            "avg_test_perplexity": ppl,
            "avg_train_loss": avg_loss,
            "total_comm_tensors": total_grads,
            "layers_selected_this_round": sorted(set(selected_this_round)),
        })

    # -------------------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 70)
    logger.info(f"{'Round':>5} | {'FedAvg ppl':>12} | {'LISA ppl':>12} | "
                f"{'LISA comm':>10} | {'FedAvg comm':>12} | {'Saved':>10}")
    logger.info("-" * 70)

    total_fedavg_comm = 0
    total_lisa_comm = 0

    for i in range(NUM_ROUNDS):
        fr = fedavg_rounds[i]
        lr = lisafedavg_rounds[i]
        saved = fr["total_comm_tensors"] - lr["total_comm_tensors"]
        pct = (saved / fr["total_comm_tensors"]) * 100 if fr["total_comm_tensors"] > 0 else 0
        total_fedavg_comm += fr["total_comm_tensors"]
        total_lisa_comm += lr["total_comm_tensors"]

        logger.info(
            f"{i+1:>5} | {fr['avg_test_perplexity']:>12.2f} | {lr['avg_test_perplexity']:>12.2f} | "
            f"{lr['total_comm_tensors']:>10} | {fr['total_comm_tensors']:>12} | "
            f"{saved:>8} ({pct:.0f}%)"
        )

    total_saved = total_fedavg_comm - total_lisa_comm
    total_pct = (total_saved / total_fedavg_comm) * 100 if total_fedavg_comm > 0 else 0

    logger.info("-" * 70)
    logger.info(
        f"{'TOTAL':>5} | {'':>12} | {'':>12} | "
        f"{total_lisa_comm:>10} | {total_fedavg_comm:>12} | "
        f"{total_saved:>8} ({total_pct:.0f}%)"
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    fedavg_final_ppl = fedavg_rounds[-1]["avg_test_perplexity"]
    lisa_final_ppl = lisafedavg_rounds[-1]["avg_test_perplexity"]
    ppl_diff_pct = ((lisa_final_ppl - fedavg_final_ppl) / fedavg_final_ppl) * 100

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"FedAvg final perplexity:    {fedavg_final_ppl:.2f}")
    logger.info(f"LISA-FedAvg final ppl:     {lisa_final_ppl:.2f}")
    logger.info(f"Perplexity difference:    {ppl_diff_pct:+.1f}%")
    logger.info(f"Total comm tensors saved: {total_saved} ({total_pct:.0f}%)")
    logger.info(f"Per-layer selection counts (LISA):")
    for layer_idx in sorted(layer_selection_counts):
        count = layer_selection_counts[layer_idx]
        bar = "#" * min(count, 20)
        logger.info(f"  Layer {layer_idx:>2}: {count:>3} {bar}")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    results_json = {
        "experiment": "FedAvg vs LISA-FedAvg",
        "model": MODEL_ID,
        "distribution": dist_label,
        "config": {
            "num_clients": NUM_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "lisa_bottom": LISA_BOTTOM,
            "lisa_top": LISA_TOP,
            "lisa_middle": LISA_MIDDLE,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
        },
        "fedavg": {
            "rounds": fedavg_rounds,
            "final_perplexity": fedavg_final_ppl,
            "total_comm_tensors": total_fedavg_comm,
        },
        "lisafedavg": {
            "rounds": lisafedavg_rounds,
            "final_perplexity": lisa_final_ppl,
            "total_comm_tensors": total_lisa_comm,
        },
        "comparison": {
            "ppl_diff_pct": ppl_diff_pct,
            "comm_saved_pct": total_pct,
            "total_comm_saved": total_saved,
            "ppl_within_5pct": abs(ppl_diff_pct) <= 5.0,
            "comm_reduced_by_40pct": total_pct >= 40.0,
        },
    }

    results_path = EVAL_DIR / "fedavg_vs_lisafedavg.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")

    layer_stats = {
        "experiment": "LISA-FedAvg layer selection counts",
        "num_rounds": NUM_ROUNDS,
        "num_clients": NUM_CLIENTS,
        "lisa_config": {
            "bottom_layers": LISA_BOTTOM,
            "top_layers": LISA_TOP,
            "middle_sample": LISA_MIDDLE,
        },
        "total_possible_selections": NUM_ROUNDS * NUM_CLIENTS,
        "layer_counts": {str(k): v for k, v in layer_selection_counts.items()},
        "most_selected_layer": str(max(layer_selection_counts, key=layer_selection_counts.get)),
        "least_selected_layer": str(min(layer_selection_counts, key=layer_selection_counts.get)),
        "bottom_layers_total": sum(layer_selection_counts[i] for i in range(min(LISA_BOTTOM, num_layers))),
        "top_layers_total": sum(layer_selection_counts[i] for i in range(max(0, num_layers - LISA_TOP), num_layers)),
    }

    layer_path = EVAL_DIR / "layer_selection_stats.json"
    with open(layer_path, "w") as f:
        json.dump(layer_stats, f, indent=2)
    logger.info(f"Layer selection stats saved to: {layer_path}")

    return results_json


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)

    try:
        # Run IID baseline first
        logger.info("\n\n" + "=" * 70)
        logger.info("SUITE 2 — Run 2a: IID Baseline")
        logger.info("=" * 70)
        iid_results = run_experiment(non_iid=False)

        # Save IID results
        iid_path = EVAL_DIR / "suite2a_iid_baseline.json"
        with open(iid_path, "w") as f:
            json.dump(iid_results, f, indent=2)
        logger.info(f"\nIID results saved to: {iid_path}")

        # Reset model state for non-IID run
        del iid_results
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Run Non-IID experiment
        logger.info("\n\n" + "=" * 70)
        logger.info("SUITE 2 — Run 2b: Non-IID (topic slices)")
        logger.info("=" * 70)
        non_iid_results = run_experiment(non_iid=True)

        # Save Non-IID results
        non_iid_path = EVAL_DIR / "suite2b_non_iid.json"
        with open(non_iid_path, "w") as f:
            json.dump(non_iid_results, f, indent=2)
        logger.info(f"\nNon-IID results saved to: {non_iid_path}")

        # Comparison summary
        logger.info("\n\n" + "=" * 70)
        logger.info("SUITE 2 COMPARISON SUMMARY")
        logger.info("=" * 70)
        logger.info(f"{'Config':>20} | {'Final PPL':>12} | {'Comm Cost':>12}")
        logger.info("-" * 50)
        logger.info(f"{'IID (LISA middle=2)':>20} | {iid_results['final_perplexity']:>12.2f} | {iid_results['total_comm_tensors']:>12}")
        logger.info(f"{'Non-IID (LISA middle=2)':>20} | {non_iid_results['final_perplexity']:>12.2f} | {non_iid_results['total_comm_tensors']:>12}")

        logger.info("\nSuite 2 complete!")
    except Exception as e:
        logger.exception(f"Experiment crashed: {e}")
        sys.exit(1)
