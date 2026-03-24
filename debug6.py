#!/usr/bin/env python3
"""
debug6.py - Minimal pinpoint test for catastrophic divergence in byzantine_stress_test.py

1 client, 1 round, 20 batches. Reports ppl before/after, delta norms for
lora_A and lora_B separately, and LoRA weight norms before/after.

If divergence reproduces here, the bug is in byzantine_stress_test.py's code,
not the multi-client parameters.
"""
import gc, json, math, os, random, sys, time
from pathlib import Path

import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---------------------------------------------------------------------------
# Config - SAME as byzantine_stress_test.py
# ---------------------------------------------------------------------------
MODEL_ID = "EleutherAI/pythia-70m"
NUM_CLIENTS = 1          # MINIMAL: 1 client
NUM_ROUNDS = 1           # MINIMAL: 1 round
LOCAL_EPOCHS = 1
BATCH_SIZE = 4
MAX_SEQ_LEN = 128
LR = 8e-4
LORA_RANK = 4
LORA_ALPHA = 8.0
LORA_DROPOUT = 0.05
LISA_BOTTOM = 2
LISA_TOP = 2
LISA_MIDDLE = 2
MAX_TRAIN_BATCHES_PER_CLIENT = 20   # Same cap as byzantine
MAX_TEST_BATCHES = 20
SEED = 42
SERVER_LR = 0.1          # Will be overridden by adaptive formula below

# ---------------------------------------------------------------------------
# LoRA - EXACT copy from byzantine_stress_test.py (LoRALinear + LoraAppliedModel)
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

    def trainable_params(self) -> list:
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
        self.lora_layers: dict = {}

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
        print(f"  [LoRA] applied to {count} layers (rank={self.rank})")
        return count

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_lora_layers(self, layer_indices: list):
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
# Helpers
# ---------------------------------------------------------------------------
def load_wikitext(tokenizer, max_seq: int = MAX_SEQ_LEN):
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
    print(f"  wikitext: {len(train_texts)} train, {len(test_texts)} test lines")
    return train_texts, test_texts


def _synthetic_data(n_train: int = 600, n_test: int = 100):
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
    print(f"  synthetic: {len(train_texts)} train, {len(test_texts)} test lines")
    return train_texts, test_texts


def tokenize_texts(tokenizer, texts, max_seq):
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


def partition_data(texts, n, seed=SEED):
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


def lisa_select_layers(num_layers, bottom=LISA_BOTTOM, top=LISA_TOP,
                       middle=LISA_MIDDLE, seed=None):
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
# Perplexity
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_perplexity(model, test_enc, batch_size=BATCH_SIZE):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    pad_token_id = getattr(model.config, "pad_token_id", None) or -100
    n_batches = min((len(test_enc["input_ids"]) + batch_size - 1) // batch_size,
                    MAX_TEST_BATCHES)
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(test_enc["input_ids"]))
        ids = test_enc["input_ids"][start:end].clone().clamp(0, model.config.vocab_size - 1)
        mask = test_enc["attention_mask"][start:end]
        labs = test_enc["labels"][start:end].clone()
        # Replace padding tokens in labels with -100 so they don't pollute loss
        labs = labs.where(labs != pad_token_id, torch.tensor(-100, device=labs.device))
        outputs = model(input_ids=ids, attention_mask=mask)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labs.view(-1))
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1)) if total_tokens > 0 else float("inf")


# ---------------------------------------------------------------------------
# State snapshots
# ---------------------------------------------------------------------------
def snapshot_lora_state(wrapper):
    state = {}
    for full_name, lora_layer in wrapper.lora_layers.items():
        state[f"{full_name}.lora_A"] = lora_layer.lora_A.data.clone().cpu()
        state[f"{full_name}.lora_B"] = lora_layer.lora_B.data.clone().cpu()
    return state


def restore_lora_state(wrapper, state):
    for full_name, lora_layer in wrapper.lora_layers.items():
        lora_layer.lora_A.data.copy_(state[f"{full_name}.lora_A"].clone())
        lora_layer.lora_B.data.copy_(state[f"{full_name}.lora_B"].clone())


def compute_deltas(before, after):
    return {k: after[k] - before[k] for k in before}


# ---------------------------------------------------------------------------
# Train one client - EXACT copy from byzantine_stress_test.py
# ---------------------------------------------------------------------------
def train_client(model, tokenizer, wrapper, client_texts, round_num, client_id,
                 selected_layers=None):
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
    n_batches = min((len(enc["input_ids"]) + BATCH_SIZE - 1) // BATCH_SIZE,
                    MAX_TRAIN_BATCHES_PER_CLIENT)
    print(f"    n_batches={n_batches}, texts={len(client_texts)}, max_possible={len(enc['input_ids'])//BATCH_SIZE}")

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
                print(f"    WARNING: NaN loss at batch {i}, skipping")
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
        "selected_layers": selected_layers,
    }


# ---------------------------------------------------------------------------
# Aggregate - EXACT copy from byzantine_stress_test.py
# ---------------------------------------------------------------------------
def aggregate_deltas(deltas, weights, wrapper, round_num=0):
    if not deltas or not weights:
        return

    deltas_copy = [{k: v.float() for k, v in d.items()} for d in deltas]

    # Standard weighted average (no Byzantine - this is a single-client test)
    acc = {}
    for delta, w in zip(deltas_copy, weights):
        for k, v in delta.items():
            acc[k] = acc.get(k, torch.zeros_like(v)) + v * w
    total_w = sum(weights)
    for k in acc:
        acc[k] /= total_w

    # --- DIAGNOSTIC: delta norms per layer and per A/B ---
    print(f"\n  [DIAG] Delta norms BEFORE aggregation (per layer, A vs B):")
    # Group by layer
    layer_keys = {}
    for k in sorted(acc.keys()):
        parts = k.rsplit(".lora_", 1)
        layer_name = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        if layer_name not in layer_keys:
            layer_keys[layer_name] = {}
        layer_keys[layer_name][suffix] = acc[k].norm().item()

    for layer_name, suffixes in layer_keys.items():
        a_norm = suffixes.get("A", 0.0)
        b_norm = suffixes.get("B", 0.0)
        total_norm = math.sqrt(a_norm**2 + b_norm**2)
        short_name = layer_name[:50]
        print(f"    {short_name}: A={a_norm:.6f}  B={b_norm:.6f}  combined={total_norm:.6f}")

    # Apply with SERVER_LR
    print(f"\n  [DIAG] LoRA weight norms BEFORE aggregation:")
    for full_name, lora_layer in wrapper.lora_layers.items():
        a_n = lora_layer.lora_A.data.norm().item()
        b_n = lora_layer.lora_B.data.norm().item()
        print(f"    {full_name[:60]}: A={a_n:.6f}  B={b_n:.6f}")

    print(f"\n  [AGG] SERVER_LR={SERVER_LR}, applying delta...")
    _dbg_norms = [v.float().norm().item() for v in acc.values()]
    print(f"  [AGG] acc norm avg={sum(_dbg_norms)/len(_dbg_norms):.6f} max={max(_dbg_norms):.6f}")

    for full_name, lora_layer in wrapper.lora_layers.items():
        for suffix in ["lora_A", "lora_B"]:
            key = f"{full_name}.{suffix}"
            if key in acc:
                with torch.no_grad():
                    _before_norm = getattr(lora_layer, suffix).data.norm().item()
                    getattr(lora_layer, suffix).add_(acc[key] * SERVER_LR)
                    _after_norm = getattr(lora_layer, suffix).data.norm().item()
                    delta_norm = _after_norm - _before_norm
                    if round_num == 1:
                        print(f"  [AGG]   {suffix} {full_name[:50]}: {_before_norm:.6f} -> {_after_norm:.6f} (d={delta_norm:+.6f})")

    print(f"\n  [DIAG] LoRA weight norms AFTER aggregation:")
    for full_name, lora_layer in wrapper.lora_layers.items():
        a_n = lora_layer.lora_A.data.norm().item()
        b_n = lora_layer.lora_B.data.norm().item()
        print(f"    {full_name[:60]}: A={a_n:.6f}  B={b_n:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DEBUG6 - Minimal 1-client pinpoint test (exact byzantine code)")
    print(f"  LR={LR}, SERVER_LR={SERVER_LR}, LORA_RANK={LORA_RANK}")
    print(f"  MAX_TRAIN_BATCHES_PER_CLIENT={MAX_TRAIN_BATCHES_PER_CLIENT}")
    print("=" * 70)

    # Load model
    print("\n[SETUP] Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

    config = AutoConfig.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, trust_remote_code=True, torch_dtype=torch.float32,
    )
    num_layers = config.num_hidden_layers
    print(f"  Model: {MODEL_ID}")
    print(f"  Layers: {num_layers}, Vocab: {config.vocab_size}")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Load data
    print("\n[SETUP] Loading wikitext...")
    t0 = time.time()
    train_texts, test_texts = load_wikitext(tokenizer)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    client_partitions = partition_data(train_texts, NUM_CLIENTS)
    print(f"  Client partitions: {[len(p) for p in client_partitions]}")
    test_enc = tokenize_texts(tokenizer, test_texts, MAX_SEQ_LEN)
    test_enc = {k: v.clone() for k, v in test_enc.items()}

    # Apply LoRA
    print("\n[SETUP] Applying LoRA...")
    wrapper = LoraAppliedModel(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    wrapper.apply_lora()

    # Initial LoRA weight norms
    print("\n[DIAG] Initial LoRA weight norms (t=0, no training):")
    total_A_init, total_B_init = 0.0, 0.0
    for full_name, lora_layer in wrapper.lora_layers.items():
        a_n = lora_layer.lora_A.data.norm().item()
        b_n = lora_layer.lora_B.data.norm().item()
        total_A_init += a_n
        total_B_init += b_n
        print(f"    {full_name[:60]}: A={a_n:.6f}  B={b_n:.6f}")
    n_layers = len(wrapper.lora_layers)
    print(f"  [SUMMARY] Mean A norm={total_A_init/n_layers:.6f}, Mean B norm={total_B_init/n_layers:.6f}")

    # ppl before training
    print("\n[DIAG] Computing perplexity BEFORE training...")
    ppl_before = compute_perplexity(model, test_enc)
    print(f"  ppl BEFORE training: {ppl_before:.4f}")

    # ---- Federated round ----
    print("\n" + "=" * 70)
    print(f"ROUND 1 (1 client, {MAX_TRAIN_BATCHES_PER_CLIENT} batches)")
    print("=" * 70)

    round_base = snapshot_lora_state(wrapper)
    deltas = []
    weights = []

    for cid_idx, texts in enumerate(client_partitions):
        client_id = f"client-{cid_idx+1}"
        seed = 1 * 100 + cid_idx * 17 + 42  # same as byzantine
        selected = lisa_select_layers(num_layers, seed=seed)
        print(f"\n  Client {client_id}: selected_layers={selected}")

        result = train_client(model, tokenizer, wrapper, texts, 1, client_id,
                              selected_layers=selected)
        weights.append(len(texts))
        print(f"    avg_train_loss={result['avg_train_loss']:.4f}")

        state_after = snapshot_lora_state(wrapper)
        delta = compute_deltas(round_base, state_after)
        deltas.append(delta)

        restore_lora_state(wrapper, round_base)

    # Aggregate
    print("\n[AGGREGATION]")
    aggregate_deltas(deltas, weights, wrapper, round_num=1)

    # ppl after
    print("\n[DIAG] Computing perplexity AFTER aggregation...")
    ppl_after = compute_perplexity(model, test_enc)
    print(f"\n{'='*70}")
    print(f"RESULT: ppl BEFORE={ppl_before:.4f}  ppl AFTER={ppl_after:.4f}")
    print(f"        ppl ratio={ppl_after/ppl_before:.2f}x")
    print(f"        delta_ppl={ppl_after - ppl_before:+.4f}")
    print(f"{'='*70}")

    if ppl_after > 1e10:
        print("!!! CATASTROPHIC DIVERGENCE CONFIRMED in 1-client minimal test")
    elif ppl_after > ppl_before * 10:
        print("!!! SIGNIFICANT DIVERGENCE detected")
    else:
        print("OK: perplexity is stable")

    gc.collect()


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    main()
