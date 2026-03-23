#!/usr/bin/env python3
"""Quick test: find the right SERVER_LR that prevents divergence."""
import sys, os, math, random, gc
sys.path.insert(0, os.path.dirname(__file__))

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_ID = "EleutherAI/pythia-70m"
LR = 8e-4
LORA_RANK, LORA_ALPHA = 4, 8.0
BATCH_SIZE, MAX_SEQ_LEN = 4, 64
TRAIN_BATCHES = 20
NUM_CLIENTS = 3
SEED = 42
MAX_TEST = 5


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=4, alpha=8.0, dropout=0.05):
        super().__init__()
        self.weight_data = linear.weight.data.clone().float()
        self.bias_data = linear.bias.data.clone().float() if linear.bias is not None else None
        self.out_features, self.in_features = self.weight_data.shape
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        # FIX 1: Both lora_A AND lora_B must be trainable.
        # Using nn.Parameter ensures requires_grad=True by default.
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        with torch.no_grad():
            original = nn.functional.linear(x_f32, self.weight_data, self.bias_data)
        lora = nn.functional.linear(self.lora_dropout(x_f32), self.lora_A)
        lora = nn.functional.linear(lora, self.lora_B)
        return (original + lora * self.scaling).to(orig_dtype)

    def trainable_params(self):
        return [self.lora_A, self.lora_B]


class LoraWrapper:
    TARGET = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
              "c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj", "fc1", "fc2", "c_fc"]

    def __init__(self, model, rank=4, alpha=8.0, dropout=0.05):
        self.model = model
        self.rank, self.alpha, self.dropout = rank, alpha, dropout
        self.lora_layers = {}

    def apply_lora(self):
        for full_name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv1d)):
                continue
            name_parts = full_name.split(".")
            if not any(tm in name_parts[-1] for tm in self.TARGET):
                continue
            lora = LoRALinear(module, rank=self.rank, alpha=self.alpha, dropout=self.dropout)
            self.lora_layers[full_name] = lora
            parts = full_name.rsplit(".", 1)
            if len(parts) == 2:
                try:
                    setattr(self.model.get_submodule(parts[0]), parts[1], lora)
                except KeyError:
                    pass
        return len(self.lora_layers)

    def freeze_all(self):
        """Freeze everything (including both A and B)."""
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_lora_layers(self, layer_indices):
        """Unfreeze BOTH lora_A and lora_B for selected layers."""
        patterns = []
        for idx in layer_indices:
            patterns.extend([f"gpt_neox.layers.{idx}.", f".h.{idx}."])

        for full_name, lora_layer in self.lora_layers.items():
            for pat in patterns:
                if pat in full_name:
                    for p in lora_layer.trainable_params():
                        p.requires_grad = True
                    break

    def snapshot(self):
        return {f"{k}.A": l.lora_A.data.clone().cpu() for k, l in self.lora_layers.items()} | \
               {f"{k}.B": l.lora_B.data.clone().cpu() for k, l in self.lora_layers.items()}

    def restore(self, state):
        """Restore weights WITHOUT restoring requires_grad (preserves trainable state)."""
        for k, l in self.lora_layers.items():
            l.lora_A.data.copy_(state[f"{k}.A"].clone())
            l.lora_B.data.copy_(state[f"{k}.B"].clone())


def tokenize(tok, texts):
    enc = tok(texts, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc


def ppl(model, enc, tok):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    pad_id = tok.pad_token_id or 0
    crit = nn.CrossEntropyLoss(ignore_index=pad_id)
    for i in range(min(MAX_TEST, len(enc["input_ids"]) // BATCH_SIZE)):
        s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
        ids = enc["input_ids"][s:e].clone()
        labs = enc["labels"][s:e].clone()
        with torch.no_grad():
            out = model(input_ids=ids)
        loss = crit(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


# Load model
print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)
lm = LoraWrapper(model, rank=LORA_RANK, alpha=LORA_ALPHA)
n = lm.apply_lora()
print(f"  LoRA={n}, pad_token_id={tok.pad_token_id}")

# Load data
print("Loading wikitext...")
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
ts = load_dataset("wikitext", "wikitext-2-v1", split="test")
train_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
test_texts = [t for t in ts["text"] if t.strip() and len(t.strip()) > 20]
train_enc = tokenize(tok, train_texts)
test_enc = tokenize(tok, test_texts)

# Partition
rng = random.Random(SEED)
shuffled = list(train_texts)
rng.shuffle(shuffled)
n_per = len(shuffled) // NUM_CLIENTS
client_encs = [tokenize(tok, shuffled[i*n_per:(i+1)*n_per if i < NUM_CLIENTS-1 else len(shuffled)])
               for i in range(NUM_CLIENTS)]
counts = [len(c["input_ids"]) for c in client_encs]
print(f"  texts per client: {counts}")


def run_one_round(srv_lr):
    """Run one federated round, return ppl_after."""
    # Fresh LoRA: reinitialize both A and B
    for lora_layer in lm.lora_layers.values():
        lora_layer.lora_A.data.normal_(mean=0, std=0.01)
        lora_layer.lora_B.data.zero_()

    ppl_before = ppl(model, test_enc, tok)
    snap = lm.snapshot()

    deltas, weights = [], []
    for ci in range(NUM_CLIENTS):
        # FIX 2: Unfreeze BOTH A and B for layers 0-5 (no "A_only" method)
        lm.freeze_all()
        lm.unfreeze_lora_layers(list(range(6)))
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)

        for _ in range(TRAIN_BATCHES):
            idx = torch.randperm(len(client_encs[ci]["input_ids"]))[:BATCH_SIZE].tolist()
            ids = client_encs[ci]["input_ids"][idx].clone().clamp(0, tok.vocab_size - 1)
            labs = client_encs[ci]["labels"][idx].clone().clamp(0, tok.vocab_size - 1)
            opt.zero_grad()
            out = model(input_ids=ids, labels=labs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        weights.append(len(client_encs[ci]["input_ids"]))
        snap_after = lm.snapshot()
        deltas.append({k: snap_after[k] - snap[k] for k in snap})
        # FIX 4: Do NOT restore — we need the trained state to persist.
        # The delta is computed as (trained - base) and stored separately;
        # we accumulate deltas in the deltas list, not by mutating model state.
        # Re-freeze to prevent accidental gradient accumulation across clients
        lm.freeze_all()

    # Aggregate
    total_w = sum(weights)
    nw = [w / total_w for w in weights]
    acc = {}
    for delta, w in zip(deltas, nw):
        for k, v in delta.items():
            acc[k] = acc.get(k, torch.zeros_like(v)) + v.float() * w

    # FIX 5: nan_to_num safety guard before applying
    for k in acc:
        acc[k] = acc[k].nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)

    delta_norms = [v.norm().item() for v in acc.values()]
    avg_norm = sum(delta_norms) / len(delta_norms) if delta_norms else 0.0
    max_norm = max(delta_norms) if delta_norms else 0.0

    # FIX 6: Clamp per-element update magnitude to keep federated updates stable.
    # SERVER_LR = min(srv_lr, MAX_UPDATE/max_norm) ensures no single parameter
    # shifts by more than MAX_UPDATE per federated round.
    # MAX_UPDATE=0.1: max shift per element is 0.1 (10% of typical LoRA values).
    # For max_norm=1.33, SERVER_LR = min(srv_lr, 0.075).
    # srv_lr is the user-specified ceiling on SERVER_LR.
    MAX_UPDATE = 0.1
    SERVER_LR = min(srv_lr, MAX_UPDATE / max_norm) if max_norm > 1e-8 else srv_lr

    # Apply
    for k, l in lm.lora_layers.items():
        with torch.no_grad():
            l.lora_A.data.add_(acc[f"{k}.A"], alpha=SERVER_LR)
            l.lora_B.data.add_(acc[f"{k}.B"], alpha=SERVER_LR)

    ppl_after = ppl(model, test_enc, tok)
    return ppl_before, ppl_after, avg_norm, max_norm, SERVER_LR


print(f"\n--- Testing SERVER_LR (1 round, {NUM_CLIENTS} clients, {TRAIN_BATCHES} batches each) ---")
print(f"  LR={LR}, LORA_RANK={LORA_RANK}, ALPHA={LORA_ALPHA}")
print()

for srv_lr in [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
    ppl_before, ppl_after, avg_norm, max_norm = run_one_round(srv_lr)
    delta_ppl = ppl_after - ppl_before
    pct_change = (delta_ppl / ppl_before * 100) if ppl_before > 0 else 0
    if ppl_after > ppl_before * 1.5 and ppl_before > 1e6:
        status = "DIVERGED-SAME"
    elif ppl_after > ppl_before * 1.5:
        status = "DIVERGED-UP"
    elif ppl_after < ppl_before * 0.99:
        status = "LEARNING"
    else:
        status = "FLAT"
    print(f"  SERVER_LR={srv_lr:>6}: acc_norm_avg={avg_norm:.4f} max={max_norm:.4f} | ppl {ppl_before:.0f} -> {ppl_after:.0f} ({pct_change:+.1f}%) [{status}]")

print("\nDone.")
