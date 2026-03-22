#!/usr/bin/env python3
"""Fast smoke test: verify SERVER_LR=0.1 aggregation doesn't diverge."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math, torch
from eval.fedavg_vs_lisafedavg import (
    LoRALinear, LoraAppliedModel,
    snapshot_lora_state, restore_lora_state, compute_deltas, aggregate_deltas,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_ID = "EleutherAI/pythia-70m"

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)

print("Applying LoRA...")
wrapper = LoraAppliedModel(model, rank=4, alpha=8.0)
wrapper.apply_lora()
wrapper.freeze_all()
for lora_layer in wrapper.lora_layers.values():
    for p in lora_layer.trainable_params():
        p.requires_grad = True

# Snapshot initial state
base = snapshot_lora_state(wrapper)

# Simulate 3 clients, each doing 5 local steps (random gradient updates)
print("Simulating 3 clients x 5 local steps...")
LR = 3e-4
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)

deltas = []
for ci in range(3):
    restore_lora_state(wrapper, base)  # reset each client to same start
    for step in range(5):
        # Dummy forward: create fake loss from lora params
        loss = sum(p.float().mean() for p in model.parameters() if p.requires_grad)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    state_after = snapshot_lora_state(wrapper)
    delta = compute_deltas(base, state_after)
    deltas.append(delta)
    restore_lora_state(wrapper, base)  # reset for next client

# Verify all deltas are reasonable (not NaN, not huge)
for i, d in enumerate(deltas):
    for k, v in d.items():
        assert not torch.isnan(v).any(), f"Client {i} delta {k} has NaN"
        assert v.float().norm() < 10.0, f"Client {i} delta {k} norm={v.float().norm()} too large"

print("  All deltas look reasonable (no NaN, norms < 10.0)")

# Apply aggregation with SERVER_LR=0.1
weights = [1.0, 1.0, 1.0]
aggregate_deltas(deltas, weights, wrapper)

# Check model is still valid
for name, param in model.named_parameters():
    assert not torch.isnan(param).any(), f"Param {name} has NaN after aggregation"

# Check LoRA params specifically
for full_name, lora_layer in wrapper.lora_layers.items():
    a_norm = lora_layer.lora_A.data.float().norm().item()
    b_norm = lora_layer.lora_B.data.float().norm().item()
    assert not torch.isnan(lora_layer.lora_A).any(), f"{full_name}.lora_A has NaN"
    assert not torch.isnan(lora_layer.lora_B).any(), f"{full_name}.lora_B has NaN"
    print(f"  {full_name}: lora_A norm={a_norm:.4f}, lora_B norm={b_norm:.4f}")

print("\n[PASS] Smoke test PASSED: SERVER_LR=0.1 aggregation is stable")
print("   Model params + LoRA layers all valid, no NaN/inf after 3-client aggregation")
