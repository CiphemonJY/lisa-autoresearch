#!/usr/bin/env python3
"""Minimal Blockwise LoRA test - just 5 rounds"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import sys

print("Starting...", flush=True)

# Load
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
print("Tokenizer loaded", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)
print("Model loaded", flush=True)

# Blockwise - only final 30%
total = len(model.model.layers)
skip = total - int(total * 0.3)
print(f"Layers: {total}, Skipping: 0-{skip-1}, Tuning: {skip}-{total-1}", flush=True)

for i in range(skip):
    for p in model.model.layers[i].parameters():
        p.requires_grad = False

# LoRA
lora = LoraConfig(r=2, target_modules=["q_proj","k_proj","v_proj","o_proj"], bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,}", flush=True)

model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# 5 training rounds
texts = ["def foo(): pass", "x = 1 + 2", "return True", "print('hi')", "y = x * 2"]
for i in range(5):
    inputs = tokenizer(texts[i], return_tensors="pt").to(model.device)
    out = model(**inputs, labels=inputs["input_ids"])
    out.loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"Round {i}: loss={out.loss.item():.4f}", flush=True)

print("DONE!", flush=True)
model.save_pretrained("/tmp/lisa_blockwise_final")
