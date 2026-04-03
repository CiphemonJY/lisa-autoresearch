#!/usr/bin/env python3
"""
LISA Blockwise LoRA Training - Streaming Version
"""

import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import time

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TUNE_FRACTION = 0.3
LORA_RANK = 2
LORA_ALPHA = 4

TRAINING_DATA = [
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "class Node: def __init__(self, val): self.val = val; self.next = None",
    "async def fetch_data(url): async with aiohttp.get(url) as r: return await r.json()",
    "for i in range(len(nums)): nums[i] *= 2",
    "result = {k: v for k, v in data.items() if v > 0}",
    "import numpy as np; arr = np.array(data).reshape(-1, 1)",
    "lambda x: x ** 2 if x > 0 else 0",
    "with open('file.txt', 'r') as f: lines = f.readlines()",
    "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
    "self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)",
]

print("🚀 Blockwise LoRA Trainer starting...", flush=True)

# Load tokenizer
print("📦 Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load model
print("📦 Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)
print("   Model loaded!", flush=True)

total_layers = len(model.model.layers)
tune_layers = int(total_layers * TUNE_FRACTION)
skip_layers = total_layers - tune_layers

print(f"   Total layers: {total_layers}", flush=True)
print(f"   Tuning final {tune_layers} layers (skipping first {skip_layers})", flush=True)

# Freeze early layers
print("🔒 Freezing early layers...", flush=True)
for i in range(skip_layers):
    for param in model.model.layers[i].parameters():
        param.requires_grad = False

# Attach LoRA
print("⚡ Attaching LoRA...", flush=True)
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"   Trainable: {trainable:,} ({100*trainable/total:.3f}%)", flush=True)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("\n🚀 Training starting...", flush=True)
for round_num in range(100):
    text = TRAINING_DATA[round_num % len(TRAINING_DATA)]
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"[R{round_num}] loss={loss.item():.4f}", flush=True)
    
    if round_num % 10 == 9:
        model.save_pretrained(f"/tmp/lisa_blockwise_r{round_num}")
        print(f"   💾 Saved checkpoint", flush=True)

print("\n✅ Done!", flush=True)
