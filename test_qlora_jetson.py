#!/usr/bin/env python3
"""
Test QLoRA loading on Jetson
Tests if bitsandbytes 4-bit quantization works
"""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

print("Testing QLoRA on Jetson Orin...")

# Test 1: Small model with QLoRA
print("\n[1] Testing small model with QLoRA...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/TinyLlama-1.1B-Chat-v1.0",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print("SUCCESS: TinyLlama QLoRA works!")
except Exception as e:
    print(f"FAILED: {e}")

# Test 2: 7B with QLoRA
print("\n[2] Testing 7B with QLoRA...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print("SUCCESS: Qwen 7B QLoRA works!")
except Exception as e:
    print(f"FAILED: {e}")

# Test 3: Check GPU availability for QLoRA
print("\n[3] GPU check...")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("CUDA not available")
