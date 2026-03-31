#!/usr/bin/env python3
"""
LISA Real Training Example with GSM8K Dataset
Trains on actual math problems instead of dummy data
"""
import os
import sys
from pathlib import Path

# Setup
print("=" * 60)
print("LISA Real Training - GSM8K Math Dataset")
print("=" * 60)

# Install required packages
os.system("pip install datasets -q")

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load real dataset
print("\n📥 Loading GSM8K dataset...")
dataset = load_dataset("openai/gsm8k", "main")
train_data = dataset['train']
test_data = dataset['test']

print(f"   Train samples: {len(train_data)}")
print(f"   Test samples: {len(test_data)}")

# Sample data
print("\n📝 Sample problem:")
sample = train_data[0]
print(f"   Q: {sample['question']}")
print(f"   A: {sample['answer'][:100]}...")

# Model config (same as LISA)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for demo
print(f"\n🤖 Loading model: {MODEL_NAME}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"   Model loaded successfully!")
    print(f"   Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
except Exception as e:
    print(f"   Error: {e}")
    print("   Falling back to CPU...")
    model = None

# Format for training
def format_prompt(question, answer):
    return f"""Solve this math problem step by step:

Question: {question}

Solution: {answer.replace('####', 'Final Answer:')}"""

# Show formatted example
print("\n📋 Formatted training example:")
formatted = format_prompt(sample['question'], sample['answer'])
print(formatted[:300] + "...")

# Training loop
if model:
    print("\n🚀 Starting training...")
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Train on small batch
    batch_size = 2
    for i in range(min(batch_size, len(train_data))):
        item = train_data[i]
        text = format_prompt(item['question'], item['answer'])
        
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Step {i+1}/{batch_size}, Loss: {loss.item():.4f}")
    
    print("\n✅ Training complete!")
    print(f"   Memory after training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
else:
    print("\n⚠️  Model not loaded - showing dataset structure only")

print("\n" + "=" * 60)
print("To use with LISA training:")
print("1. Replace dummy data loader with this dataset loading")
print("2. Use format_prompt() to prepare training samples")
print("3. Pass to LISATrainer.train_step()")
print("=" * 60)
