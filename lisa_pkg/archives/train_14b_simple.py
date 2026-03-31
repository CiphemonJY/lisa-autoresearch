#!/usr/bin/env python3
"""
LISA Training - Simple 14B Model
Train Qwen2.5-14B with LISA + LoRA on hardware with 8GB RAM + 23GB swap

Usage:
    python3 train_14b_simple.py --steps 500

Hardware Requirements:
    - 8GB RAM minimum
    - 23GB swap space (critical!)
    - CPU only (GPU training not supported on Jetson Orin)

NOTE: This script requires significant memory. Make sure swap is configured:
    sudo fallocate -l 23G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
"""
import argparse
import gc
import os
import sys
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Config
MODEL_NAME = "Qwen/Qwen2.5-14B"
DEFAULT_STEPS = 500
CHECKPOINT_DIR = "/tmp/lisa_14b_checkpoints"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

def setup_logging():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log_file = f"{CHECKPOINT_DIR}/training.log"
    with open(log_file, "a") as f:
        f.write(f"\n\n=== Training started {datetime.now()} ===\n")

def load_model():
    log(f"Loading {MODEL_NAME}...")
    log("WARNING: This model requires ~28GB total memory (RAM + swap)")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    log(f"Model loaded: {len(model.model.layers)} layers")
    return model

def apply_lisa_lora(model):
    # LISA: Freeze all, unfreeze last 2 layers
    num_layers = len(model.model.layers)
    log(f"Applying LISA (training last 2 of {num_layers} layers)...")
    
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model.layers[-2:].parameters():
        p.requires_grad = True
    
    # LoRA on attention projections
    log("Applying LoRA...")
    model = get_peft_model(model, LoraConfig(
        r=1,
        lora_alpha=2,
        target_modules=["q_proj", "k_proj", "v_proj"]
    ))
    
    # LISA on LoRA: Only train LoRA params in last 2 layers
    log("Setting up LISA on LoRA...")
    target_layers = [num_layers - 2, num_layers - 1]
    
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(f"layers.{l}." in name for l in target_layers) and "lora_" in name:
            p.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable params: {trainable:,}")
    
    return model, target_layers

def train(model, tokenizer, target_layers, steps):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    # Simple training texts
    TEXTS = [
        "The cat sat on the mat and purred softly",
        "Machine learning enables computers to understand language",
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence is transforming our world",
        "Neural networks learn from data patterns",
    ]
    
    LAYER_CYCLE = target_layers[::-1]  # Alternate: [46, 47, 46, ...]
    
    log(f"Training {steps} steps...")
    log("=" * 60)
    
    step = 0
    start_time = time.time()
    
    try:
        while step < steps:
            step += 1
            text_idx = (step - 1) % len(TEXTS)
            layer = LAYER_CYCLE[(step - 1) % len(LAYER_CYCLE)]
            
            gc.collect()
            
            t0 = time.time()
            inputs = tokenizer(TEXTS[text_idx], return_tensors="pt", max_length=8)
            
            out = model(**inputs, labels=inputs["input_ids"])
            t1 = time.time()
            
            optimizer.zero_grad()
            out.loss.backward()
            t2 = time.time()
            
            grad_norm = sum(
                p.grad.norm().item() 
                for p in model.parameters() 
                if p.grad is not None
            )
            optimizer.step()
            t3 = time.time()
            
            loss = out.loss.item()
            elapsed = time.time() - start_time
            rate = step / elapsed if elapsed > 0 else 0
            
            log(f"Step {step}/{steps}: layer={layer}, loss={loss:.4f}, "
                f"fwd={t1-t0:.1f}s, bwd={t2-t1:.1f}s")
            
            # Save checkpoint every 50 steps
            if step % 50 == 0:
                checkpoint = {
                    "step": step,
                    "loss": loss,
                    "model_state": {
                        k: v.clone() 
                        for k, v in model.named_parameters() 
                        if v.requires_grad
                    }
                }
                ckpt_path = f"{CHECKPOINT_DIR}/step_{step}.pt"
                torch.save(checkpoint, ckpt_path)
                log(f"  -> Checkpoint saved: {ckpt_path}")
            
            # Cleanup every 10 steps
            if step % 10 == 0:
                gc.collect()
    
    except KeyboardInterrupt:
        log("Training interrupted!")
    except Exception as e:
        log(f"Error at step {step}: {e}")
        raise
    
    total_time = time.time() - start_time
    log("=" * 60)
    log(f"DONE! Steps: {step}, Final loss: {loss}")
    log(f"Total time: {total_time/3600:.1f} hours")
    
    return step, loss

def main():
    parser = argparse.ArgumentParser(description="LISA 14B Training")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Number of training steps (default: {DEFAULT_STEPS})")
    args = parser.parse_args()
    
    log("=" * 60)
    log("LISA Training - Qwen2.5-14B")
    log("=" * 60)
    
    # Check swap
    swap_total = os.popen("free -b | awk '/Swap:/ {print $2}'").read().strip()
    if swap_total and int(swap_total) < 20 * 1024**3:
        log("WARNING: Less than 20GB swap detected!")
        log("14B model requires ~23GB swap. Configure with:")
        log("  sudo fallocate -l 23G /swapfile")
        log("  sudo chmod 600 /swapfile")
        log("  sudo mkswap /swapfile")
        log("  sudo swapon /swapfile")
    
    # Setup
    setup_logging()
    
    # Load model
    model = load_model()
    
    # Apply LISA + LoRA
    model, target_layers = apply_lisa_lora(model)
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Train
    steps, loss = train(model, tokenizer, target_layers, args.steps)
    
    log("Training complete!")
    log(f"Checkpoints saved to: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()
