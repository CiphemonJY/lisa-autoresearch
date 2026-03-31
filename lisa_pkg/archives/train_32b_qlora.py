#!/usr/bin/env python3
"""
LISA Training - 32B Model with QLoRA
Train Qwen2.5-32B with 4-bit quantization on constrained hardware

Usage:
    python3 train_32b_qlora.py --steps 500

Hardware Requirements:
    - 16GB RAM minimum (for quantized weights + activations)
    - 8GB GPU (for inference, not training)
    - 32GB swap recommended

NOTE: 32B model requires quantization (NF4) to fit in memory.
      Training is done on LoRA adapters only, with quantized base model.

Theory:
    - 32B model in FP16 = 64GB
    - 32B model in NF4 = ~16GB (4x compression)
    - LoRA adapters = ~100MB
    - Activations during backward = ~8GB
    - Total memory needed = ~24GB

This script uses bitsandbytes for 4-bit quantization.
"""
import argparse
import gc
import os
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Config
MODEL_NAME = "Qwen/Qwen2.5-32B"
DEFAULT_STEPS = 500
CHECKPOINT_DIR = "/tmp/lisa_32b_checkpoints"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

def setup_logging():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log_file = f"{CHECKPOINT_DIR}/training.log"
    with open(log_file, "a") as f:
        f.write(f"\n\n=== Training started {datetime.now()} ===\n")

def load_quantized_model():
    log(f"Loading {MODEL_NAME} with 4-bit quantization...")
    log("WARNING: Using NF4 quantization to fit 32B in memory")
    
    # Quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        log(f"Quantized loading failed: {e}")
        log("Falling back to CPU-only with reduced memory...")
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
    
    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
    
    # Unfreeze last 2 layers for LISA
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
    
    LAYER_CYCLE = target_layers[::-1]
    
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
            
            # Move inputs to same device as model
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
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
            
            log(f"Step {step}/{steps}: layer={layer}, loss={loss:.4f}, "
                f"fwd={t1-t0:.1f}s, bwd={t2-t1:.1f}s")
            
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
    parser = argparse.ArgumentParser(description="LISA 32B Training with QLoRA")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Number of training steps (default: {DEFAULT_STEPS})")
    args = parser.parse_args()
    
    log("=" * 60)
    log("LISA Training - Qwen2.5-32B with QLoRA")
    log("=" * 60)
    
    # Setup
    setup_logging()
    
    # Load model with quantization
    model = load_quantized_model()
    
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
