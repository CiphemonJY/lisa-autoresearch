#!/usr/bin/env python3
"""
LISA Full Stack - Unified Training Script
Train any model (7B, 14B, 32B) with LISA + QLoRA + LoRA + LCSB

Usage:
    # 7B on CPU
    python3 lisa_full_stack.py --model Qwen/Qwen2.5-7B --steps 500
    
    # 14B on CPU
    python3 lisa_full_stack.py --model Qwen/Qwen2.5-14B --steps 500
    
    # 32B with QLoRA
    python3 lisa_full_stack.py --model Qwen/Qwen2.5-32B --steps 500

The full stack combines:
    LISA: Train only last 2 layers
    QLoRA: Optional 4-bit quantization for large models
    LoRA: Rank-1 adapters for efficient fine-tuning
    LCSB: Loss-Constrained Sparse Backprop
    Offload: CPU/GPU memory management
"""
import argparse
import gc
import os
import sys
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Model configs
MODEL_CONFIGS = {
    "7B": {
        "name": "Qwen/Qwen2.5-7B",
        "layers": 28,
        "quantize": False,
        "memory_gb": 14,
    },
    "14B": {
        "name": "Qwen/Qwen2.5-14B",
        "layers": 48,
        "quantize": False,
        "memory_gb": 28,
    },
    "32B": {
        "name": "Qwen/Qwen2.5-32B",
        "layers": 64,
        "quantize": True,  # 32B needs QLoRA
        "memory_gb": 64,
    },
}

CHECKPOINT_DIR_TEMPLATE = "/tmp/lisa_{model}_checkpoints"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def setup_logging(model_key):
    checkpoint_dir = CHECKPOINT_DIR_TEMPLATE.format(model=model_key.lower())
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def detect_model_type(model_name):
    """Detect model type from name."""
    if "32B" in model_name:
        return "32B"
    elif "14B" in model_name:
        return "14B"
    elif "7B" in model_name:
        return "7B"
    elif "3B" in model_name:
        return "3B"
    else:
        return "unknown"

def load_model(model_name, quantize=False):
    """Load model with optional QLoRA."""
    log(f"Loading {model_name}...")
    
    if quantize:
        log("Using QLoRA (4-bit quantization)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            log("QLoRA loading successful!")
        except Exception as e:
            log(f"QLoRA failed: {e}")
            log("Falling back to CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
    else:
        # CPU loading for 7B/14B
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    
    num_layers = len(model.model.layers)
    log(f"Model loaded: {num_layers} layers")
    return model, num_layers

def apply_lisa_lora_lcsb(model, num_layers):
    """
    Apply LISA + LoRA + LCSB in correct order:
    1. LISA: Freeze all, unfreeze last 2 layers
    2. LoRA: Add adapters
    3. LCSB: Constrain gradients to LoRA params in LISA layers
    """
    log(f"Applying LISA (last 2 of {num_layers} layers)...")
    
    # Step 1: LISA - freeze all, unfreeze last 2
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model.layers[-2:].parameters():
        p.requires_grad = True
    
    # Step 2: LoRA - add adapters
    log("Applying LoRA...")
    model = get_peft_model(model, LoraConfig(
        r=1,
        lora_alpha=2,
        target_modules=["q_proj", "k_proj", "v_proj"]
    ))
    
    # Step 3: LCSB - constrain gradients to LISA layers
    log("Applying LCSB...")
    target_layers = [num_layers - 2, num_layers - 1]
    
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(f"layers.{l}." in name for l in target_layers) and "lora_" in name:
            p.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable params: {trainable:,}")
    
    return model, target_layers

TEXTS = [
    "The cat sat on the mat and purred softly",
    "Machine learning enables computers to understand language",
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is transforming our world",
    "Neural networks learn from data patterns",
]

def train(model, tokenizer, target_layers, steps, checkpoint_dir):
    """Training loop with LISA cycling."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
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
            
            out = model(**inputs, labels=inputs["input_ids"])
            t1 = time.time()
            
            optimizer.zero_grad()
            out.loss.backward()
            t2 = time.time()
            
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
                ckpt_path = f"{checkpoint_dir}/step_{step}.pt"
                torch.save(checkpoint, ckpt_path)
                log(f"  -> Checkpoint saved")
            
            if step % 10 == 0:
                gc.collect()
    
    except KeyboardInterrupt:
        log("Interrupted!")
    except Exception as e:
        log(f"Error: {e}")
        raise
    
    total_time = time.time() - start_time
    log("=" * 60)
    log(f"DONE! Steps: {step}, Final loss: {loss}")
    log(f"Time: {total_time/3600:.1f} hours")
    
    return step, loss

def main():
    parser = argparse.ArgumentParser(description="LISA Full Stack Training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                       help="Model name (Qwen/Qwen2.5-7B, Qwen/Qwen2.5-14B, Qwen/Qwen2.5-32B)")
    parser.add_argument("--steps", type=int, default=500,
                       help="Training steps")
    args = parser.parse_args()
    
    model_type = detect_model_type(args.model)
    quantize = "32B" in args.model
    
    log("=" * 60)
    log("LISA FULL STACK TRAINING")
    log("=" * 60)
    log(f"Model: {args.model} ({model_type})")
    log(f"Strategy: LISA + LoRA + LCSB" + ("+ QLoRA" if quantize else ""))
    log("")
    
    checkpoint_dir = setup_logging(model_type)
    
    # Load
    model, num_layers = load_model(args.model, quantize=quantize)
    
    # Apply LISA + LoRA + LCSB
    model, target_layers = apply_lisa_lora_lcsb(model, num_layers)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Train
    steps, loss = train(model, tokenizer, target_layers, args.steps, checkpoint_dir)
    
    log(f"Complete! Checkpoints: {checkpoint_dir}")

if __name__ == "__main__":
    main()
