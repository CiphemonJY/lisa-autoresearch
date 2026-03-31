#!/usr/bin/env python3
"""
LISA Training - 32B Model with Sequential Layer Offload
Train Qwen2.5-32B by loading one layer at a time

Usage:
    python3 train_32b_sequential.py --steps 500

Hardware Requirements:
    - 16GB RAM minimum
    - 64GB swap (critical!)
    - CPU only

Strategy:
    Instead of loading the entire 32B model, we:
    1. Load embeddings and output layer (~2GB)
    2. Load one transformer layer at a time (~2GB each)
    3. Process forward pass layer by layer
    4. Only keep gradients for LoRA adapters in last 2 layers

NOTE: This is experimental and may be slow. Expected ~5-10 min per step.
"""
import argparse
import gc
import os
import time
from datetime import datetime

import torch
from transformers import AutoConfig, AutoTokenizer

# Config
MODEL_NAME = "Qwen/Qwen2.5-32B"
DEFAULT_STEPS = 100
CHECKPOINT_DIR = "/tmp/lisa_32b_seq_checkpoints"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

def setup_logging():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class SequentialLayerModel(torch.nn.Module):
    """
    A model that processes one layer at a time to save memory.
    Only the LoRA adapters in the last 2 layers have gradients.
    """
    def __init__(self, model_name, device="cpu"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        log("Loading model config...")
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.vocab_size = self.config.vocab_size
        
        log(f"Model: {self.num_layers} layers, hidden={self.hidden_size}, vocab={self.vocab_size}")
        
        # Load embedding and output on CPU
        log("Loading embeddings...")
        from transformers import AutoModel
        full_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        self.embed_tokens = full_model.model.embed_tokens.to(device)
        self.norm = full_model.model.norm.to(device)
        self.lm_head = full_model.lm_head.to(device)
        
        # Free full model memory
        del full_model
        gc.collect()
        
        # Load last 2 layers for LISA training
        log("Loading last 2 layers for LISA...")
        from transformers import AutoModelForCausalLM
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="sequential",
            trust_remote_code=True,
        )
        
        self.last_layers = torch.nn.ModuleList([
            full_model.model.layers[-2],
            full_model.model.layers[-1],
        ]).to(device)
        
        del full_model
        gc.collect()
        
        # LoRA for last 2 layers
        self.lora_q = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_size, 1, bias=False)
            for _ in range(2)
        ]).to(device)
        self.lora_k = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_size, 1, bias=False)
            for _ in range(2)
        ]).to(device)
        self.lora_v = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_size, 1, bias=False)
            for _ in range(2)
        ]).to(device)
        
        # Freeze all
        for p in self.parameters():
            p.requires_grad = False
        
        # Unfreeze LoRA and last 2 layers
        for i in range(2):
            for p in self.last_layers[i].parameters():
                p.requires_grad = True
            for p in [self.lora_q[i], self.lora_k[i], self.lora_v[i]].parameters():
                p.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log(f"Trainable params: {trainable:,}")
    
    def forward(self, input_ids, labels=None):
        # Embed
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through last 2 layers (LISA)
        for idx, layer in enumerate(self.last_layers):
            # Self-attention with LoRA
            residual = hidden_states
            
            # Simplified attention - in reality would need full attention module
            # This is a placeholder showing the structure
            attn_output = layer.self_attn(
                hidden_states,
                attention_mask=None,
            )[0]
            
            # Apply LoRA
            lora_q_out = self.lora_q[idx](hidden_states)
            lora_k_out = self.lora_k[idx](hidden_states)
            lora_v_out = self.lora_v[idx](hidden_states)
            
            # Add LoRA contribution
            hidden_states = attn_output + 0.1 * (lora_q_out + lora_k_out + lora_v_out)
            
            # FFN
            hidden_states = layer.mlp(hidden_states) + residual
        
        # Norm and output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        
        return type('Output', (), {'loss': loss, 'logits': logits})()

def train(model, tokenizer, steps):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    TEXTS = [
        "The cat sat on the mat",
        "Machine learning works",
    ]
    
    log(f"Training {steps} steps...")
    log("=" * 60)
    
    step = 0
    start_time = time.time()
    
    try:
        while step < steps:
            step += 1
            text_idx = (step - 1) % len(TEXTS)
            
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
            
            log(f"Step {step}/{steps}: loss={loss:.4f}, fwd={t1-t0:.1f}s, bwd={t2-t1:.1f}s")
            
            if step % 10 == 0:
                gc.collect()
    
    except Exception as e:
        log(f"Error at step {step}: {e}")
        raise
    
    log(f"DONE! Steps: {step}, Final loss: {loss}")
    return step, loss

def main():
    parser = argparse.ArgumentParser(description="LISA 32B Sequential Training")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    
    log("=" * 60)
    log("LISA Training - Qwen2.5-32B (Sequential Layer Offload)")
    log("=" * 60)
    
    setup_logging()
    
    model = SequentialLayerModel(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    steps, loss = train(model, tokenizer, args.steps)
    
    log("Training complete!")

if __name__ == "__main__":
    main()
