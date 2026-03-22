#!/usr/bin/env python3
"""Quick 1-round comparative test: FedAvg vs LISA-FedAvg"""
import os, sys, time
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, os.path.dirname(__file__))

import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.fedavg_vs_lisafedavg import (
    LoRALinear, LoraAppliedModel,
    load_wikitext, partition_data, tokenize_texts,
    MODEL_ID, LOCAL_EPOCHS, BATCH_SIZE, MAX_SEQ_LEN,
    LR, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
    LISA_BOTTOM, LISA_TOP, LISA_MIDDLE,
    MAX_TRAIN_BATCHES_PER_CLIENT, SEED
)

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

NUM_CLIENTS = 3
NUM_ROUNDS = 1          # 1 round for quick test
TRAIN_BATCHES = 10      # reduced from 40 for speed
TEST_BATCHES = 10       # reduced from 50 for speed

print("=" * 60)
print(" Quick FedAvg vs LISA-FedAvg Comparative Test")
print("=" * 60)
start_total = time.time()

# Load tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token
print(f"\n[t={time.time()-start_total:.0f}s] Tokenizer loaded")

# Load data
train_texts, test_texts = load_wikitext(tok, max_seq=MAX_SEQ_LEN)
client_texts = partition_data(train_texts, NUM_CLIENTS, non_iid=True)
test_batches = tokenize_texts(tok, test_texts[:TEST_BATCHES * BATCH_SIZE], MAX_SEQ_LEN)
print(f"[t={time.time()-start_total:.0f}s] Data loaded: {len(train_texts)} texts, {NUM_CLIENTS} clients")

# Compute perplexity
def compute_perplexity(model, batch, device='cpu'):
    model.eval()
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss.item()
    return math.exp(loss)

import math

# Train one experiment
def run_experiment(method, lisa_middle_sample, train_all_layers=False):
    print(f"\n{'='*60}")
    print(f"  Method: {method}")
    print(f"{'='*60}")
    t0 = time.time()

    # Fresh model each time
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)
    model.to('cpu')

    # Apply LoRA
    lora_wrapper = LoraAppliedModel(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    n_lora = lora_wrapper.apply_lora()
    lora_wrapper.freeze_all()

    # LISA layer selection
    num_layers = model.config.num_hidden_layers
    bottom = list(range(LISA_BOTTOM))
    top = list(range(num_layers - LISA_TOP, num_layers))
    middle_pool = list(range(LISA_BOTTOM, num_layers - LISA_TOP))

    if train_all_layers:
        # True FedAvg baseline: all layers trainable
        selected_layers = list(range(num_layers))
        print(f"  [FedAvg baseline] All {num_layers} layers trainable")
    else:
        # LISA: only mandatory bottom + sampled middle + top
        middle = random.sample(middle_pool, min(lisa_middle_sample, len(middle_pool))) if middle_pool else []
        selected_layers = sorted(set(bottom + middle + top))
        print(f"  Selected layers: bottom={bottom}, middle={middle}, top={top}")

    print(f"  Trainable params: {lora_wrapper.get_trainable_count():,}")

    lora_wrapper.unfreeze_lora_layers(selected_layers)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR, weight_decay=0.01
    )

    # Initial perplexity
    ppl_before = compute_perplexity(model, test_batches)
    print(f"  Round 0 perplexity: {ppl_before:.4f}")

    # Federated round
    client_grads = []
    for client_id in range(NUM_CLIENTS):
        t1 = time.time()
        lora_wrapper.freeze_all()
        lora_wrapper.unfreeze_lora_layers(selected_layers)

        # Local training
        texts = client_texts[client_id][:TRAIN_BATCHES * BATCH_SIZE]
        for i in range(0, len(texts), BATCH_SIZE):
            batch = tokenize_texts(tok, texts[i:i+BATCH_SIZE], MAX_SEQ_LEN)
            input_ids = batch['input_ids']
            labels = batch['labels']

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Collect gradients (average them across clients)
        grad = {name: p.grad.data.clone() for name, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        client_grads.append(grad)
        print(f"  Client {client_id}: trained, grad norm={sum(g.float().norm().item()**2 for g in grad.values())**0.5:.4f} [{time.time()-t1:.1f}s]")

    # FedAvg: average gradients
    avg_grad = {}
    for name in client_grads[0]:
        avg_grad[name] = sum(g[name].float() for g in client_grads) / NUM_CLIENTS

    # Apply averaged gradient (approximate — adds to params directly)
    for name, p in model.named_parameters():
        if name in avg_grad and p.requires_grad:
            p.data.add_(avg_grad[name], alpha=0.1)  # small learning rate step

    # Final perplexity
    ppl_after = compute_perplexity(model, test_batches)
    print(f"  Round 1 perplexity: {ppl_after:.4f} (before={ppl_before:.4f}, delta={ppl_after-ppl_before:+.4f})")
    print(f"  Time: {time.time()-t0:.1f}s")
    return {"method": method, "ppl_before": ppl_before, "ppl_after": ppl_after, "delta": ppl_after - ppl_before}

# Run both
print(f"\nTotal setup time: {time.time()-start_total:.0f}s")
print("\n" + "="*60)
print("  RUNNING FEDAVG (LISA_MIDDLE_SAMPLE=0 — all layers)")
print("="*60)
result_fedavg = run_experiment("FedAvg (baseline)", lisa_middle_sample=0, train_all_layers=True)

print("\n" + "="*60)
print("  RUNNING LISA-FEDAVG (LISA_MIDDLE_SAMPLE=2 — selected layers)")
print("="*60)
result_lisa = run_experiment("LISA-FedAvg", lisa_middle_sample=LISA_MIDDLE)

# Summary
print("\n" + "="*60)
print("  RESULTS SUMMARY")
print("="*60)
print(f"  FedAvg perplexity delta:      {result_fedavg['delta']:+.4f}")
print(f"  LISA-FedAvg perplexity delta: {result_lisa['delta']:+.4f}")
improvement = result_fedavg['delta'] - result_lisa['delta']
print(f"\n  LISA advantage:                {improvement:+.4f} (negative = LISA better)")
print(f"  Total time:                    {time.time()-start_total:.1f}s")
print("="*60)
