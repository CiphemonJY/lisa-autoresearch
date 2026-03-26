#!/usr/bin/env python3
"""
LISA Federated Learning - Continuous Training with Auto-Evaluation
Runs training rounds continuously and tracks model quality over time
"""
import os
import sys
import time
import json
import subprocess
import requests
import threading
from datetime import datetime
from pathlib import Path

SERVER_URL = "http://10.0.0.43:8080"
JETSON_HOST = "jetson@10.0.0.145"
CHECKPOINT_DIR = "checkpoints"
EVAL_RESULTS_FILE = "/tmp/lisa_eval_results.json"
LOG_FILE = "/tmp/lisa_continuous.log"

# Training config
TRAIN_STEPS = 20
CLIENTS_PER_ROUND = 3
EVAL_EVERY_ROUNDS = 5

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def check_server():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.json().get("status") == "ok"
    except:
        return False

def get_status():
    try:
        r = requests.get(f"{SERVER_URL}/status", timeout=5)
        return r.json()
    except:
        return {}

def run_client(client_id, train_steps=TRAIN_STEPS):
    cmd = [
        "ssh", JETSON_HOST,
        f"cd /home/jetson/lisa_proj && /home/jetson/lisa_proj/venv/bin/python3 main.py --mode client --client-id {client_id} --server {SERVER_URL} --model Qwen/Qwen2.5-0.5B --train-steps {train_steps} --rounds 1"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return "Gradient submitted OK" in result.stdout
    except:
        return False

def evaluate_model():
    """Evaluate current model quality using wikitext perplexity."""
    log("🔍 Evaluating model quality...")
    
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')], reverse=True)
    if not checkpoints:
        return None
    
    latest_ckpt = os.path.join(CHECKPOINT_DIR, checkpoints[0])
    log(f"   Loading {checkpoints[0]}...")
    
    # Create evaluation script
    eval_script = f"""
import sys
sys.path.insert(0, '/Users/Ciphemon/.openclaw/workspace/lisa_proj')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

try:
    state = torch.load("{latest_ckpt}", map_location="cpu", weights_only=False)
    if "model_state" in state:
        state = state["model_state"]
    
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", config=config, trust_remote_code=True, torch_dtype=torch.float32)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
    test_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20][:50]
    
    total_loss, total_tokens = 0, 0
    for i in range(0, min(50, len(test_texts)), 4):
        enc = tokenizer(test_texts[i:i+4], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits[..., :-1, :].contiguous()
            labels = enc["input_ids"][..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='mean')
            tokens = (enc["input_ids"] != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    
    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    print(f"PERPLEXITY:{ppl:.4f}")
except Exception as e:
    print(f"ERROR:{e}")
"""
    
    with open("/tmp/eval_ppl.py", "w") as f:
        f.write(eval_script)
    
    try:
        result = subprocess.run(["python3", "/tmp/eval_ppl.py"], capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        
        if "PERPLEXITY:" in output:
            ppl = float(output.split("PERPLEXITY:")[1].split()[0])
            log(f"   ✅ Perplexity: {ppl:.4f}")
            return ppl
        else:
            log(f"   ❌ Eval failed: {output[-200:]}")
            return None
    except Exception as e:
        log(f"   ❌ Error: {e}")
        return None

def get_dashboard_data():
    """Gather all data for dashboard."""
    status = get_status()
    
    # Get checkpoints
    checkpoints = []
    if os.path.exists(CHECKPOINT_DIR):
        for f in sorted(os.listdir(CHECKPOINT_DIR), reverse=True)[:10]:
            if f.endswith('.pt'):
                path = os.path.join(CHECKPOINT_DIR, f)
                checkpoints.append({
                    "name": f,
                    "size_mb": os.path.getsize(path) / (1024*1024),
                    "modified": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%H:%M:%S")
                })
    
    # Get eval results
    eval_results = []
    if os.path.exists(EVAL_RESULTS_FILE):
        with open(EVAL_RESULTS_FILE) as f:
            eval_results = json.load(f)[-20:]  # Last 20
    
    return {
        "server_status": status,
        "checkpoints": checkpoints,
        "eval_results": eval_results,
        "timestamp": datetime.now().isoformat()
    }

def save_eval_result(round_num, perplexity):
    """Save evaluation result to history."""
    results = []
    if os.path.exists(EVAL_RESULTS_FILE):
        with open(EVAL_RESULTS_FILE) as f:
            results = json.load(f)
    
    results.append({
        "round": round_num,
        "perplexity": perplexity,
        "timestamp": datetime.now().isoformat()
    })
    
    with open(EVAL_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

def training_loop():
    """Main training loop."""
    log("🚀 LISA Continuous Training Started")
    log(f"   Server: {SERVER_URL}")
    log(f"   Clients/round: {CLIENTS_PER_ROUND}")
    log(f"   Steps/client: {TRAIN_STEPS}")
    log(f"   Eval every: {EVAL_EVERY_ROUNDS} rounds")
    
    round_num = 1
    eval_round = 0
    
    while True:
        if not check_server():
            log("⚠️  Server not available, waiting...")
            time.sleep(60)
            continue
        
        status = get_status()
        log(f"\n📍 Round {round_num}: {status}")
        
        # Run clients
        success_count = 0
        for i in range(CLIENTS_PER_ROUND):
            client_id = f"auto_{round_num}_c{i+1}"
            log(f"   Running {client_id}...")
            if run_client(client_id):
                success_count += 1
                log(f"   ✅ {client_id}")
            time.sleep(2)
        
        log(f"   Round {round_num}: {success_count}/{CLIENTS_PER_ROUND} clients succeeded")
        
        # Evaluate periodically
        if round_num % EVAL_EVERY_ROUNDS == 0:
            eval_round += 1
            perplexity = evaluate_model()
            if perplexity:
                save_eval_result(eval_round, perplexity)
        
        round_num += 1
        
        # Save dashboard data
        dashboard_data = get_dashboard_data()
        with open("/tmp/lisa_dashboard.json", "w") as f:
            json.dump(dashboard_data, f)
        
        # Wait before next round
        log(f"   Sleeping 30s before next round...")
        time.sleep(30)

if __name__ == "__main__":
    training_loop()
