#!/usr/bin/env python3
"""
LISA Continuous Trainer - Mac Mini Client
Runs training locally on Mac Mini and submits to server
"""
import os, sys, time, json, subprocess, requests, base64, pickle
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model

SERVER_URL = "http://10.0.0.43:8080"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LOG_FILE = "/tmp/lisa_mac_train.log"
CHECKPOINT_DIR = "checkpoints"
EVAL_FILE = "/tmp/lisa_eval.json"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
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

def start_round(round_num):
    try:
        r = requests.post(f"{SERVER_URL}/start_round/{round_num}", timeout=5)
        return r.json()
    except Exception as e:
        log(f"Start round error: {e}")
        return None

def load_model():
    """Load Qwen model with LoRA."""
    log(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, config=config, trust_remote_code=True, torch_dtype=torch.float32
    )
    lora_config = LoraConfig(
        r=4, lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    log(f"Model ready: {sum(p.numel() for p in model.parameters()):,} params")
    return model, tokenizer

def train_client(model, tokenizer, client_id, steps=10):
    """Train and submit gradient."""
    log(f"  Training {client_id} ({steps} steps)...")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time in a far away land,",
        "Machine learning is transforming the world.",
        "Artificial intelligence is the future of technology.",
    ]
    
    total_loss = 0
    for i in range(steps):
        text = texts[i % len(texts)]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        
        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (i + 1) % 5 == 0:
            log(f"    Step {i+1}/{steps}: loss={loss.item():.4f}")
    
    avg_loss = total_loss / steps
    log(f"    Avg loss: {avg_loss:.4f}")
    
    # Get gradient
    gradient = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient[name] = param.grad.detach().cpu().numpy()
    
    # Compress and submit
    data = pickle.dumps(gradient)
    encoded = base64.b64encode(data).decode("utf-8")
    
    log(f"    Gradient: {len(encoded):,} bytes")
    
    payload = {
        "client_id": client_id,
        "round_number": 1,  # Always round 1 for now
        "gradient_data": encoded,
        "compression_info": {"method": "none"}
    }
    
    try:
        r = requests.post(f"{SERVER_URL}/submit", json=payload, timeout=60)
        result = r.json()
        log(f"    ✅ Submitted: {result}")
        return True
    except Exception as e:
        log(f"    ❌ Error: {e}")
        return False

def evaluate_model():
    """Evaluate model perplexity."""
    log("🔍 Evaluating...")
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')], reverse=True)
    if not checkpoints:
        return None
    
    ckpt = os.path.join(CHECKPOINT_DIR, checkpoints[0])
    log(f"  Loading {checkpoints[0]}...")
    
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "model_state" in state:
            state = state["model_state"]
        
        config = AutoConfig.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, trust_remote_code=True, torch_dtype=torch.float32)
        model.load_state_dict(state, strict=False)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
        test_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20][:30]
        
        total_loss, total_tokens = 0, 0
        for i in range(0, min(30, len(test_texts)), 4):
            enc = tokenizer(test_texts[i:i+4], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**enc)
                logits = outputs.logits[..., :-1, :].contiguous()
                labels = enc["input_ids"][..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), reduction='mean'
                )
                tokens = (enc["input_ids"] != tokenizer.pad_token_id).sum().item()
                total_loss += loss.item() * tokens
                total_tokens += tokens
        
        ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        log(f"  ✅ Perplexity: {ppl:.4f}")
        
        # Save eval
        evals = []
        if os.path.exists(EVAL_FILE):
            with open(EVAL_FILE) as f:
                evals = json.load(f)
        evals.append({"ppl": ppl, "checkpoint": checkpoints[0], "time": datetime.now().isoformat()})
        with open(EVAL_FILE, "w") as f:
            json.dump(evals[-20:], f, indent=2)
        
        return ppl
    except Exception as e:
        log(f"  ❌ Eval error: {e}")
        return None

def main():
    log("🚀 LISA Mac Mini Trainer Started")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Load model once
    model, tokenizer = load_model()
    
    round_num = 1
    eval_count = 0
    
    while True:
        if not check_server():
            log("⚠️  Server down, waiting...")
            time.sleep(30)
            continue
        
        log(f"\n📍 Round {round_num}")
        
        # Start round
        start_round(round_num)
        time.sleep(1)
        
        # Train 3 clients
        for i in range(1, 4):
            client_id = f"mac_r{round_num}_c{i}"
            success = train_client(model, tokenizer, client_id, steps=10)
            if success:
                log(f"  ✅ {client_id}")
            else:
                log(f"  ❌ {client_id} failed")
            time.sleep(2)
        
        # Check status
        try:
            r = requests.get(f"{SERVER_URL}/round/{round_num}", timeout=5)
            status = r.json()
            log(f"  Round {round_num}: {status}")
        except:
            pass
        
        # Auto-retry on failure - wait for completion or timeout
        retry_count = 0
        while retry_count < 3:
            try:
                r = requests.get(f"{SERVER_URL}/status", timeout=5)
                st = r.json()
                if str(round_num) in st.get("rounds", {}) and st["rounds"][str(round_num)]["status"] != "complete":
                    retry_count += 1
                    log(f"  ⏳ Round not complete, retry {retry_count}/3...")
                    time.sleep(30)
                else:
                    break
            except:
                retry_count += 1
                time.sleep(10)
        
        round_num += 1
        
        # Evaluate every 5 rounds
        if round_num % 5 == 0:
            eval_count += 1
            ppl = evaluate_model()
            if ppl:
                log(f"  📊 Eval #{eval_count}: perplexity={ppl:.4f}")
        
        time.sleep(30)

if __name__ == "__main__":
    main()
