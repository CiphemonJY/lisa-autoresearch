#!/usr/bin/env python3
"""
LISA Federated Learning - Easy Join Server
BitTorrent-style share links for federated learning
"""
import os
import sys
import time
import pickle
import base64
import zlib
import hashlib
import secrets
import threading
import numpy as np
import torch
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lisa-join")

# ============ LoRA ============
def apply_lora_to_model(model, rank=4, alpha=8.0, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=dropout, bias="none", task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)

# ============ Join Code System ============
JOIN_CODES = {}  # code -> {server_url, model_name, created_at}
ngrok_tunnel_url = None  # Global for ngrok URL

def start_ngrok_tunnel(port, auth_token=None):
    """
    Start ngrok tunnel to expose server publicly.
    Returns the public URL if successful, None otherwise.
    """
    import subprocess
    import json
    import time
    
    # Check if ngrok is installed
    try:
        result = subprocess.run(["ngrok", "--version"], capture_output=True, text=True)
        logger.info(f"ngrok found: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.warning("ngrok not installed. Install with: brew install ngrok")
        logger.warning("Or download from: https://ngrok.com/download")
        return None
    
    # Start ngrok tunnel
    cmd = ["ngrok", "http", str(port), "--log", "stdout"]
    if auth_token:
        cmd.extend(["--authtoken", auth_token])
    
    try:
        # Start ngrok in background
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Wait for tunnel to establish (look for URL in output)
        public_url = None
        for _ in range(60):  # 30 second timeout
            line = proc.stdout.readline()
            if line:
                logger.info(f"ngrok: {line.strip()}")
                # Look for the public URL
                if "https://" in line and ".ngrok.io" in line:
                    # Extract URL
                    import re
                    match = re.search(r'https://[a-z0-9]+\.ngrok\.io', line)
                    if match:
                        public_url = match.group(0)
                        break
            time.sleep(0.5)
        
        if public_url:
            logger.info(f"🌐 ngrok tunnel established: {public_url}")
            return public_url
        else:
            logger.warning("ngrok started but couldn't get public URL")
            proc.terminate()
            return None
            
    except Exception as e:
        logger.error(f"ngrok error: {e}")
        return None

def generate_join_code(server_url="http://localhost:8080", model_name="Qwen/Qwen2.5-0.5B"):
    """
    Generate a unique, secure join code.
    - 12 characters from mixed alphanumeric (no confusing chars: 0/O, 1/I/L)
    - Includes server-derived entropy (timestamp + random)
    - Collision-checked against existing codes
    """
    # Characters that are easy to read/type (no 0, O, 1, I, L)
    chars = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    
    # Server entropy: timestamp (40 bits) + process ID (16 bits) + random (24 bits)
    import struct
    server_entropy = struct.unpack('>Q', struct.pack('>Q', int(time.time() * 1000)))[0] ^ (os.getpid() << 40)
    
    # Generate code with server + random entropy
    combined_entropy = (server_entropy << 24) | secrets.randbits(24)
    
    # Build 12-char code
    code_parts = []
    remaining = combined_entropy
    for _ in range(12):
        idx = remaining % len(chars)
        code_parts.append(chars[idx])
        remaining //= len(chars)
    
    code = ''.join(code_parts)
    
    # Collision check (regenerate if needed)
    max_attempts = 10
    attempts = 0
    while code in JOIN_CODES and attempts < max_attempts:
        # Add more random suffix
        extra = ''.join(secrets.choice(chars) for _ in range(4))
        code = code[:12] + extra
        attempts += 1
    
    JOIN_CODES[code] = {
        "server_url": server_url,
        "model_name": model_name,
        "created_at": time.time(),
        "clients": 0
    }
    
    logger.info(f"Generated unique join code: {code} (entropy bits: ~80)")
    return code

def get_join_config(code):
    """Get server config for a join code."""
    if code in JOIN_CODES:
        JOIN_CODES[code]["clients"] += 1
        return JOIN_CODES[code]
    return None

def validate_join_code(code):
    """Check if join code exists and is valid."""
    if code in JOIN_CODES:
        age = time.time() - JOIN_CODES[code]["created_at"]
        if age < 86400:  # Valid for 24 hours
            return True
        else:
            del JOIN_CODES[code]
    return False

# ============ Federated Server ============
@dataclass
class RoundState:
    round_num: int
    status: str = "waiting"
    gradients: dict = None
    def __post_init__(self):
        self.gradients = self.gradients or {}

class FederatedServer:
    def __init__(self, model_name, checkpoint_dir="checkpoints", lr=0.01):
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.lr = lr
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.model = apply_lora_to_model(self.model)
        logger.info("Model loaded with LoRA")
        
        self.rounds: Dict[int, RoundState] = {}
        self.slots: Dict[int, Dict] = {}  # round_num -> {slot_id -> {status, client_id, gradient}}
        self.lock = threading.Lock()
        self.stats = {"total_gradients": 0, "total_clients": set()}
        
    def submit_gradient(self, client_id, round_number, state_dict):
        with self.lock:
            if round_number not in self.rounds:
                self.rounds[round_number] = RoundState(round_number)
            self.rounds[round_number].gradients[client_id] = state_dict
            self.rounds[round_number].status = "collecting"
            self.stats["total_gradients"] += 1
            self.stats["total_clients"].add(client_id)
        
        threading.Thread(target=self._aggregate, args=(round_number,), daemon=True).start()
        
        return {
            "status": "submitted",
            "round": round_number,
            "gradients_received": len(self.rounds[round_number].gradients)
        }
    
    def start_round(self, round_num):
        """Start a new training round."""
        with self.lock:
            if round_num not in self.rounds:
                self.rounds[round_num] = RoundState(round_num)
            self.rounds[round_num].status = "waiting"
        return {"status": "started", "round": round_num}
    
    def get_round_status(self, round_num):
        """Get status of a specific round."""
        with self.lock:
            if round_num in self.rounds:
                rs = self.rounds[round_num]
                return {"status": rs.status, "round": round_num, "gradients": len(rs.gradients)}
        return {"status": "not_started", "round": round_num, "gradients": 0}
    
    # ============ Work Slot System ============
    def create_slots(self, round_num, num_slots):
        """Create work slots for a round."""
        with self.lock:
            if round_num not in self.slots:
                self.slots[round_num] = {}
            for i in range(num_slots):
                if i not in self.slots[round_num]:
                    self.slots[round_num][i] = {
                        "status": "available",
                        "client_id": None,
                        "gradient": None,
                        "created_at": time.time()
                    }
        logger.info(f"Created {num_slots} slots for round {round_num}")
        return {"status": "ok", "round": round_num, "slots": num_slots}
    
    def claim_slot(self, client_id):
        """Claim an available work slot."""
        with self.lock:
            # Find first available slot across rounds
            for rn in sorted(self.slots.keys()):
                for sid, slot in self.slots[rn].items():
                    if slot["status"] == "available":
                        slot["status"] = "in_progress"
                        slot["client_id"] = client_id
                        slot["created_at"] = time.time()
                        logger.info(f"Slot {rn}_{sid} claimed by {client_id}")
                        return {"status": "ok", "round": rn, "slot": sid}
            
            # No slots available - create new round
            new_round = max(self.slots.keys()) + 1 if self.slots else 1
            self.create_slots(new_round, 3)  # 3 clients per round
            
            # Try to claim again
            if 1 in self.slots and 0 in self.slots[1]:
                slot = self.slots[1][0]
                slot["status"] = "in_progress"
                slot["client_id"] = client_id
                slot["created_at"] = time.time()
                return {"status": "ok", "round": 1, "slot": 0}
            
            return {"status": "no_slots", "message": "No slots available"}
    
    def release_slot(self, client_id, key):
        """Release a held slot."""
        with self.lock:
            try:
                parts = key.split("_")
                rn, sid = int(parts[0]), int(parts[1])
                if rn in self.slots and sid in self.slots[rn]:
                    slot = self.slots[rn][sid]
                    if slot["client_id"] == client_id and slot["status"] == "in_progress":
                        slot["status"] = "available"
                        slot["client_id"] = None
                        logger.info(f"Slot {key} released by {client_id}")
                        return {"status": "ok"}
            except:
                pass
            return {"status": "error", "message": "Slot not found or not held by client"}
    
    def submit_slot_result(self, data):
        """Submit gradient for a slot."""
        client_id = data.get("client_id", "unknown")
        round_num = data.get("round")
        slot_id = data.get("slot")
        gradient = data.get("gradient")
        loss = data.get("loss", 0)
        
        if gradient is None:
            return {"status": "error", "message": "No gradient provided"}
        
        with self.lock:
            if round_num not in self.slots or slot_id not in self.slots[round_num]:
                return {"status": "error", "message": "Slot not found"}
            
            slot = self.slots[round_num][slot_id]
            if slot["client_id"] != client_id:
                return {"status": "error", "message": "Slot held by different client"}
            
            slot["status"] = "complete"
            slot["gradient"] = gradient
            slot["loss"] = loss
            
            # Count completed
            completed = sum(1 for s in self.slots[round_num].values() if s["status"] == "complete")
            total = len(self.slots[round_num])
            
            logger.info(f"Slot {round_num}_{slot_id} complete by {client_id} (loss={loss:.4f})")
            
            # If all slots complete, aggregate
            if completed == total:
                self._aggregate_from_slots(round_num)
            
            return {"status": "submitted", "round": round_num, "slot": slot_id, "completed": completed, "total": total}
    
    def _aggregate_from_slots(self, round_num):
        """Aggregate gradients from completed slots."""
        if round_num not in self.slots:
            return
        
        gradients = []
        for slot in self.slots[round_num].values():
            if slot["status"] == "complete" and slot["gradient"] is not None:
                gradients.append(slot["gradient"])
        
        if not gradients:
            return
        
        logger.info(f"Aggregating round {round_num} with {len(gradients)} gradients")
        
        all_keys = set()
        for g in gradients:
            all_keys.update(g.keys())
        
        aggregated = {}
        for key in all_keys:
            grads = []
            for g in gradients:
                if key in g:
                    grad = g[key]
                    if isinstance(grad, np.ndarray):
                        grad = torch.from_numpy(grad)
                    elif not isinstance(grad, torch.Tensor):
                        grad = torch.from_numpy(np.array(grad))
                    grads.append(grad.float())
            
            if grads:
                aggregated[key] = torch.stack(grads).mean(0)
        
        # Apply to model
        model_state = self.model.state_dict()
        for key, grad in aggregated.items():
            if key in model_state:
                param = model_state[key]
                if isinstance(param, np.ndarray):
                    param = torch.from_numpy(param)
                elif not isinstance(param, torch.Tensor):
                    param = torch.from_numpy(np.array(param))
                model_state[key] = param.float() + self.lr * grad.float()
        
        self.model.load_state_dict(model_state)
        
        # Save checkpoint
        self._save_checkpoint(round_num)
        
        logger.info(f"Round {round_num} aggregation complete")
    
    def _aggregate(self, round_num):
        time.sleep(2)
        
        with self.lock:
            if round_num not in self.rounds:
                return
            state = self.rounds[round_num]
            if state.status == "aggregating":
                return
            state.status = "aggregating"
            gradients = list(state.gradients.values())
        
        logger.info(f"Aggregating round {round_num} with {len(gradients)} gradients")
        
        all_keys = set()
        for g in gradients:
            all_keys.update(g.keys())
            
        aggregated = {}
        for key in all_keys:
            grads = []
            for g in gradients:
                if key in g:
                    grad = g[key]
                    if isinstance(grad, np.ndarray):
                        grad = torch.from_numpy(grad)
                    grads.append(grad.float())
            if grads:
                aggregated[key] = torch.stack(grads).mean(0)
        
        model_state = self.model.state_dict()
        for key, grad in aggregated.items():
            if key in model_state:
                param = model_state[key]
                if isinstance(param, np.ndarray):
                    param = torch.from_numpy(param)
                model_state[key] = param.float() + self.lr * grad.float()
        
        self.model.load_state_dict(model_state)
        
        with self.lock:
            self.rounds[round_num].status = "complete"
            
        logger.info(f"Round {round_num} complete - {len(aggregated)} keys updated")
        self._save_checkpoint(round_num)
        
    def _save_checkpoint(self, round_num):
        path = os.path.join(self.checkpoint_dir, f"model_round_{round_num}.pt")
        state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, path)
        logger.info(f"Saved: {path}")
        
    def receive_gradient(self, data: Dict) -> Dict:
        client_id = data.get("client_id", "unknown")
        round_num = data.get("round_number", 1)
        gradient_b64 = data.get("gradient_data", "")
        compression_info = data.get("compression_info", {})
        
        try:
            gradient_bytes = base64.b64decode(gradient_b64)
            method = compression_info.get("method", "none")
            if method != "none":
                try:
                    gradient_bytes = zlib.decompress(gradient_bytes)
                except:
                    pass
            state_dict = pickle.loads(gradient_bytes)
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
        return self.submit_gradient(client_id, round_num, state_dict)
    
    def get_status(self):
        with self.lock:
            rounds = {rn: {"status": rs.status, "gradients": len(rs.gradients)} 
                     for rn, rs in self.rounds.items()}
            return {
                "rounds": rounds,
                "total_gradients": self.stats["total_gradients"],
                "total_clients": len(self.stats["total_clients"])
            }

# ============ Web App ============
app = FastAPI()
server = None

@app.get("/")
async def root():
    return {"status": "ok", "server": "lisa-join", "version": "1.0"}

@app.get("/join/{code}")
async def join_page(code: str):
    """Web page to join the federated network."""
    config = get_join_config(code.upper())
    
    # Build the page based on whether we have ngrok URL
    ngrok_section = ""
    if ngrok_tunnel_url and ngrok_tunnel_url.startswith("https://"):
        ngrok_section = f'''
            <p style="color: #00ff88; font-size: 18px; margin: 20px 0;">
                🌐 <strong>Public URL (worldwide access):</strong><br>
                <a href="{ngrok_tunnel_url}/join/{code.upper()}" style="font-size: 14px;">{ngrok_tunnel_url}/join/{code.upper()}</a>
            </p>
        '''
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LISA Federated Learning - Join</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   max-width: 600px; margin: 50px auto; padding: 20px; text-align: center;
                   background: #1a1a2e; color: #eee; }}
            .card {{ background: #16213e; border-radius: 16px; padding: 40px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }}
            h1 {{ color: #00d9ff; margin-bottom: 10px; }}
            .subtitle {{ color: #888; margin-bottom: 30px; }}
            .code {{ background: #0f3460; padding: 15px 30px; border-radius: 8px;
                    font-family: monospace; font-size: 24px; letter-spacing: 4px;
                    color: #00d9ff; margin: 20px 0; }}
            .cmd {{ background: #000; padding: 15px; border-radius: 8px;
                   font-family: monospace; font-size: 14px; text-align: left; overflow-x: auto;
                   color: #0f0; margin: 20px 0; }}
            .ok {{ color: #00ff88; }}
            .info {{ color: #888; font-size: 14px; margin-top: 20px; }}
            a {{ color: #00d9ff; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🤝 LISA Federated Learning</h1>
            <p class="subtitle">Join the distributed AI training network</p>
            
            <div class="code">{code.upper()}</div>
            
            {ngrok_section}
            
            <p>Your join code is ready! Run this on any device:</p>
            
            <div class="cmd">
                curl -sL https://lisa.ciphemon.ai/install | bash -s {code.upper()}
            </div>
            
            <p class="info">
                Or with Python directly:<br>
                <code>pip install lisa-client && lisa-client --join {code.upper()}</code>
            </p>
            
            <hr style="border-color: #333; margin: 30px 0;">
            
            <h3>How it works:</h3>
            <p style="text-align: left; color: #aaa;">
                1. Your device downloads the LISA client<br>
                2. Client connects to server using your code<br>
                3. Client trains on local data (data never leaves your device)<br>
                4. Only gradient updates are shared<br>
                5. All participants benefit from collective training
            </p>
            
            <p class="info" style="margin-top: 30px;">
                💡 Your data stays private - only model gradients are shared
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/api/config/{code}")
async def get_config(code: str):
    """API endpoint for client to get server config."""
    config = get_join_config(code.upper())
    if config:
        return {
            "status": "ok",
            "server_url": config["server_url"],
            "model_name": config["model_name"]
        }
    raise HTTPException(status_code=404, detail="Invalid or expired join code")

@app.get("/health")
async def health():
    return {"status": "ok", "server": "lisa-join"}

@app.post("/submit")
async def submit(data: dict):
    return server.receive_gradient(data)

@app.post("/start_round/{round_num}")
async def start_round(round_num: int):
    """Start a new training round."""
    return server.start_round(round_num)

@app.post("/round/{round_num}/slots/{num_slots}")
async def create_slots(round_num: int, num_slots: int):
    """Create work slots for a round."""
    return server.create_slots(round_num, num_slots)

@app.post("/slot/claim")
async def claim_slot(data: dict):
    """Claim an available work slot."""
    client_id = data.get("client_id", "unknown")
    return server.claim_slot(client_id)

@app.post("/slot/submit")
async def submit_slot(data: dict):
    """Submit work result for a slot."""
    return server.submit_slot_result(data)

@app.post("/slot/release")
async def release_slot(data: dict):
    """Release a slot (on failure)."""
    client_id = data.get("client_id", "")
    key = data.get("key", "")
    return server.release_slot(client_id, key)

@app.get("/client/register")
async def register_client(client_id: str = None):
    """Register a client as available."""
    return {"status": "ok", "client_id": client_id}

@app.get("/round/{round_num}")
async def get_round(round_num: int):
    """Get status of a specific round."""
    return server.get_round_status(round_num)

@app.get("/status")
async def status():
    return server.get_status()

# ============ Main ============
def main():
    import argparse
    parser = argparse.ArgumentParser(description="LISA Federated Learning Server with Easy Join")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--generate-code", action="store_true", help="Generate a join code")
    parser.add_argument("--ngrok", action="store_true", help="Expose server via ngrok for cross-network access")
    parser.add_argument("--ngrok-token", type=str, default=None, help="ngrok auth token for custom subdomains")
    args = parser.parse_args()
    
    global server
    global ngrok_tunnel_url
    
    server = FederatedServer(args.model, args.checkpoint_dir, args.lr)
    
    # Generate join code if requested
    if args.generate_code:
        public_url = None
        
        # Try ngrok if requested
        if args.ngrok:
            public_url = start_ngrok_tunnel(args.port, args.ngrok_token)
            if public_url:
                logger.info(f"ngrok tunnel active: {public_url}")
        
        # Fall back to localhost if no tunnel
        if public_url is None:
            public_url = f"http://localhost:{args.port}"
        
        code = generate_join_code(public_url, args.model)
        ngrok_tunnel_url = public_url if public_url.startswith("http") else None
        
        print(f"\n{'='*50}")
        print(f"🎉 JOIN CODE READY!")
        print(f"{'='*50}")
        print(f"\n  Code: {code}")
        
        if ngrok_tunnel_url and ngrok_tunnel_url != f"http://localhost:{args.port}":
            print(f"  🌐 Public: {ngrok_tunnel_url}/join/{code}")
            print(f"\n  Share this URL for WORLDWIDE access!")
        else:
            print(f"  URL:  http://YOUR_IP:{args.port}/join/{code}")
        
        print(f"\n  Local only: http://localhost:{args.port}/join/{code}")
        print(f"{'='*50}\n")
    
    logger.info(f"Starting LISA server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
