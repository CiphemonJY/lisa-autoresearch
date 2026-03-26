#!/usr/bin/env python3
"""
LISA Robust Federated Client System
- Auto-retry on failure with exponential backoff
- Clients claim work slots dynamically
- Timeout handling - if client fails, slot becomes available
- Server tracks which slots are filled
"""
import os, sys, time, json, threading, requests, random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
import base64, pickle, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model

# ============ Config ============
SERVER_URL = "http://10.0.0.43:8080"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CLIENT_ID = f"client_{os.urandom(4).hex()}"
LOG_FILE = "/tmp/lisa_robust_client.log"
MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # seconds
MAX_BACKOFF = 60

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ============ Server API ============
def api_get(path):
    try:
        r = requests.get(f"{SERVER_URL}{path}", timeout=10)
        return r.json()
    except Exception as e:
        log(f"API error GET {path}: {e}")
        return None

def api_post(path, data):
    try:
        r = requests.post(f"{SERVER_URL}{path}", json=data, timeout=60)
        return r.json()
    except Exception as e:
        log(f"API error POST {path}: {e}")
        return None

# ============ Client Model ============
class FederatedClient:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.client_id = CLIENT_ID
        log(f"Client ID: {self.client_id}")
    
    def load_model(self):
        """Load model with LoRA - done once."""
        if self.model is not None:
            return
        
        log(f"Loading {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.eos_token = self.tokenizer.eos_token
        
        config = AutoConfig.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, config=config, trust_remote_code=True, torch_dtype=torch.float32
        )
        
        lora_config = LoraConfig(
            r=4, lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        log(f"Model ready: {sum(p.numel() for p in self.model.parameters()):,} params")
    
    def train(self, steps=10):
        """Train model locally."""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Once upon a time in a far away land,",
            "Machine learning is transforming the world.",
            "Artificial intelligence is the future.",
            "Federated learning enables privacy-preserving AI.",
        ]
        
        total_loss = 0
        for i in range(steps):
            text = texts[i % len(texts)]
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / steps
    
    def get_gradient(self):
        """Extract gradient as state dict."""
        gradient = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradient[name] = param.grad.detach().cpu().numpy()
        return gradient
    
    def apply_gradient(self, gradient):
        """Apply received gradient to model."""
        state = self.model.state_dict()
        for name, grad in gradient.items():
            if name in state:
                if isinstance(grad, list):
                    grad = torch.stack([torch.from_numpy(g) for g in grad]).mean(0)
                elif isinstance(grad, torch.Tensor):
                    grad = grad.float()
                else:
                    grad = torch.from_numpy(grad).float()
                state[name] = state[name].float() + 0.01 * grad
        self.model.load_state_dict(state)

# ============ Work Slot System ============
@dataclass
class WorkSlot:
    round_num: int
    slot_id: int
    status: str = "available"  # available, in_progress, complete, failed
    client_id: str = ""
    created_at: float = field(default_factory=time.time)

class RobustServer:
    """
    Server with robust work slot management.
    - Rounds are divided into slots
    - Clients claim available slots
    - Timeout detection marks slots as available
    """
    
    def __init__(self):
        self.slots: Dict[str, WorkSlot] = {}  # key = "round_slot"
        self.gradients: Dict[str, Dict] = {}  # round_num -> {slot_id -> gradient}
        self.lock = threading.Lock()
        self.timeout = 120  # seconds before slot is considered failed
        
        # Start cleanup thread
        self.running = True
        threading.Thread(target=self._cleanup_loop, daemon=True).start()
    
    def _cleanup_loop(self):
        """Periodically check for timed-out slots."""
        while self.running:
            time.sleep(10)
            self._check_timeouts()
    
    def _check_timeouts(self):
        """Mark timed-out slots as available."""
        now = time.time()
        with self.lock:
            for key, slot in list(self.slots.items()):
                if slot.status == "in_progress" and (now - slot.created_at) > self.timeout:
                    log(f"Slot {key} timed out (was held by {slot.client_id})")
                    slot.status = "available"
                    slot.client_id = ""
    
    def get_available_slot(self, client_id):
        """Client requests work - returns slot or None."""
        with self.lock:
            now = time.time()
            
            # First check for timed-out slots
            self._check_timeouts()
            
            # Find available slot
            for key, slot in self.slots.items():
                if slot.status == "available":
                    slot.status = "in_progress"
                    slot.client_id = client_id
                    slot.created_at = now
                    log(f"Slot {key} claimed by {client_id}")
                    return {
                        "round": slot.round_num,
                        "slot": slot.slot_id,
                        "key": key
                    }
            
            # No slots available
            return None
    
    def register_round(self, round_num, num_slots):
        """Register a new round with N slots."""
        with self.lock:
            for i in range(num_slots):
                key = f"{round_num}_{i}"
                if key not in self.slots:
                    self.slots[key] = WorkSlot(round_num=round_num, slot_id=i)
                    log(f"Registered slot {key}")
    
    def submit_result(self, client_id, round_num, slot_id, gradient, loss):
        """Client submits work result."""
        with self.lock:
            key = f"{round_num}_{slot_id}"
            if key not in self.slots:
                return {"status": "error", "message": "Slot not found"}
            
            slot = self.slots[key]
            if slot.client_id != client_id:
                return {"status": "error", "message": "Slot held by different client"}
            
            slot.status = "complete"
            
            # Store gradient
            if round_num not in self.gradients:
                self.gradients[round_num] = {}
            self.gradients[round_num][slot_id] = gradient
            
            # Check if round is complete
            total_slots = len([k for k in self.slots if k.startswith(f"{round_num}_")])
            completed = len(self.gradients.get(round_num, {}))
            
            return {
                "status": "submitted",
                "round": round_num,
                "slot": slot_id,
                "completed": completed,
                "total": total_slots
            }
    
    def release_slot(self, client_id, key):
        """Client releases a slot (failure)."""
        with self.lock:
            if key in self.slots:
                slot = self.slots[key]
                if slot.client_id == client_id and slot.status == "in_progress":
                    slot.status = "available"
                    slot.client_id = ""
                    log(f"Slot {key} released by {client_id}")
                    return True
        return False
    
    def get_status(self):
        """Get current status."""
        with self.lock:
            rounds = {}
            for key, slot in self.slots.items():
                rn = slot.round_num
                if rn not in rounds:
                    rounds[rn] = {"total": 0, "complete": 0, "available": 0, "in_progress": 0}
                rounds[rn]["total"] += 1
                if slot.status == "complete":
                    rounds[rn]["complete"] += 1
                elif slot.status == "available":
                    rounds[rn]["available"] += 1
                elif slot.status == "in_progress":
                    rounds[rn]["in_progress"] += 1
            
            return {
                "slots": len(self.slots),
                "rounds": rounds,
                "gradients_received": sum(len(g) for g in self.gradients.values())
            }

# ============ Client Loop ============
class FederatedClientLoop:
    def __init__(self):
        self.client = FederatedClient()
        self.server = RobustServer()
        self.backoff = INITIAL_BACKOFF
        self.running = True
    
    def run(self):
        """Main client loop."""
        log("Starting federated client loop...")
        self.client.load_model()
        
        while self.running:
            try:
                # Try to get work
                work = self._request_work()
                
                if work is None:
                    # No work available, wait and retry
                    wait = random.uniform(5, 15)
                    log(f"No work available, waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                
                # Got work
                self.backoff = INITIAL_BACKOFF  # Reset backoff on success
                self._do_work(work)
                
            except KeyboardInterrupt:
                log("Shutting down...")
                self.running = False
                break
            except Exception as e:
                log(f"Error in main loop: {e}")
                time.sleep(self.backoff)
                self.backoff = min(self.backoff * 2, MAX_BACKOFF)
    
    def _request_work(self):
        """Request available work slot from server."""
        # Register as available
        api_post("/client/register", {"client_id": self.client.client_id})
        
        # Try to claim a slot
        result = api_post("/slot/claim", {"client_id": self.client.client_id})
        
        if result and result.get("status") == "ok":
            return result
        
        return None
    
    def _do_work(self, work):
        """Do the assigned work."""
        round_num = work["round"]
        slot_id = work["slot"]
        key = work["key"]
        
        log(f"Working on round {round_num}, slot {slot_id}...")
        
        try:
            # Train
            loss = self.client.train(steps=10)
            log(f"Training complete, loss={loss:.4f}")
            
            # Get gradient
            gradient = self.client.get_gradient()
            
            # Submit
            result = api_post("/slot/submit", {
                "client_id": self.client.client_id,
                "round": round_num,
                "slot": slot_id,
                "gradient": gradient,
                "loss": loss
            })
            
            if result and result.get("status") == "submitted":
                log(f"Submitted! Round {round_num} progress: {result.get('completed')}/{result.get('total')}")
            else:
                log(f"Submission failed: {result}")
                
        except Exception as e:
            log(f"Work failed: {e}")
            # Release slot so others can take it
            api_post("/slot/release", {"client_id": self.client.client_id, "key": key})

# ============ Main ============
if __name__ == "__main__":
    print(f"""
╔════════════════════════════════════════════════════════════╗
║     LISA Robust Federated Client - Auto-retry + Slots     ║
╚════════════════════════════════════════════════════════════╝
Client ID: {CLIENT_ID}
Server: {SERVER_URL}

Features:
- Auto-retries on failure with exponential backoff
- Claims available work slots dynamically
- Timed-out slots are re-claimed by other clients
- Gradient aggregation when all slots complete

Press Ctrl+C to stop.
""")
    
    loop = FederatedClientLoop()
    loop.run()
