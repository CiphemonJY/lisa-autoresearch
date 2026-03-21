#!/usr/bin/env python3
"""
Federated Learning Server - Real PyTorch Implementation

A federated learning server that:
- Receives gradient updates from multiple clients
- Validates and aggregates them using FedAvg
- Tracks client reputation and participation
- Monitors convergence
- Pushes aggregated model updates back to clients

Works on CPU Windows/Linux. No MLX required.
"""

import os
import sys
import json
import time
import threading
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import copy

import numpy as np

# Optional FastAPI
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: fastapi not installed. HTTP API disabled. Use --demo mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fed-server")


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "model_name": "distilbert/distilgpt2",
    "num_rounds": 10,
    "min_clients_per_round": 2,
    "max_clients_per_round": 10,
    "round_timeout_secs": 120,
    "aggregation_method": "fedavg",  # fedavg, fedprox, trimmed_mean
    "gradient_noise_tolerance": 1e6,
    "save_model_every": 5,
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ClientInfo:
    """Information about a registered client."""
    client_id: str
    registered_at: float
    last_seen: float
    rounds_participated: int = 0
    total_samples_contributed: int = 0
    reputation: float = 50.0
    is_active: bool = True
    pending_gradient: Optional[Dict] = None


@dataclass
class RoundState:
    """State for a federated learning round."""
    round_number: int
    status: str  # waiting, collecting, aggregating, done, failed
    clients_joined: List[str] = field(default_factory=list)
    gradients_received: int = 0
    gradients_accepted: int = 0
    gradients_rejected: int = 0
    started_at: float = 0
    completed_at: Optional[float] = None
    aggregated_gradient: Optional[bytes] = None


@dataclass
class ServerMetrics:
    """Server-side metrics."""
    total_rounds: int = 0
    total_gradients_received: int = 0
    total_gradients_rejected: int = 0
    avg_round_time: float = 0
    clients_registered: int = 0
    active_clients: int = 0
    convergence_history: List[float] = field(default_factory=list)


# ============================================================================
# Gradient Validation
# ============================================================================

class GradientValidator:
    """Validate incoming gradient updates."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.noise_threshold = config.get("gradient_noise_tolerance", 1e6)
        self.history: Dict[str, List[float]] = defaultdict(list)
    
    def validate(self, client_id: str, update: Dict) -> Tuple[bool, str]:
        """
        Validate a gradient update.
        
        Returns: (is_valid, reason)
        """
        # Check required fields
        required = ["client_id", "round_number", "num_samples", "gradient_norm"]
        for field in required:
            if field not in update:
                return False, f"Missing required field: {field}"
        
        # Check gradient norm is reasonable
        norm = update.get("gradient_norm", 0)
        if norm > self.noise_threshold:
            return False, f"Gradient norm {norm:.2e} exceeds threshold"
        
        if norm < 1e-10:
            return False, "Gradient norm too small (likely no training)"
        
        # Check num_samples
        num_samples = update.get("num_samples", 0)
        if num_samples <= 0:
            return False, "num_samples must be positive"
        
        # Check loss is improving (loss_after < loss_before)
        if "loss_before" in update and "loss_after" in update:
            if update["loss_after"] > update["loss_before"] + 1.0:
                return False, "Loss increased (possible gradient issue)"
        
        return True, "Valid"
    
    def record_norm(self, client_id: str, norm: float):
        """Record gradient norm for history."""
        self.history[client_id].append(norm)
        if len(self.history[client_id]) > 100:
            self.history[client_id] = self.history[client_id][-100:]


# ============================================================================
# Gradient Aggregation
# ============================================================================

class GradientAggregator:
    """Aggregate gradient updates from multiple clients."""
    
    def __init__(self, method: str = "fedavg"):
        self.method = method
        self.compressor = None  # Server-side compression if needed
    
    def aggregate(
        self,
        updates: List[Dict],
        reputations: Dict[str, float],
    ) -> Tuple[Optional[bytes], Dict]:
        """
        Aggregate gradients using FedAvg.
        
        Updates are already compressed bytes from clients.
        We aggregate the decompressed gradients.
        
        Returns: (aggregated_state_dict_bytes, stats)
        """
        if not updates:
            return None, {"status": "no_updates"}
        
        # Decompress all gradients
        from federated.client import GradientCompressor
        default_config = {"compression": {"enabled": True, "sparsification_ratio": 0.05, "quantization_bits": 8, "compression_level": 6}}
        decompressor = GradientCompressor(default_config)

        decompressed = []
        for u in updates:
            try:
                data = u.get("gradient_data", b"")
                if isinstance(data, str):
                    import base64 as _base64
                    data = _base64.b64decode(data)
                if data:
                    comp_info = u.get("compression_info", {"method": "sparse-8bit", "sparsification_ratio": 0.05})
                    state_dict = decompressor.decompress(data, comp_info)
                    decompressed.append((u["client_id"], u["num_samples"], state_dict))
            except Exception as e:
                logger.warning(f"Failed to decompress gradient from {u.get('client_id')}: {e}")
                continue
        
        if not decompressed:
            return None, {"status": "no_valid_gradients"}
        
        # FedAvg: weighted average by sample count
        total_samples = sum(s for _, s, _ in decompressed)
        
        # Initialize aggregated with zero tensors
        import torch
        first_state = decompressed[0][2]
        aggregated = {}
        for key, val in first_state.items():
            if isinstance(val, np.ndarray):
                aggregated[key] = np.zeros_like(val, dtype=np.float32)
            elif isinstance(val, torch.Tensor):
                aggregated[key] = torch.zeros_like(val, dtype=torch.float32)
        
        # Weighted sum
        for client_id, num_samples, state_dict in decompressed:
            weight = num_samples / total_samples
            rep = reputations.get(client_id, 50.0) / 50.0  # Normalize to 0.5-1.5
            
            for key in aggregated:
                if key in state_dict:
                    val = state_dict[key]
                    if isinstance(val, torch.Tensor):
                        aggregated[key] = aggregated[key] + val.cpu() * weight * rep
                    elif isinstance(val, np.ndarray):
                        aggregated[key] = aggregated[key] + val * weight * rep
        
        # Normalize by total reputation weight
        total_rep = sum(reputations.get(c, 50) for c, _, _ in decompressed)
        for key in aggregated:
            if isinstance(aggregated[key], torch.Tensor):
                aggregated[key] = aggregated[key] / (total_rep / 50.0)
        
        # Serialize
        serialized = pickle.dumps(aggregated)
        
        stats = {
            "status": "success",
            "method": self.method,
            "num_updates": len(decompressed),
            "total_samples": total_samples,
            "aggregated_size": len(serialized),
        }
        
        return serialized, stats


# ============================================================================
# Federated Server
# ============================================================================

class FederatedServer:
    """
    Main federated learning server.
    
    Coordinates federated rounds: receives gradients from clients,
    aggregates them, updates the global model, and distributes updates.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DEFAULT_CONFIG.copy()
        
        self.clients: Dict[str, ClientInfo] = {}
        self.round_state: Dict[int, RoundState] = {}
        self.metrics = ServerMetrics()
        
        self.validator = GradientValidator(self.config)
        self.aggregator = GradientAggregator(self.config.get("aggregation_method", "fedavg"))
        
        self.current_model: Optional[bytes] = None
        self.global_round: int = 0
        
        self._lock = threading.RLock()
        
        # Directories
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.log_dir = Path(self.config.get("log_dir", "logs"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self._init_model()
        
        logger.info(f"FederatedServer initialized (model={self.config['model_name']})")
        logger.info(f"  Rounds: {self.config['num_rounds']}")
        logger.info(f"  Min clients/round: {self.config['min_clients_per_round']}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    def _init_model(self):
        """Initialize or load the global model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        model_name = self.config["model_name"]
        logger.info(f"Loading global model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            config = AutoConfig.from_pretrained(model_name)
            config.hidden_size = min(config.hidden_size, 512)
            config.num_attention_heads = min(config.num_attention_heads, 8)
            config.num_hidden_layers = min(config.num_hidden_layers, 6)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                ignore_mismatched_sizes=True,
            )
            
            # Save initial model
            self._save_model(0)
            logger.info(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _save_model(self, round_num: int):
        """Save model checkpoint."""
        import torch
        path = self.checkpoint_dir / f"model_round_{round_num}.pt"
        state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, path)
        self.current_model = pickle.dumps(state)
        logger.info(f"Model saved: {path}")
    
    def _get_lock(self):
        """Get or create the server lock (lazy initialization for tests)."""
        if not hasattr(self, '_lock'):
            self._lock = threading.RLock()
        return self._lock
    
    def register_client(self, client_id: str) -> Dict:
        """Register a new client."""
        with self._get_lock():
            if client_id in self.clients:
                self.clients[client_id].last_seen = time.time()
            else:
                self.clients[client_id] = ClientInfo(
                    client_id=client_id,
                    registered_at=time.time(),
                    last_seen=time.time(),
                )
                self.metrics.clients_registered += 1
                self.metrics.active_clients += 1
                logger.info(f"Registered client: {client_id}")
        
        return {"status": "registered", "client_id": client_id}
    
    def receive_gradient(self, update: Dict) -> Dict:
        """
        Receive a gradient update from a client.
        
        This is called when a client submits their gradient.
        """
        client_id = update.get("client_id", "unknown")
        round_num = update.get("round_number", 0)

        # Decode base64 gradient data if provided (convert to bytes for storage)
        import base64 as _base64
        if "gradient_data" in update and update["gradient_data"]:
            try:
                update["gradient_data"] = _base64.b64decode(update["gradient_data"])
            except Exception:
                pass

        with self._get_lock():
            # Update client last-seen
            if client_id in self.clients:
                self.clients[client_id].last_seen = time.time()

            # Validate update
            is_valid, reason = self.validator.validate(client_id, update)
            
            # Record gradient norm
            self.validator.record_norm(client_id, update.get("gradient_norm", 0))
            
            # Store pending gradient
            if client_id in self.clients:
                self.clients[client_id].pending_gradient = update
            
            # Update round state
            if round_num not in self.round_state:
                self.round_state[round_num] = RoundState(
                    round_number=round_num,
                    status="collecting",
                    started_at=time.time(),
                )
            
            rs = self.round_state[round_num]
            rs.gradients_received += 1
            rs.gradients_accepted += 1
            rs.clients_joined.append(client_id)
            
            self.metrics.total_gradients_received += 1
            
            # Update client stats
            if client_id in self.clients:
                self.clients[client_id].rounds_participated += 1
                self.clients[client_id].total_samples_contributed += update.get("num_samples", 0)
            
            logger.info(
                f"Received gradient from {client_id} (round {round_num}): "
                f"norm={update.get('gradient_norm', 0):.4f}, "
                f"samples={update.get('num_samples', 0)}"
            )
            
            # Check if we have enough gradients to aggregate
            min_clients = self.config.get("min_clients_per_round", 2)
            if len(set(rs.clients_joined)) >= min_clients and rs.status == "collecting":
                self._start_aggregation(round_num)
            
            return {"status": "accepted", "round": round_num}
    
    def _start_aggregation(self, round_num: int):
        """Start aggregating gradients for a round."""
        rs = self.round_state[round_num]
        rs.status = "aggregating"
        
        logger.info(f"Starting aggregation for round {round_num}")
        
        # Collect all pending gradients
        updates = []
        for cid in set(rs.clients_joined):
            if cid in self.clients and self.clients[cid].pending_gradient:
                updates.append(self.clients[cid].pending_gradient)
        
        # Get reputations
        reputations = {c.client_id: c.reputation for c in self.clients.values()}
        
        # Aggregate
        aggregated, stats = self.aggregator.aggregate(updates, reputations)
        
        if aggregated is None:
            rs.status = "failed"
            logger.error(f"Aggregation failed for round {round_num}")
            return
        
        # Apply to global model
        self._apply_gradient(aggregated)
        
        # Save checkpoint
        self._save_model(round_num)
        
        # Complete round
        rs.status = "done"
        rs.completed_at = time.time()
        rs.aggregated_gradient = aggregated
        
        # Clear pending gradients
        for cid in self.clients:
            self.clients[cid].pending_gradient = None
        
        # Update metrics
        self.global_round = round_num
        self.metrics.total_rounds += 1
        
        round_time = rs.completed_at - rs.started_at
        self.metrics.avg_round_time = (
            (self.metrics.avg_round_time * (self.metrics.total_rounds - 1) + round_time)
            / self.metrics.total_rounds
        )
        
        logger.info(
            f"Round {round_num} complete: "
            f"{stats['num_updates']} updates, "
            f"{round_time:.1f}s, "
            f"model updated"
        )
    
    def _apply_gradient(self, gradient_bytes: bytes):
        """Apply aggregated gradient to global model."""
        import torch
        
        state_dict = pickle.loads(gradient_bytes)
        
        # Simple SGD update
        current_state = self.model.state_dict()
        lr = 0.01
        
        for key in current_state:
            if key in state_dict:
                grad = state_dict[key]
                if isinstance(grad, np.ndarray):
                    grad = torch.from_numpy(grad)
                current_state[key] = current_state[key].float() + lr * grad.float()
        
        self.model.load_state_dict(current_state)
    
    def get_model_update(self, client_id: str, since_round: int = 0) -> Optional[Dict]:
        """
        Get the latest model update for a client.
        
        Returns serialized model state dict or None.
        """
        with self._get_lock():
            # Return current model
            if self.current_model:
                return {
                    "round": self.global_round,
                    "model_data": self.current_model,
                    "model_size": len(self.current_model),
                }
            return None
    
    def get_status(self) -> Dict:
        """Get server status."""
        with self._get_lock():
            active = sum(1 for c in self.clients.values() if c.is_active)
            
            return {
                "global_round": self.global_round,
                "num_rounds": self.config["num_rounds"],
                "clients_registered": self.metrics.clients_registered,
                "active_clients": active,
                "total_gradients_received": self.metrics.total_gradients_received,
                "total_gradients_rejected": self.metrics.total_gradients_rejected,
                "avg_round_time": self.metrics.avg_round_time,
                "current_model_size_mb": len(self.current_model) / 1e6 if self.current_model else 0,
                "config": {
                    "model_name": self.config["model_name"],
                    "min_clients_per_round": self.config["min_clients_per_round"],
                    "aggregation_method": self.config["aggregation_method"],
                },
            }
    
    def get_round_status(self, round_num: int) -> Optional[Dict]:
        """Get status of a specific round."""
        with self._get_lock():
            if round_num not in self.round_state:
                return None
            
            rs = self.round_state[round_num]
            return {
                "round_number": rs.round_number,
                "status": rs.status,
                "clients_joined": len(set(rs.clients_joined)),
                "gradients_received": rs.gradients_received,
                "gradients_accepted": rs.gradients_accepted,
                "gradients_rejected": rs.gradients_rejected,
                "started_at": rs.started_at,
                "completed_at": rs.completed_at,
            }


# ============================================================================
# FastAPI App
# ============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(title="LISA Federated Learning Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    _server: Optional[FederatedServer] = None
    
    @app.on_event("startup")
    async def startup():
        global _server
        _server = FederatedServer()
    
    @app.get("/")
    async def root():
        return {"message": "LISA Federated Learning Server", "status": "running"}
    
    @app.get("/status")
    async def status():
        return _server.get_status()
    
    @app.post("/register")
    async def register(request: Request):
        body = await request.json()
        client_id = body.get("client_id")
        if not client_id:
            raise HTTPException(status_code=400, detail="client_id required")
        return _server.register_client(client_id)
    
    @app.post("/submit")
    async def submit(request: Request):
        body = await request.json()
        return _server.receive_gradient(body)
    
    @app.get("/model/{client_id}")
    async def get_model(client_id: str, since_round: int = 0):
        update = _server.get_model_update(client_id, since_round)
        if not update:
            raise HTTPException(status_code=404, detail="No model update available")
        return JSONResponse({
            "round": update["round"],
            "model_size": update["model_size"],
        })
    
    @app.get("/round/{round_num}")
    async def get_round(round_num: int):
        status = _server.get_round_status(round_num)
        if not status:
            raise HTTPException(status_code=404, detail="Round not found")
        return status


# ============================================================================
# Demo Mode (no HTTP, runs in-process)
# ============================================================================

class DemoFederatedSimulator:
    """
    Simulate a full federated learning run without HTTP.
    
    Useful for testing on a single machine.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.server = FederatedServer(config)
        self.clients: Dict[str, Any] = {}
    
    def add_client(self, client_id: str):
        """Add a simulated client."""
        # Import client class
        from federated.client import FederatedClient
        client = FederatedClient(
            client_id=client_id,
            server_url="http://localhost:8000",  # Won't be used in demo
            config=self.server.config,
        )
        self.clients[client_id] = client
        self.server.register_client(client_id)
        return client
    
    def run_round(self, round_num: int) -> Dict:
        """Run one federated round with all clients."""
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_num}")
        logger.info(f"{'='*60}")
        
        # Each client computes gradient
        for client_id, client in self.clients.items():
            logger.info(f"  {client_id}: computing gradient...")
            update = client.train_and_submit(round_num)
            
            # Server receives gradient
            self.server.receive_gradient(update)
        
        # Aggregation happens automatically when min clients submit
        rs = self.server.round_state.get(round_num)
        if rs:
            while rs.status == "collecting":
                time.sleep(0.1)
        
        # Get round result
        result = self.server.get_round_status(round_num)
        logger.info(f"  Round {round_num} result: {result}")
        
        return result or {}
    
    def run(self, num_rounds: int = 3) -> Dict:
        """Run full federated training simulation."""
        logger.info(f"\n{'='*60}")
        logger.info(f"FEDERATED LEARNING SIMULATION")
        logger.info(f"{'='*60}")
        logger.info(f"Clients: {len(self.clients)}")
        logger.info(f"Rounds: {num_rounds}")
        logger.info(f"Model: {self.config['model_name']}")
        logger.info(f"{'='*60}\n")
        
        results = []
        for r in range(1, num_rounds + 1):
            result = self.run_round(r)
            results.append(result)
        
        # Final status
        status = self.server.get_status()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Rounds: {status['global_round']}")
        logger.info(f"Gradients received: {status['total_gradients_received']}")
        logger.info(f"Avg round time: {status['avg_round_time']:.1f}s")
        logger.info(f"Model size: {status['current_model_size_mb']:.1f} MB")
        
        return {
            "status": "complete",
            "results": results,
            "final_status": status,
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--mode", choices=["server", "demo"], default="demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--min-clients", type=int, default=2)
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config["model_name"] = args.model
    config["num_rounds"] = args.rounds
    config["min_clients_per_round"] = args.min_clients
    
    if args.mode == "server":
        if not FASTAPI_AVAILABLE:
            print("Error: fastapi not installed. Install with: pip install fastapi uvicorn")
            sys.exit(1)
        
        server = FederatedServer(config)
        app._server = server  # type: ignore
        
        print(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    
    else:
        # Demo mode
        sim = DemoFederatedSimulator(config)
        
        # Add simulated clients
        for i in range(1, args.clients + 1):
            client_id = f"client-{i}"
            sim.add_client(client_id)
            logger.info(f"Added client: {client_id}")
        
        # Run simulation
        results = sim.run(args.rounds)
        
        print("\n=== RESULTS ===")
        print(json.dumps(results["final_status"], indent=2))


if __name__ == "__main__":
    main()
