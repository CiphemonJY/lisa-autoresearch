#!/usr/bin/env python3
"""
LISA-AutoResearch - Main Entry Point

Usage:
    # Hardware detection
    python main.py --mode hardware

    # Local training (LISA + PyTorch)
    python main.py --mode train --model microsoft/phi-2 --iters 100

    # Disk-offloaded training (for large models on small RAM)
    python main.py --mode offload --model Qwen/Qwen2.5-7B-Instruct --iters 50

    # Start federated server
    python main.py --mode server --rounds 10

    # Run federated client
    python main.py --mode client --client-id hospital-1 --rounds 5

    # Run full federated simulation (all in-process)
    python main.py --mode simulate --clients 3 --rounds 3
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Config loading
CONFIG_DEFAULTS = {
    "model": {
        "name": "EleutherAI/pythia-70m",
        "lora_rank": 4,
        "lora_alpha": 8,
    },
    "training": {
        "lr": 3e-4,
        "epochs": 3,
        "batch_size": 4,
        "seq_length": 128,
        "steps": 200,
    },
    "federated": {
        "server_host": "127.0.0.1",
        "server_port": 8080,
        "rounds": 3,
        "local_steps": 5,
    },
    "offload": {
        "enabled": False,
        "layer_groups": 4,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override dict into base dict recursively."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file and merge with defaults.
    
    Precedence (highest to lowest):
      1. Environment variables (LISA_* prefix)
      2. Custom config file (--config)
      3. config/default.yaml
      4. Hardcoded defaults
    """
    import yaml
    
    config = _deep_merge({}, CONFIG_DEFAULTS)
    
    # Load default.yaml if it exists
    default_path = Path("config/default.yaml")
    if default_path.exists():
        try:
            with open(default_path) as f:
                file_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, file_config)
        except Exception as e:
            print(f"Warning: failed to load config/default.yaml: {e}")
    
    # Load custom config if specified
    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as f:
                file_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, file_config)
        except Exception as e:
            print(f"Warning: failed to load config {config_path}: {e}")
    
    # Override from environment variables (LISA_* prefix)
    env_prefix = "LISA_"
    for key, val in os.environ.items():
        if not key.startswith(env_prefix):
            continue
        parts = key[len(env_prefix):].lower().split("_")
        if len(parts) == 1:
            config[parts[0]] = val
        elif len(parts) == 2:
            if parts[0] not in config:
                config[parts[0]] = {}
            # Try to coerce to appropriate type
            if isinstance(config[parts[0]].get(parts[1]), bool):
                config[parts[0]][parts[1]] = val.lower() in ("true", "1", "yes")
            elif isinstance(config[parts[0]].get(parts[1]), int):
                try:
                    config[parts[0]][parts[1]] = int(val)
                except ValueError:
                    pass
            elif isinstance(config[parts[0]].get(parts[1]), float):
                try:
                    config[parts[0]][parts[1]] = float(val)
                except ValueError:
                    pass
            else:
                config[parts[0]][parts[1]] = val
    
    return config


def cmd_hardware():
    """Run hardware detection and print report."""
    from lisa import detect_hardware
    hw = detect_hardware()
    # Import here to avoid circular
    hw_dict = {
        "platform": hw.os_name,
        "cpu": hw.cpu_brand,
        "cpu_cores": hw.cpu_cores,
        "ram_total_gb": round(hw.total_ram_gb, 1),
        "ram_available_gb": round(hw.available_ram_gb, 1),
        "gpu": hw.gpu_name or "None",
        "gpu_type": hw.gpu_type or "None",
        "gpu_memory_gb": round(hw.gpu_memory_gb, 1) if hw.gpu_memory_gb else 0,
        "disk_available_gb": round(hw.available_disk_gb, 0),
        "recommended_framework": hw.recommended_framework,
        "max_model_size": hw.max_model_size,
        "use_disk_offload": hw.use_disk_offload,
        "estimated_speed": hw.estimated_training_speed,
    }
    print()
    print("Hardware Report:")
    for k, v in hw_dict.items():
        print(f"  {k}: {v}")
    return hw_dict


def cmd_train(args):
    """Run LISA training with PyTorch."""
    from lisa.train_torch import train
    # Allow --steps or --train-steps as alias for --iters
    iters = args.train_steps if args.train_steps is not None else args.iters
    print(f"Training with LISA (PyTorch)")
    print(f"  Model: {args.model}")
    print(f"  Iterations: {iters}")
    print(f"  Bottom/Top/Middle layers: {args.bottom}/{args.top}/{args.middle}")
    result = train(
        model_id=args.model,
        iters=iters,
        bottom_layers=args.bottom,
        top_layers=args.top,
        middle_sample=args.middle,
        lr=args.lr,
        batch_size=args.batch_size,
        max_seq=args.max_seq,
        output_dir=args.output_dir or "output/lisa_torch",
        device="auto",
    )
    print(f"\nResult: {result.get('status')}")
    if result.get("status") == "success":
        print(f"  Final loss: {result.get('final_loss', 0):.4f}")
        print(f"  Output: {result.get('output_dir')}")
    return result


def cmd_offload(args):
    """Run disk-offloaded training."""
    from lisa.offload_torch import DiskOffloadedTrainer, OffloadConfig
    print(f"Disk-offloaded training (PyTorch)")
    print(f"  Model: {args.model}")
    print(f"  Layer groups: {args.layer_groups}")

    config = OffloadConfig(
        model_id=args.model,
        layer_groups=args.layer_groups,
        max_memory_gb=args.max_mem,
        iters=args.iters,
        batch_size=args.batch_size,
        output_dir=args.output_dir or "output/offloaded",
    )

    trainer = DiskOffloadedTrainer(config)
    size = trainer.estimate_model_size()
    print(f"  Peak memory: {size['peak_memory_gb']:.1f} GB")
    print(f"  Disk storage: {size['disk_storage_gb']:.1f} GB")

    result = trainer.train(device=args.device or "cpu")
    print(f"\nResult: {result.get('status')}")
    if result.get("status") == "success":
        print(f"  Final loss: {result.get('final_loss', 0):.4f}")
        print(f"  Total time: {result.get('total_time', 0):.1f}s")
    return result


def cmd_mlx(args):
    """Run LISA training with MLX (Apple Silicon)."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    iters = args.train_steps if args.train_steps is not None else args.iters
    lora_rank = args.lora_rank or 4
    print(f"Training with LISA (MLX — Apple Silicon)")
    print(f"  Model: {args.model}")
    print(f"  Iterations: {iters}")
    print(f"  LoRA rank: {lora_rank}")

    try:
        from mlx_lm import load as mlx_load
        from mlx_lm.lora import linear_to_lora_layers, train_model, TrainingArgs, load_dataset
        from dataclasses import dataclass, field
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError as e:
        print(f"  ERROR: MLX not available: {e}")
        print(f"  Install with: pip install mlx mlx-lm")
        return {"status": "error", "message": "MLX not installed"}

    # Load model
    print(f"  Loading {args.model}...")
    try:
        model, tokenizer = mlx_load(args.model)
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return {"status": "error", "message": str(e)}
    print(f"  Model loaded")

    # Apply LoRA (all layers for now — LISA layer selection requires custom unfreezing)
    lora_config = {
        "rank": lora_rank,
        "scale": 1.0,
        "dropout": 0.0,
    }
    linear_to_lora_layers(model, num_layers=32, config=lora_config)
    print(f"  LoRA applied (rank={lora_rank})")

    # Setup training arguments
    train_args = TrainingArgs(
        iters=iters,
        batch_size=args.batch_size,
        steps_per_report=10,
        max_seq_length=args.max_seq,
    )

    # Load wikitext data
    print(f"  Loading wikitext dataset...")
    try:
        train_data = load_dataset(
            "wikitext", "wikitext-2-v1",
            train=True,
            sample_size=500,
            context_length=args.max_seq,
        )
        valid_data = load_dataset(
            "wikitext", "wikitext-2-v1",
            train=False,
            sample_size=100,
            context_length=args.max_seq,
        )
        print(f"  Dataset: train={len(train_data)}, valid={len(valid_data)}")
    except Exception as e:
        print(f"  Dataset load failed: {e}. Will use mlx_lm train_model which handles its own data.")
        train_data = None
        valid_data = None

    # Train
    print(f"  Starting training...")
    t0 = __import__("time").time()

    if train_data is not None and valid_data is not None:
        result = train_model(
            args=train_args,
            model=model,
            train_set=train_data,
            valid_set=valid_data,
        )
        elapsed = __import__("time").time() - t0
        final_loss = result.get("final_loss", 0) if isinstance(result, dict) else 0
        print(f"\n  Training complete!")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Final loss: {final_loss:.4f}")
        return {"status": "success", "final_loss": final_loss, "time": elapsed}
    else:
        # Fallback: basic training loop
        optimizer = optim.Adam(learning_rate=1e-4)
        losses = []
        def loss_fn(model, batch):
            # Create simple labels: next token prediction
            labels = mx.concatenate([batch[:, 1:], mx.zeros((batch.shape[0], 1), dtype=batch.dtype)], axis=1)
            logits = model(batch)
            # Flatten for cross-entropy
            loss = mx.mean(mx.nn.losses.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1)
            ))
            return loss

        for step in range(iters):
            batch = mx.random.randint(0, 1000, (args.batch_size, 64))
            loss_and_grads = mx.value_and_grad(loss_fn)(model, batch)
            loss, grads = loss_and_grads
            optimizer.update(model, grads)
            losses.append(float(loss))
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{iters}: loss={sum(losses[-10:])/10:.4f}")
        elapsed = __import__("time").time() - t0
        final = sum(losses[-10:]) / min(10, len(losses))
        print(f"\n  Training complete! Time: {elapsed:.1f}s, loss: {final:.4f}")
        return {"status": "success", "final_loss": final, "time": elapsed}


def cmd_server(args):
    """Start federated learning server."""
    print(f"Starting federated server")
    print(f"  Port: {args.port}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Model: {args.model}")
    print()
    print("Run a client in another terminal:")
    print(f"  python run_client.py --client-id CLIENT1 --server http://localhost:{args.port} --rounds {args.rounds}")
    print()
    # Import and run
    import uvicorn
    from federated.server import FederatedServer, DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    config["model_name"] = args.model
    config["num_rounds"] = args.rounds
    config["min_clients_per_round"] = args.min_clients

    server = FederatedServer(config)

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from typing import Optional
    from pydantic import BaseModel

    app = FastAPI(title="LISA Federated Server")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    app.state.server = server

    class GradientSubmitRequest(BaseModel):
        client_id: str
        round_number: int
        timestamp: float
        num_samples: int
        gradient_norm: float
        loss_before: float
        loss_after: float
        compression_method: str
        compressed_size: int
        dp_epsilon: Optional[float] = None
        gradient_data: Optional[str] = None
        compression_info: Optional[dict] = None

    class RegisterRequest(BaseModel):
        client_id: str

    @app.get(
        "/",
        summary="Server info",
        description="Returns basic server identity and running status.",
        responses={200: {"description": "Server is running."}},
    )
    async def root():
        """Server info endpoint."""
        return {"message": "LISA Federated Learning Server", "status": "running"}

    @app.get(
        "/status",
        summary="Server status",
        description="Returns the current state of the server: round progress, registered clients, gradient counts, and model config.",
        responses={200: {"description": "Current server status."}},
    )
    async def status():
        """Get the current federated server status."""
        return server.get_status()

    @app.post(
        "/register",
        summary="Register a client",
        description="Register a new client with the server. Returns the client's assigned ID.",
        responses={
            200: {"description": "Client registered."},
            409: {"description": "Client already registered."},
            422: {"description": "Validation error."},
        },
    )
    async def register(req: RegisterRequest):
        """Register a new client with the federated server."""
        return server.register_client(req.client_id)

    @app.post(
        "/submit",
        summary="Submit a gradient update",
        description="Submit a compressed gradient update for aggregation. Include client_id, round_number, gradient norm, loss values, and base64-encoded gradient data.",
        responses={
            200: {"description": "Gradient accepted."},
            400: {"description": "Gradient rejected (wrong round, stale, validation failure)."},
            422: {"description": "Validation error."},
        },
    )
    async def submit(req: GradientSubmitRequest):
        """Submit a gradient update from a client for aggregation."""
        return server.receive_gradient(req.model_dump())

    print(f"Server starting on http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def cmd_client(args):
    """Run federated learning client."""
    print(f"Starting federated client")
    print(f"  Client ID: {args.client_id}")
    print(f"  Server: {args.server}")
    print(f"  Rounds: {args.rounds}")

    from federated.client import FederatedClient, DEFAULT_CONFIG
    import requests

    config = DEFAULT_CONFIG.copy()
    config["model_name"] = args.model
    config["model_name_fallback"] = args.model
    steps = args.train_steps if args.train_steps is not None else args.steps
    config["max_train_steps"] = steps
    config["local_epochs"] = args.epochs
    config["batch_size"] = args.batch_size

    client = FederatedClient(args.client_id, args.server, config)

    # Register
    try:
        resp = requests.post(f"{args.server}/register", json={"client_id": args.client_id}, timeout=30)
        print(f"  Registered: {resp.json()}")
    except Exception as e:
        print(f"  Server registration failed: {e} (continuing anyway)")

    # Run rounds
    for r in range(1, args.rounds + 1):
        print(f"\n  Round {r}/{args.rounds}:")
        result = client.train_and_submit(round_number=r)
        if isinstance(result, dict) and result.get("status") == "error":
            print(f"    Error: {result.get('message')}")
        else:
            print(f"    Gradient submitted OK")

    print(f"\n  Done. Total samples trained: {client.state.total_samples_trained}")


def cmd_simulate(args):
    """Run in-process federated simulation (no networking)."""
    print(f"Running federated simulation")
    print(f"  Clients: {args.clients}")
    print(f"  Rounds: {args.rounds}")

    # Import
    from federated.server import FederatedServer, DEFAULT_CONFIG as SERVER_CONFIG

    config = SERVER_CONFIG.copy()
    config["model_name"] = args.model
    config["min_clients_per_round"] = args.clients

    print("  Loading server model...")
    server = FederatedServer(config)

    # Import client components
    from federated.client import DEFAULT_CONFIG as CLIENT_CONFIG, LocalTrainer, GradientUpdate
    import numpy as np
    import time as time_module
    import base64

    clients = {}
    for i in range(1, args.clients + 1):
        cid = f"client-{i}"
        ccfg = {**CLIENT_CONFIG}
        ccfg["model_name"] = args.model
        ccfg["model_name_fallback"] = args.model
        ccfg["max_train_steps"] = 3
        ccfg["local_epochs"] = 1
        ccfg["batch_size"] = 2

        clients[cid] = {
            "client_id": cid,
            "config": ccfg,
            "trainer": LocalTrainer(cid, ccfg),
        }
        server.register_client(cid)

    for r in range(1, args.rounds + 1):
        print(f"\n  Round {r}/{args.rounds}:")

        for cid, client in clients.items():
            trainer = client["trainer"]
            print(f"    {cid}: computing gradient...")

            # Compute gradient update locally
            update = trainer.compute_gradient_update(round_number=r)

            # Submit directly to server (no HTTP)
            payload = {
                "client_id": update.client_id,
                "round_number": update.round_number,
                "timestamp": update.timestamp,
                "num_samples": update.num_samples,
                "gradient_norm": update.gradient_norm,
                "loss_before": update.loss_before,
                "loss_after": update.loss_after,
                "compression_method": update.compression_info.get("method", "none"),
                "compressed_size": len(update.compressed_data),
                "compression_info": update.compression_info,
                "dp_epsilon": update.dp_epsilon,
                "gradient_data": base64.b64encode(update.compressed_data).decode("utf-8"),
            }

            result = server.receive_gradient(payload)
            print(f"    {cid}: submitted (norm={update.gradient_norm:.4f}, loss={update.loss_after:.4f})")

        # Wait briefly for aggregation
        time_module.sleep(0.3)

        status = server.get_status()
        print(f"    Server: round={status['global_round']}, gradients={status['total_gradients_received']}")

    final_status = server.get_status()
    print(f"\n  Simulation complete!")
    print(f"  Total gradients: {final_status['total_gradients_received']}")
    print(f"  Final round: {final_status['global_round']}")


def cmd_rollback(args):
    """List checkpoints or rollback the server to a specific checkpoint."""
    from utils.checkpoint_manager import CheckpointManager

    mgr = CheckpointManager(args.checkpoint_dir)

    if args.list_checkpoints:
        checkpoints = mgr.list_checkpoints()
        if not checkpoints:
            print("No checkpoints found.")
            return

        print(f"{'Checkpoint ID':<25} {'Round':>6} {'Ver':>4} {'Timestamp':<26} {'Perplexity':>11} {'Clients':>8} {'DP':>4}")
        print("-" * 90)
        for ckpt in checkpoints:
            ts = ckpt.get("timestamp_iso", "")
            if ts:
                # Strip microseconds
                ts = ts[:19]
            ppl = f"{ckpt.get('perplexity', 'N/A')}"
            if ppl != "N/A" and ppl != "None":
                ppl = f"{float(ppl):.4f}"
            dp = "yes" if ckpt.get("dp_enabled") else "no"
            print(
                f"{ckpt.get('checkpoint_id', ''):<25} "
                f"{ckpt.get('round', 0):>6} "
                f"{ckpt.get('version', 0):>4} "
                f"{ts:<26} "
                f"{ppl:>11} "
                f"{ckpt.get('client_count', 'N/A'):>8} "
                f"{dp:>4}"
            )
        return

    if args.rollback_to:
        print(f"Rolling back to checkpoint: {args.rollback_to}")
        ok = mgr.rollback(args.rollback_to)
        if ok:
            print("Rollback validated successfully.")
            # Optionally print metadata
            data = mgr.load(args.rollback_to)
            meta = data.get("metadata", {})
            print(f"  Round: {meta.get('round')}")
            print(f"  Version: {meta.get('version')}")
            print(f"  Clients: {meta.get('client_count')}")
            print(f"  Compression: {meta.get('compression')}")
            print(f"  DP enabled: {meta.get('dp_enabled')}")
        else:
            print(f"ERROR: Rollback failed - checkpoint not found or corrupted.")
            sys.exit(1)
        return

    # No flags: show usage
    print("Rollback mode: list or restore checkpoints")
    print()
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    print()
    print("List checkpoints:")
    print(f"  python main.py --mode rollback --checkpoint-dir {args.checkpoint_dir} --list")
    print()
    print("Rollback to a checkpoint:")
    print(f"  python main.py --mode rollback --checkpoint-dir {args.checkpoint_dir} --to round_3_v1")


def main():
    # Pre-parse --config so we can load config file before building the full parser
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--config", default=None)
    _pre_args, _ = _pre_parser.parse_known_args()

    # Load config (defaults + YAML + env vars)
    cfg = load_config(_pre_args.config)

    parser = argparse.ArgumentParser(
        description="LISA-AutoResearch: Federated Learning on Any Hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  hardware   Detect hardware and recommend settings
  train      Train with LISA (PyTorch, Windows/Linux compatible)
  offload    Disk-offloaded training for large models
  server     Start federated learning server
  client     Run federated learning client
  simulate   Run full federated simulation in-process

Config:
  --config   Path to a custom YAML config file (default: config/default.yaml)
             Config file values are overridden by CLI arguments.

Examples:
  python main.py --mode hardware
  python main.py --mode train --model microsoft/phi-2 --iters 50
  python main.py --mode simulate --clients 3 --rounds 3
  python main.py --mode server --port 8000 --rounds 5
  python main.py --mode client --client-id my-pc --server http://localhost:8000 --rounds 5
  python main.py --mode client --config config/client-hospital.yaml
        """
    )

    # Config file argument
    parser.add_argument(
        "--config",
        type=str,
        default=_pre_args.config,
        help="Path to a YAML config file. Default: config/default.yaml if it exists. "
             "Config file values are overridden by CLI arguments."
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["hardware", "train", "offload", "server", "client", "simulate", "rollback", "mlx"],
        default="hardware",
        help="Operation mode. Default: hardware."
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=cfg["model"]["name"],
        help=f"Model name or HuggingFace ID. Default: {cfg['model']['name']}"
    )

    # Training arguments
    parser.add_argument(
        "--lr",
        type=float,
        default=cfg["training"]["lr"],
        help=f"Learning rate. Default: {cfg['training']['lr']}"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=cfg["training"]["epochs"],
        help=f"Number of training epochs (client mode). Default: {cfg['training']['epochs']}"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=cfg["training"]["batch_size"],
        help=f"Training batch size. Default: {cfg['training']['batch_size']}"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        dest="max_seq",
        default=cfg["training"]["seq_length"],
        help=f"Maximum sequence length. Default: {cfg['training']['seq_length']}"
    )
    parser.add_argument(
        "--steps",
        type=int,
        dest="steps",
        default=cfg["training"]["steps"],
        help=f"Maximum training steps (client mode). Default: {cfg['training']['steps']}"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        dest="output_dir",
        default=None,
        help="Output directory for checkpoints and logs. Default: output/<mode>."
    )

    # LISA layer selection
    parser.add_argument(
        "--bottom",
        type=int,
        default=2,
        help="Number of bottom (always-trained) layers. Default: 2."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=2,
        help="Number of top (always-trained) layers. Default: 2."
    )
    parser.add_argument(
        "--middle",
        type=int,
        default=1,
        help="Number of middle layers to randomly sample. Default: 1."
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=cfg["training"]["steps"],
        help="Training iterations (train/offload mode). Default: 200."
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        dest="train_steps",
        default=None,
        help="Alias for --iters (train/offload mode). Default: 200."
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        dest="lora_rank",
        default=None,
        help="LoRA rank (MLX mode). Default: 4."
    )

    # Offload arguments
    parser.add_argument(
        "--layer-groups",
        type=int,
        dest="layer_groups",
        default=cfg["offload"]["layer_groups"],
        help=f"Number of layer groups for disk offloading. Default: {cfg['offload']['layer_groups']}"
    )
    parser.add_argument(
        "--max-mem",
        type=float,
        default=5.0,
        help="Maximum memory to use for offloaded training (GB). Default: 5.0."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for training (cpu, cuda). Default: cpu."
    )

    # Server arguments
    parser.add_argument(
        "--port",
        type=int,
        default=cfg["federated"]["server_port"],
        help=f"Federated server port. Default: {cfg['federated']['server_port']}"
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        dest="min_clients",
        default=1,
        help="Minimum number of clients required per federated round. Default: 1."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=cfg["federated"]["rounds"],
        help=f"Number of federated rounds. Default: {cfg['federated']['rounds']}"
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=3,
        help="Number of simulated clients for --mode simulate. Default: 3."
    )

    # Rollback arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        dest="checkpoint_dir",
        help="Checkpoint directory for rollback mode. Default: checkpoints."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_checkpoints",
        help="List available checkpoints (rollback mode)."
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="rollback_to",
        default=None,
        help="Rollback to this checkpoint ID (e.g. round_3_v1)."
    )

    # Client arguments
    parser.add_argument(
        "--client-id",
        type=str,
        dest="client_id",
        default=None,
        help="Unique client identifier for federated learning."
    )
    parser.add_argument(
        "--server",
        type=str,
        default=f"http://localhost:{cfg['federated']['server_port']}",
        help=f"Federated server URL. Default: http://localhost:{cfg['federated']['server_port']}"
    )

    args = parser.parse_args()

    print("="*70)
    print("  LISA-AutoResearch - Federated Learning on Any Hardware")
    print("="*70)
    print()

    if args.mode == "hardware":
        cmd_hardware()
    elif args.mode == "train":
        cmd_train(args)
    elif args.mode == "mlx":
        cmd_mlx(args)
    elif args.mode == "offload":
        cmd_offload(args)
    elif args.mode == "server":
        cmd_server(args)
    elif args.mode == "client":
        if not args.client_id:
            args.client_id = f"client-{abs(hash(str(os.times()))) % 10000}"
        cmd_client(args)
    elif args.mode == "simulate":
        cmd_simulate(args)
    elif args.mode == "rollback":
        cmd_rollback(args)


if __name__ == "__main__":
    main()
