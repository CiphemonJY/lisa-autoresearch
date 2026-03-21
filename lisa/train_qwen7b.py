#!/usr/bin/env python3
"""
Train Qwen models with optimal settings.

Supports: 7B, 14B, and larger models (4-bit quantization)

Usage:
    python train_qwen7b.py --iters 500
    python train_qwen7b.py --model 14b --iters 200
    python train_qwen7b.py --config autoresearch_best.tsv
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add LISA package to path for imports
_LISA_PATH = Path(__file__).parent.parent / "packages" / "lisa-autoresearch"
sys.path.insert(0, str(_LISA_PATH))

# Available models
MODELS = {
    "7b": {
        "id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "memory_gb": 4.9,
        "description": "7B model, fast training (recommended)"
    },
    "14b": {
        "id": "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "memory_gb": 8.9,
        "description": "14B model, better quality"
    },
    "3b": {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "memory_gb": 6.5,
        "description": "3B model, quick testing"
    }
}

def train_model(
    model_key: str = "7b",
    data_dir: str = "mlx_data",
    adapter_path: str = "adapters/qwen_trained",
    learning_rate: float = 1e-5,
    iters: int = 500,
    batch_size: int = 1,
    use_autoresearch_config: bool = False,
):
    """Train Qwen model with 4-bit quantization."""
    
    # Get model config
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)
    
    model_config = MODELS[model_key]
    model_id = model_config["id"]
    memory_gb = model_config["memory_gb"]
    
    print("="*60)
    print(f"Training Qwen {model_key.upper()} 4-bit")
    print("="*60)
    print(f"Model: {model_id}")
    print(f"Memory required: ~{memory_gb} GB")
    print(f"Data: {data_dir}")
    print(f"Adapter: {adapter_path}")
    print("="*60)
    
    # Check for autoresearch config
    if use_autoresearch_config:
        config_file = Path("autoresearch_best.tsv")
        if config_file.exists():
            print(f"Reading config from {config_file}")
            with open(config_file) as f:
                lines = f.readlines()
                if len(lines) > 1:
                    parts = lines[1].strip().split('\t')
                    learning_rate = float(parts[2])
                    iters = int(parts[3])
                    batch_size = int(parts[4])
                    print(f"Using discovered config: lr={learning_rate}, iters={iters}, batch={batch_size}")
    
    # Check data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory {data_dir} not found. Run prepare_data.py first.")
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_id,
        "--data", data_dir,
        "--train",
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--iters", str(iters),
        "--adapter-path", adapter_path,
        "--grad-checkpoint",
        "--seed", "42",
    ]
    
    print(f"\nLearning rate: {learning_rate}")
    print(f"Iterations: {iters}")
    print(f"Batch size: {batch_size}")
    print("="*60 + "\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✅ Training complete!")
        print("="*60)
        print(f"Adapter saved to: {adapter_path}")
        print(f"Memory used: ~{memory_gb} GB")
    else:
        print("\n❌ Training failed")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen models")
    parser.add_argument("--model", "-m", default="7b", 
                        choices=list(MODELS.keys()),
                        help="Model to train (7b, 14b, 3b)")
    parser.add_argument("--data", default="mlx_data", help="Data directory")
    parser.add_argument("--adapter", default=None, help="Adapter output path")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--iters", type=int, default=500, help="Training iterations")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--use-autoresearch", action="store_true", 
                        help="Use discovered config")
    
    args = parser.parse_args()
    
    # Set default adapter path based on model
    if args.adapter is None:
        args.adapter = f"adapters/qwen_{args.model}_trained"
    
    train_model(
        model_key=args.model,
        data_dir=args.data,
        adapter_path=args.adapter,
        learning_rate=args.lr,
        iters=args.iters,
        batch_size=args.batch,
        use_autoresearch_config=args.use_autoresearch,
    )