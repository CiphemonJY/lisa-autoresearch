#!/usr/bin/env python3
"""
Test Qwen 2.5 3B LoRA Training (smaller, should fit in memory)
"""

import subprocess
import sys
import os
from pathlib import Path
import json

os.environ['PYTHONWARNINGS'] = 'ignore::urllib3.warnings.NotOpenSSLWarning'

TRAINING_DIR = Path.home() / ".lisa" / "training-data"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = TRAINING_DIR / "adapters" / "test_qwen3b"

def prepare_mlx_data():
    """Prepare data in MLX format for Qwen."""
    data_file = TRAINING_DIR / "training_data.jsonl"
    lines = data_file.read_text().strip().split('\n')
    
    mlx_dir = TRAINING_DIR / "mlx_data_qwen3b"
    mlx_dir.mkdir(exist_ok=True)
    
    converted = []
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            text = item.get("text", "")
            
            if "USER:" in text and "ASSISTANT:" in text:
                parts = text.split("ASSISTANT:")
                user_part = parts[0].replace("USER:", "").strip()
                assistant_part = parts[1].strip() if len(parts) > 1 else ""
                qwen_text = f"<|im_start|>user\n{user_part}<|im_end|>\n<|im_start|>assistant\n{assistant_part}<|im_end|>"
                converted.append({"text": qwen_text})
            else:
                converted.append({"text": text})
        except json.JSONDecodeError:
            continue
    
    train_file = mlx_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for item in converted:
            f.write(json.dumps(item) + '\n')
    
    valid_file = mlx_dir / "valid.jsonl"
    with open(valid_file, 'w') as f:
        for item in converted[:3]:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Prepared {len(converted)} samples for Qwen 3B")
    return str(mlx_dir)

def run_test_training():
    print("\n" + "="*60)
    print("Qwen 3B LoRA Training Test (smaller model)")
    print("="*60)
    
    print("\n[1/2] Preparing MLX data...")
    mlx_data_dir = prepare_mlx_data()
    
    print("\n[2/2] Running test training (50 iterations)...")
    ADAPTER_PATH.parent.mkdir(exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", MODEL_ID,
        "--data", mlx_data_dir,
        "--train",
        "--batch-size", "1",
        "--learning-rate", "1e-5",
        "--iters", "50",
        "--adapter-path", str(ADAPTER_PATH),
        "--seed", "42",
        "--grad-checkpoint"  # Memory optimization
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✅ Qwen 3B training test PASSED!")
        print("="*60)
        print(f"\nAdapter saved to: {ADAPTER_PATH}")
        return True
    else:
        print("\n❌ Qwen 3B training test FAILED")
        return False

if __name__ == "__main__":
    success = run_test_training()
    sys.exit(0 if success else 1)