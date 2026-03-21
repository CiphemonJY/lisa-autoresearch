#!/usr/bin/env python3
"""
Prepare training data for Qwen format.

Converts JSONL data to MLX format with Qwen chat template.

Usage:
    python prepare_data.py --input training_data.jsonl --output mlx_data/
"""

import argparse
import json
from pathlib import Path

def prepare_data(input_file: str, output_dir: str):
    """Convert training data to Qwen MLX format."""
    
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input file {input_file} not found")
        return False
    
    # Read input data
    print(f"Reading {input_file}...")
    lines = input_path.read_text().strip().split('\n')
    
    converted = []
    for line in lines:
        if not line.strip():
            continue
        
        try:
            item = json.loads(line)
            text = item.get("text", "")
            
            # Convert USER/ASSISTANT to Qwen format
            if "USER:" in text and "ASSISTANT:" in text:
                parts = text.split("ASSISTANT:")
                user_part = parts[0].replace("USER:", "").strip()
                assistant_part = parts[1].strip() if len(parts) > 1 else ""
                
                qwen_text = f"<|im_start|>user\n{user_part}<|im_end|>\n<|im_start|>assistant\n{assistant_part}<|im_end|>"
                converted.append({"text": qwen_text})
            else:
                # Already in correct format or plain text
                converted.append({"text": text})
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse line: {e}")
            continue
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write train.jsonl
    train_file = output_path / "train.jsonl"
    with open(train_file, 'w') as f:
        for item in converted:
            f.write(json.dumps(item) + '\n')
    
    # Write valid.jsonl (use first 3 for validation)
    valid_file = output_path / "valid.jsonl"
    with open(valid_file, 'w') as f:
        for item in converted[:3]:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Prepared {len(converted)} samples")
    print(f"   Train: {train_file}")
    print(f"   Valid: {valid_file}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for Qwen")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", default="mlx_data", help="Output directory")
    
    args = parser.parse_args()
    
    success = prepare_data(args.input, args.output)
    if not success:
        sys.exit(1)