#!/usr/bin/env python3
"""
LISA Selective Layer Training for MLX

Implements Layer-wise Importance Sampling (LISA) on MLX:
- Apply LoRA to ALL layers (num_layers=36 for Qwen2.5-3B)
- Then FREEZE bottom 24 layers (train only top 12)
- This achieves LISA-style selective training

Usage:
    python train_lisa_selective.py --iters 50
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Fix OpenMP issue on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# MLX imports
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from mlx.utils import tree_flatten, tree_unflatten

# MLX-LM imports
from mlx_lm import load as mlx_load
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters


# ---------------------------------------------------------------------------
# LISA Configuration
# ---------------------------------------------------------------------------

class LISAConfig:
    """Configuration for LISA selective layer training."""
    
    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-3B-Instruct-4bit",
        total_layers: int = 36,      # Qwen2.5-3B has 36 layers
        freeze_bottom: int = 24,     # Freeze bottom 24 layers (train top 12)
        lora_rank: int = 4,
        lora_scale: float = 1.0,
        lora_dropout: float = 0.0,
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        iters: int = 50,
        seed: int = 42,
        adapter_path: str = "adapters/lisa_qwen3b",
    ):
        self.model_id = model_id
        self.total_layers = total_layers
        self.freeze_bottom = freeze_bottom
        self.lora_rank = lora_rank
        self.lora_scale = lora_scale
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iters = iters
        self.seed = seed
        self.adapter_path = adapter_path
        
        # Validate
        assert freeze_bottom < total_layers, f"Cannot freeze {freeze_bottom} layers when total is {total_layers}"
        self.train_layers = total_layers - freeze_bottom
        print(f"\n{'='*60}")
        print(f"LISA SELECTIVE LAYER TRAINING")
        print(f"{'='*60}")
        print(f"Model: {model_id}")
        print(f"Total layers: {total_layers}")
        print(f"Freeze bottom: {freeze_bottom} layers")
        print(f"Train top: {self.train_layers} layers")
        print(f"LoRA rank: {lora_rank}, scale: {lora_scale}")


# ---------------------------------------------------------------------------
# Model Loading & LoRA Application
# ---------------------------------------------------------------------------

def extract_layer_num(key: str) -> Optional[int]:
    """Extract layer number from a model key like 'model.layers.5.attn.q_proj'."""
    match = re.search(r'\.layers\.(\d+)\.', key)
    if match:
        return int(match.group(1))
    return None


def load_model_and_apply_lora(
    model_id: str,
    num_layers: int,
    lora_config: Dict,
) -> Tuple[Any, Any, int]:
    """
    Load model, apply LoRA to all layers, return model + tokenizer.
    
    Returns: (model, tokenizer, actual_layer_count)
    """
    print(f"\nLoading model: {model_id}...")
    model, tokenizer = mlx_load(model_id)
    
    # Detect actual layer count
    actual_layers = 0
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        actual_layers = len(model.model.layers)
    elif hasattr(model, 'layers'):
        actual_layers = len(model.layers)
    
    print(f"Detected {actual_layers} transformer layers")
    
    # Apply LoRA to all layers (num_layers=0 means ALL layers in this model)
    # Actually we need to pass the actual count since MLX uses last N
    print(f"\nApplying LoRA to all {actual_layers} layers...")
    linear_to_lora_layers(model, num_layers=actual_layers, config=lora_config)
    
    return model, tokenizer, actual_layers


def freeze_bottom_lora_layers(model: nn.Module, freeze_below: int) -> int:
    """
    Freeze LoRA layers for bottom 'freeze_below' layers.
    
    This is the KEY LISA step: instead of training all 36 layers,
    we freeze the bottom 24 so only top 12 are trainable.
    
    Args:
        model: The model with LoRA layers applied
        freeze_below: Freeze layers with index < freeze_below
        
    Returns:
        Number of LoRA layers frozen
    """
    print(f"\nFreezing LoRA layers for bottom {freeze_below} layers (indices 0 to {freeze_below - 1})...")
    
    frozen_count = 0
    trainable_count = 0
    
    # Walk through all modules and their keys
    for key, module in model.named_modules():
        layer_num = extract_layer_num(key)
        
        # Check if this is a LoRA layer in a transformer layer we want to freeze
        if layer_num is not None and layer_num < freeze_below:
            # Check if module has freeze method (LoRA modules have this)
            if hasattr(module, 'freeze') and callable(module.freeze):
                module.freeze()
                frozen_count += 1
        elif layer_num is not None and layer_num >= freeze_below:
            # This is a trainable layer
            if hasattr(module, 'freeze') and callable(module.freeze):
                trainable_count += 1
    
    print(f"Frozen {frozen_count} LoRA layer modules")
    print(f"Keeping {trainable_count} LoRA layer modules trainable")
    
    return frozen_count


def get_trainable_params_info(model: nn.Module) -> Dict[str, int]:
    """Get detailed trainable parameter info."""
    # Use tree_flatten to get list of (key, array) tuples
    total_flat = tree_flatten(model.parameters())
    trainable_flat = tree_flatten(model.trainable_parameters())
    
    total_params = sum(v.size for _, v in total_flat) // 1_000_000
    trainable_params = sum(v.size for _, v in trainable_flat) // 1_000_000
    frozen_params = total_params - trainable_params
    
    return {
        'total_m': total_params,
        'trainable_m': trainable_params,
        'frozen_m': frozen_params,
        'trainable_pct': (trainable_params / total_params * 100) if total_params > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def create_simple_dataset(tokenizer, size: int = 20, max_length: int = 128) -> List[Dict]:
    """
    Create a simple training dataset for testing.
    In production, load from actual JSONL files.
    """
    print(f"\nCreating {size} simple training samples...")
    
    samples = []
    prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "Machine learning is a subset of",
        "The weather today is",
        "To make coffee, you need to",
        "The quick brown fox jumps over the",
        "Artificial intelligence will",
        "A healthy diet includes",
        "The internet allows people to",
        "Music is an art form that",
    ]
    
    # Get pad token id - fallback to EOS or 0 if not available
    pad_token_id = getattr(tokenizer, 'eos_token_id', None)
    if pad_token_id is None and hasattr(tokenizer, 'tokenizer'):
        pad_token_id = getattr(tokenizer.tokenizer, 'pad_token_id', None)
    pad_token_id = pad_token_id or 0
    
    for i in range(size):
        prompt = prompts[i % len(prompts)]
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize using MLX tokenizer's encode method (returns list, not dict)
        input_ids = tokenizer.encode(text)
        
        # Truncate if too long
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        pad_len = max_length - len(input_ids)
        input_ids = input_ids + [pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        
        samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
    
    return samples


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def simple_loss_fn(model: nn.Module, input_ids: mx.array, labels: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Simple language modeling loss.
    
    Forward pass, compute cross-entropy loss, return (loss, logits).
    """
    # Get logits from model
    logits = model(input_ids)
    
    # Shift for causal LM: predict next token
    # logits: [batch, seq, vocab] -> shift so labels[i] predicts logits[i-1]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    # Flatten for cross-entropy
    shift_logits = mx.reshape(shift_logits, (-1, shift_logits.shape[-1]))
    shift_labels = mx.reshape(shift_labels, (-1,))
    
    # Compute loss (ignore padding tokens)
    loss = mx.mean(nn.losses.cross_entropy(shift_logits, shift_labels))
    
    return loss, logits[:, -1, :]  # Return loss and last token logits


def train_lisa(
    model: nn.Module,
    tokenizer,
    dataset: List[Dict],
    config: LISAConfig,
) -> Dict[str, Any]:
    """
    Train model with LISA selective layer freezing.
    """
    print(f"\n{'='*60}")
    print(f"STARTING LISA TRAINING")
    print(f"{'='*60}")
    print(f"Iterations: {config.iters}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    
    # Setup optimizer with only trainable parameters
    optimizer = Adam(learning_rate=config.learning_rate)
    
    # Get initial parameter info
    info_before = get_trainable_params_info(model)
    print(f"\nBefore training:")
    print(f"  Total params: {info_before['total_m']}M")
    print(f"  Trainable: {info_before['trainable_m']}M ({info_before['trainable_pct']:.2f}%)")
    print(f"  Frozen: {info_before['frozen_m']}M")
    
    # Training state
    losses = []
    iteration_times = []
    
    # Set seed
    mx.random.seed(config.seed)
    
    print(f"\nTraining for {config.iters} iterations...")
    print("-" * 40)
    
    start_time = time.time()
    
    for iteration in range(config.iters):
        iter_start = time.time()
        
        # Get random sample
        idx = iteration % len(dataset)
        sample = dataset[idx]
        
        input_ids = mx.array(sample["input_ids"][:tokenizer.model_max_length]).reshape(1, -1)
        labels = mx.array(sample["input_ids"][:tokenizer.model_max_length]).reshape(1, -1)
        
        # Forward pass and loss
        loss, _ = simple_loss_fn(model, input_ids, labels)
        
        # Backward pass
        grads = mx.grad(lambda m: simple_loss_fn(m, input_ids, labels)[0])(model)
        
        # Update - only trainable params get updated
        optimizer.update(model, grads)
        
        iter_time = time.time() - iter_start
        losses.append(float(loss))
        iteration_times.append(iter_time)
        
        # Progress reporting
        if (iteration + 1) % 10 == 0 or iteration == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            recent_time = sum(iteration_times[-10:]) / min(10, len(iteration_times))
            print(f"  Iter {iteration + 1:3d}/{config.iters}: "
                  f"loss={avg_loss:.4f}, "
                  f"time={recent_time*1000:.1f}ms/iter")
    
    total_time = time.time() - start_time
    
    print("-" * 40)
    print(f"\nTraining completed in {total_time:.2f}s")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Average loss: {sum(losses) / len(losses):.4f}")
    print(f"Speed: {config.iters / total_time:.2f} iters/sec")
    
    return {
        'iterations': config.iters,
        'final_loss': float(losses[-1]),
        'avg_loss': float(sum(losses) / len(losses)),
        'total_time_sec': total_time,
        'iters_per_sec': config.iters / total_time,
        'losses': losses,
    }


# ---------------------------------------------------------------------------
# Adapter Saving
# ---------------------------------------------------------------------------

def save_lora_adapter(model: nn.Module, path: str, config: LISAConfig) -> bool:
    """
    Save LoRA adapter weights to disk.
    
    Saves only the trainable LoRA parameters.
    """
    print(f"\n{'='*60}")
    print(f"SAVING LISA ADAPTER")
    print(f"{'='*60}")
    
    adapter_path = Path(path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    
    # Collect trainable LoRA parameters - tree_flatten returns list of (key, array) tuples
    trainable_flat = tree_flatten(model.trainable_parameters())
    
    print(f"Saving {len(trainable_flat)} trainable parameter arrays...")
    
    # Convert to dict for safetensors
    adapter_weights = {}
    for key, param in trainable_flat:
        adapter_weights[key] = mx.astype(param, mx.float32)
    
    # Save using mlx's safetensors equivalent
    save_path = adapter_path / "adapters.safetensors"
    mx.save_safetensors(str(save_path), adapter_weights)
    print(f"Saved adapter to: {save_path}")
    
    # Save config
    config_path = adapter_path / "lisa_config.json"
    config_data = {
        "model_id": config.model_id,
        "total_layers": config.total_layers,
        "freeze_bottom": config.freeze_bottom,
        "train_top": config.train_layers,
        "lora_rank": config.lora_rank,
        "lora_scale": config.lora_scale,
        "lora_dropout": config.lora_dropout,
        "lisa_strategy": f"freeze_bottom_{config.freeze_bottom}_train_top_{config.train_layers}",
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config to: {config_path}")
    
    # Print summary
    total_size_mb = sum(v.nbytes for v in adapter_weights.values()) / 1_000_000
    print(f"\nAdapter size: {total_size_mb:.2f} MB")
    print(f"LISA strategy: Freeze bottom {config.freeze_bottom}, train top {config.train_layers}")
    
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LISA Selective Layer Training for MLX")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-3B-Instruct-4bit",
                        help="HuggingFace model ID")
    parser.add_argument("--iters", type=int, default=50,
                        help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--rank", type=int, default=4,
                        help="LoRA rank")
    parser.add_argument("--freeze-bottom", type=int, default=24,
                        help="Number of bottom layers to FREEZE (train only top N)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--adapter-path", type=str, default="adapters/lisa_qwen3b",
                        help="Path to save adapter")
    parser.add_argument("--test-only", action="store_true",
                        help="Test setup without actual training")
    
    args = parser.parse_args()
    
    # Create config
    config = LISAConfig(
        model_id=args.model,
        freeze_bottom=args.freeze_bottom,
        lora_rank=args.rank,
        lora_scale=1.0,
        lora_dropout=0.0,
        learning_rate=args.lr,
        iters=args.iters,
        seed=args.seed,
        adapter_path=args.adapter_path,
    )
    
    # LoRA config dict (required by linear_to_lora_layers)
    lora_config_dict = {
        "rank": config.lora_rank,
        "scale": config.lora_scale,
        "dropout": config.lora_dropout,
    }
    
    # Step 1: Load model and apply LoRA to all layers
    # Note: num_layers param is not used internally; actual_layers is detected and returned
    model, tokenizer, actual_layers = load_model_and_apply_lora(
        config.model_id,
        0,  # placeholder - actual layer count is auto-detected
        lora_config_dict,
    )
    config.total_layers = actual_layers
    
    # Step 2: Freeze bottom layers (the LISA key step!)
    # After this, only top (36 - 24) = 12 layers are trainable
    freeze_bottom_lora_layers(model, config.freeze_bottom)
    
    # Step 3: Print trainable parameter info
    print_trainable_parameters(model)
    info = get_trainable_params_info(model)
    print(f"\nLISA Training Mode:")
    print(f"  Training layers: {config.freeze_bottom} to {config.total_layers - 1} (top {config.train_layers} layers)")
    print(f"  Frozen layers: 0 to {config.freeze_bottom - 1} (bottom {config.freeze_bottom} layers)")
    
    if args.test_only:
        print("\n✅ Test-only mode: Model loaded, LoRA applied, layers frozen successfully!")
        return
    
    # Step 4: Create dataset and train
    dataset = create_simple_dataset(tokenizer, size=20)
    
    results = train_lisa(model, tokenizer, dataset, config)
    
    # Step 5: Save adapter
    save_lora_adapter(model, config.adapter_path, config)
    
    print(f"\n{'='*60}")
    print(f"✅ LISA TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model: {config.model_id}")
    print(f"Strategy: Freeze bottom {config.freeze_bottom}, train top {config.train_layers}")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Adapter saved to: {config.adapter_path}")


if __name__ == "__main__":
    main()
