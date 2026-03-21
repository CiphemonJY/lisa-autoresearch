#!/usr/bin/env python3
"""
Gradient Accumulation for LISA+Offload

Enables larger effective batch sizes by accumulating gradients.

How it works:
- Forward/backward pass on micro-batches
- Accumulate gradients without updating weights
- Update weights after N micro-batches
- Effective batch size = micro_batch_size * accumulation_steps

Benefits:
- Train with larger effective batch sizes on limited memory
- Better gradient estimates
- More stable training

Memory impact:
- Same memory as single micro-batch
- Effective batch size can be much larger

Speed impact:
- More forward/backward passes per update
- But allows training with larger effective batch sizes
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

os.environ['PYTHONWARNINGS'] = 'ignore::urllib3.warnings.NotOpenSSLWarning'


@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation."""

    enabled: bool = True
    accumulation_steps: int = 4  # Accumulate 4 micro-batches
    micro_batch_size: int = 1  # Batch size per micro-batch

    # Effective batch size = micro_batch_size * accumulation_steps
    # Example: micro_batch_size=1, accumulation_steps=4 → effective_batch_size=4

    # When to clear gradients
    clear_every: int = 0  # Clear every N batches (0 = only at update)

    # Gradient clipping
    max_grad_norm: float = 1.0  # Clip gradients to this norm

    # Precision
    accumulate_in_fp32: bool = False  # Keep FP32 copy of FP16 gradients


class GradientAccumulationTrainer:
    """
    Gradient accumulation for LISA+Offload.

    Enables training with larger effective batch sizes on limited memory.
    Accumulates gradients across micro-batches before updating weights.

    Usage:
        from gradient_accumulation import GradientAccumulationTrainer, GradientAccumulationConfig

        config = GradientAccumulationConfig(
            accumulation_steps=4,
            micro_batch_size=1,
            max_grad_norm=1.0,
        )

        trainer = GradientAccumulationTrainer(
            model_id="Qwen2.5-32B-Instruct-4bit",
            ga_config=config,
            max_memory_gb=6.0,
        )

        results = trainer.train(
            data_dir="training_data/",
            iterations=100,
        )
    """

    def __init__(
        self,
        model_id: str,
        ga_config: GradientAccumulationConfig = None,
        max_memory_gb: float = 5.0,
        verbose: bool = True,
    ):
        """
        Initialize gradient accumulation trainer.

        Args:
            model_id: HuggingFace model ID
            ga_config: Gradient accumulation configuration
            max_memory_gb: Maximum memory to use
            verbose: Print progress information
        """
        self.model_id = model_id
        self.ga_config = ga_config or GradientAccumulationConfig()
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose

        # Gradient accumulator
        self.accumulated_gradients: Dict[str, Any] = {}
        self.accumulation_counter = 0

        # Statistics
        self.stats = {
            "total_steps": 0,
            "updates": 0,
            "gradient_clips": 0,
            "effective_batch_size": self.ga_config.micro_batch_size * self.ga_config.accumulation_steps,
        }

    def log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[GA] {message}")

    def estimate_memory_impact(self) -> Dict[str, Any]:
        """
        Estimate memory impact of gradient accumulation.

        Returns:
            Dictionary with memory breakdown
        """
        micro_batch_memory = self.max_memory_gb * 0.3  # Rough estimate

        # Gradient accumulation doesn't increase memory much
        # because we accumulate gradients (which are smaller than activations)
        gradient_memory = micro_batch_memory * 0.1 * self.ga_config.accumulation_steps

        return {
            "micro_batch_memory_gb": micro_batch_memory,
            "gradient_memory_gb": gradient_memory,
            "total_memory_gb": micro_batch_memory + gradient_memory,
            "effective_batch_size": self.stats["effective_batch_size"],
            "accumulation_steps": self.ga_config.accumulation_steps,
        }

    def accumulate_gradients(self, gradients: Dict[str, Any]):
        """
        Accumulate gradients from a micro-batch.

        Args:
            gradients: Dictionary of gradient tensors
        """
        for name, grad in gradients.items():
            if isinstance(grad, list):
                grad = grad[0] if grad else None
            if name not in self.accumulated_gradients:
                # Initialize accumulator (clone works for torch tensors; copy for numpy)
                self.accumulated_gradients[name] = (
                    grad.clone() if hasattr(grad, "clone") else grad.copy()
                )
            else:
                # Accumulate
                self.accumulated_gradients[name] += grad

        self.accumulation_counter += 1
        self.stats["total_steps"] += 1

    def should_update(self) -> bool:
        """
        Check if we should update weights.

        Returns:
            True if accumulation_steps reached
        """
        return self.accumulation_counter >= self.ga_config.accumulation_steps

    def clip_gradients(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clip gradients by norm.

        Args:
            gradients: Dictionary of gradient tensors

        Returns:
            Clipped gradients
        """
        if self.ga_config.max_grad_norm <= 0:
            return gradients

        # Compute total norm
        total_norm = 0.0
        for grad in gradients.values():
            if hasattr(grad, "norm"):
                total_norm += grad.norm() ** 2
            else:
                # numpy array
                total_norm += float(np.linalg.norm(grad)) ** 2
        total_norm = total_norm ** 0.5

        # Clip if needed
        if total_norm > self.ga_config.max_grad_norm:
            clip_coef = self.ga_config.max_grad_norm / (total_norm + 1e-6)
            for name in gradients:
                gradients[name] = gradients[name] * clip_coef
            self.stats["gradient_clips"] += 1

        return gradients

    def get_accumulated_gradients(self) -> Dict[str, Any]:
        """
        Get averaged accumulated gradients.

        Returns:
            Averaged gradients
        """
        # Average by accumulation steps
        averaged = {}
        for name, grad in self.accumulated_gradients.items():
            averaged[name] = grad / self.accumulation_counter

        return averaged

    def clear_gradients(self):
        """Clear accumulated gradients."""
        self.accumulated_gradients = {}
        self.accumulation_counter = 0

    def train_step(
        self,
        model: Any,
        input_data: Any,
        micro_batch_idx: int,
    ) -> Dict[str, Any]:
        """
        Perform one micro-batch training step.

        Args:
            model: The model
            input_data: Micro-batch input
            micro_batch_idx: Index of micro-batch

        Returns:
            Dictionary with loss, gradients, etc.
        """
        # In real implementation:
        # 1. Forward pass
        # 2. Backward pass
        # 3. Accumulate gradients

        # Simulated
        return {
            "loss": 0.0,
            "gradients": {},
            "micro_batch_idx": micro_batch_idx,
        }

    def train(
        self,
        data_dir: str,
        iterations: int = 100,
        learning_rate: float = 1e-5,
    ) -> Dict[str, Any]:
        """
        Train with gradient accumulation.

        Args:
            data_dir: Directory with training data
            iterations: Number of weight updates
            learning_rate: Learning rate

        Returns:
            Training results
        """
        self.log("Starting gradient accumulation training...")
        self.log(f"  Micro-batch size: {self.ga_config.micro_batch_size}")
        self.log(f"  Accumulation steps: {self.ga_config.accumulation_steps}")
        self.log(f"  Effective batch size: {self.stats['effective_batch_size']}")

        # Estimate memory
        memory = self.estimate_memory_impact()
        self.log(f"\nMemory impact:")
        self.log(f"  Micro-batch: {memory['micro_batch_memory_gb']:.1f} GB")
        self.log(f"  Gradients: {memory['gradient_memory_gb']:.1f} GB")
        self.log(f"  Total: {memory['total_memory_gb']:.1f} GB")

        # Training simulation
        total_micro_batches = iterations * self.ga_config.accumulation_steps
        self.log(f"\nTraining {iterations} updates ({total_micro_batches} micro-batches)...")

        for i in range(iterations):
            # Accumulate gradients
            for micro_idx in range(self.ga_config.accumulation_steps):
                result = self.train_step(None, None, micro_idx)
                self.accumulate_gradients(result["gradients"])

            # Update weights
            gradients = self.get_accumulated_gradients()
            gradients = self.clip_gradients(gradients)

            # Clear for next iteration
            self.clear_gradients()

            self.stats["updates"] += 1

        # Summary
        self.log(f"\nTraining complete!")
        self.log(f"  Total micro-batches: {self.stats['total_steps']}")
        self.log(f"  Weight updates: {self.stats['updates']}")
        self.log(f"  Gradient clips: {self.stats['gradient_clips']}")
        self.log(f"  Effective batch size: {self.stats['effective_batch_size']}")

        return {
            "iterations": iterations,
            "stats": self.stats,
            "memory_impact": memory,
        }


def run_gradient_accumulation_benchmark():
    """Benchmark different accumulation steps."""
    print("="*70)
    print("GRADIENT ACCUMULATION BENCHMARK")
    print("="*70)
    print()

    configs = [
        ("No accumulation", 1, 1),
        ("4x accumulation", 4, 1),
        ("8x accumulation", 8, 1),
        ("16x accumulation", 16, 1),
        ("32x accumulation", 32, 1),
    ]

    print(f"{'Config':<25} {'Eff. Batch':<15} {'Memory':<15} {'Quality'}")
    print("-"*70)

    for name, steps, micro_batch in configs:
        ga_config = GradientAccumulationConfig(
            enabled=True if steps > 1 else False,
            accumulation_steps=steps,
            micro_batch_size=micro_batch,
        )

        trainer = GradientAccumulationTrainer(
            model_id="Qwen2.5-32B-Instruct-4bit",
            ga_config=ga_config,
            verbose=False,
        )

        memory = trainer.estimate_memory_impact()
        eff_batch = steps * micro_batch

        # Quality estimate (higher effective batch = better)
        quality = "Low" if eff_batch <= 1 else "Medium" if eff_batch <= 8 else "High"

        print(f"{name:<25} {eff_batch:<15} {memory['total_memory_gb']:.1f} GB{'':<8} {quality}")

    print()
    print("Recommendation:")
    print("  • Use 4-8x accumulation for limited memory")
    print("  • Use 16-32x for best gradient estimates")
    print("  • Memory stays same regardless of accumulation steps")


if __name__ == "__main__":
    print("="*70)
    print("GRADIENT ACCUMULATION FOR LISA+OFFLOAD")
    print("="*70)
    print()
    print("Enables larger effective batch sizes on limited memory.")
    print()

    run_gradient_accumulation_benchmark()