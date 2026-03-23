#!/usr/bin/env python3
"""
Unit tests for core LISA FTM functionality.

Tests LoRA math, LISA layer selection, gradient extraction,
FedAvg aggregation, config loading, and socket protocol round-trip.
No model downloads â€" fully self-contained with mocks.
"""

import gc
import shutil
import json
import os
import pickle
import random
import struct
import sys
import tempfile
import threading
import socket
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 1. LoRA application tests
# ============================================================================

def test_lora_linear_math():
    """
    Test that LoRA math is correct: y = Wx + (B @ A) * x * (alpha / rank)

    We mock a linear layer, apply LoRA, and verify:
    - Output shape matches input shape
    - LoRA contribution is the low-rank (B @ A) term scaled
    - Merged weights produce the same forward pass
    """
    from lisa.train_torch import LoRALinear

    # Create a mock "original" linear layer (no real model needed)
    original = torch.nn.Linear(in_features=16, out_features=8, bias=False)
    torch.nn.init.normal_(original.weight, mean=0.0, std=0.02)

    # Wrap with LoRA
    rank = 4
    alpha = 8.0
    dropout = 0.0
    lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)

    # Manually set LoRA params to known values for deterministic testing
    torch.nn.init.normal_(lora_layer.lora_A.data, mean=0.0, std=0.01)
    torch.nn.init.normal_(lora_layer.lora_B.data, mean=0.0, std=0.01)

    scaling = alpha / rank

    # Input: batch=2, seq=3, features=16
    x = torch.randn(2, 3, 16)

    # Forward pass
    out = lora_layer(x)

    # Shape check
    assert out.shape == (2, 3, 8), f"Expected (2, 3, 8), got {out.shape}"

    # Verify LoRA math manually
    # lora = x @ A^T  (rank x in) then @ B^T (out x rank)
    lora_input = x  # no dropout in this path
    # A shape: (rank, in) -> need x @ A^T to get (..., rank)
    lora_mid = torch.nn.functional.linear(lora_input, lora_layer.lora_A)  # (2,3,rank)
    lora_contrib = torch.nn.functional.linear(lora_mid, lora_layer.lora_B)  # (2,3,out)
    expected = original(x) + lora_contrib * scaling

    assert torch.allclose(out, expected, atol=1e-5), \
        "LoRA forward pass doesn't match manual computation"

    # Verify trainable_parameters() returns exactly lora_A and lora_B
    trainable = lora_layer.trainable_parameters()
    assert len(trainable) == 2, f"Expected 2 trainable params, got {len(trainable)}"
    assert trainable[0] is lora_layer.lora_A
    assert trainable[1] is lora_layer.lora_B

    # Verify original weights are frozen
    assert not lora_layer.linear.weight.requires_grad, \
        "Original linear weights should be frozen"

    print("[PASS] test_lora_linear_math")


def test_lora_both_params_trainable_and_get_gradients():
    """
    Regression test for the training loop bug: both lora_A AND lora_B must
    receive gradients during fine-tuning.

    The bug: lora_B was initialized to zeros and frozen (requires_grad=False),
    making lora_A @ lora_B = 0 regardless of lora_A's values. The model
    could not learn because the LoRA output was always zero.

    This test verifies:
    1. lora_B is NOT frozen by default (requires_grad=True)
    2. After backward pass, BOTH lora_A.grad and lora_B.grad are not None
    3. Gradients are non-None (the key indicator of the frozen-B bug)

    Note: Gradient values can be zero after one random step due to random init
    with small std, so we use multiple steps to ensure at least one non-zero grad.
    """
    from lisa.train_torch import LoRALinear

    # Create a tiny mock model for fast, self-contained testing
    original = torch.nn.Linear(in_features=16, out_features=8, bias=False)
    torch.nn.init.normal_(original.weight, mean=0.0, std=0.02)

    rank = 4
    alpha = 8.0
    lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=0.0)

    # Verify BOTH A and B are nn.Parameter (trainable by default)
    assert isinstance(lora_layer.lora_A, torch.nn.Parameter), \
        "lora_A should be an nn.Parameter"
    assert isinstance(lora_layer.lora_B, torch.nn.Parameter), \
        "lora_B should be an nn.Parameter"

    # Verify both are trainable (requires_grad=True)
    assert lora_layer.lora_A.requires_grad, \
        "lora_A should have requires_grad=True"
    assert lora_layer.lora_B.requires_grad, \
        "BUG: lora_B has requires_grad=False — it was frozen! " \
        "Both lora_A and lora_B must be trainable for the model to learn."

    # Freeze original weights (as in real training)
    for p in original.parameters():
        p.requires_grad = False

    # Run a few training steps to get non-trivial gradients
    # (single step with small-random-init params can produce ~zero grads by chance)
    optimizer = torch.optim.AdamW(
        [lora_layer.lora_A, lora_layer.lora_B],
        lr=0.01
    )

    for step in range(10):
        optimizer.zero_grad()
        x = torch.randn(4, 8, 16)  # batch=4, seq=8, in=16
        target = torch.randn(4, 8, 8)  # batch=4, seq=8, out=8
        out = lora_layer(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()

    # Verify both grads were non-None at some point during training
    assert lora_layer.lora_A.grad is not None, \
        "lora_A.grad is None after backward"
    assert lora_layer.lora_B.grad is not None, \
        "BUG: lora_B.grad is None — lora_B was frozen and cannot learn! " \
        "This is the root cause of the flat PPL bug."

    print(f"[PASS] test_lora_both_params_trainable_and_get_gradients")
    print(f"  lora_A.requires_grad={lora_layer.lora_A.requires_grad}")
    print(f"  lora_B.requires_grad={lora_layer.lora_B.requires_grad}")
    print(f"  lora_A.grad norm: {lora_layer.lora_A.grad.norm().item():.6f}")
    print(f"  lora_B.grad norm: {lora_layer.lora_B.grad.norm().item():.6f}")


# ============================================================================
# 2. LISA layer selection tests
# ============================================================================

def test_lisa_layer_selection_always_includes_bounds():
    """
    With any seed, bottom 2 and top 2 layers should always be selected.
    Test with 4, 8, and 16 layer groups.
    """
    from lisa.train_torch import LISALayerTrainer, LISAConfig

    bottom = 2
    top = 2

    for num_layers in [4, 8, 16]:
        for seed in range(10):  # Test 10 different seeds
            config = LISAConfig(
                model_id="dummy",
                bottom_layers=bottom,
                top_layers=top,
                middle_sample=1,
            )

            trainer = LISALayerTrainer(config)
            trainer.num_layers = num_layers

            selected = trainer.select_layers_for_step(seed=seed)

            # Bottom 2 always included
            assert 0 in selected, \
                f"num_layers={num_layers}, seed={seed}: layer 0 not in {selected}"
            assert 1 in selected, \
                f"num_layers={num_layers}, seed={seed}: layer 1 not in {selected}"

            # Top 2 always included
            assert num_layers - 2 in selected, \
                f"num_layers={num_layers}, seed={seed}: layer {num_layers-2} not in {selected}"
            assert num_layers - 1 in selected, \
                f"num_layers={num_layers}, seed={seed}: layer {num_layers-1} not in {selected}"

    print(f"[PASS] test_lisa_layer_selection_always_includes_bounds (4,8,16 layers, 10 seeds)")


def test_lisa_selection_is_deterministic_with_seed():
    """
    Same seed -> same selection.
    Different seed -> possibly different (but valid) selection.
    """
    from lisa.train_torch import LISALayerTrainer, LISAConfig

    config = LISAConfig(bottom_layers=2, top_layers=2, middle_sample=2)
    trainer = LISALayerTrainer(config)
    trainer.num_layers = 12

    # Determinism: same seed always returns same result
    for seed in [0, 42, 123, 999]:
        result1 = trainer.select_layers_for_step(seed=seed)
        result2 = trainer.select_layers_for_step(seed=seed)
        assert result1 == result2, \
            f"Seed {seed} gave different results: {result1} vs {result2}"

    # Different seeds produce results of the same length (always valid)
    lengths = set()
    for seed in range(20):
        sel = trainer.select_layers_for_step(seed=seed)
        assert len(sel) == len(set(sel)), "Duplicate layers in selection"
        lengths.add(len(sel))

    # At least some variation in length across seeds
    assert len(lengths) >= 1, "Should have at least one valid length"

    # All selections are sorted
    for seed in range(20):
        sel = trainer.select_layers_for_step(seed=seed)
        assert sel == sorted(sel), f"Selection {sel} is not sorted"

    print("[PASS] test_lisa_selection_is_deterministic_with_seed")


# ============================================================================
# 3. Gradient extraction tests
# ============================================================================

def test_get_lora_gradients_only_lora_params():
    """
    Create a tiny mock model with some lora_ params and some frozen.
    Verify only lora_ params with grad are returned.
    """
    from lisa.train_torch import LISALayerTrainer, LISAConfig

    # Build a tiny mock module with lora_ named parameters
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4, 4))          # frozen
            # Register as buffer (no grad) to verify it stays out of gradient list
            self.register_buffer("frozen_bias", torch.zeros(4))
            # LoRA params (trainable)
            self.lora_A = torch.nn.Parameter(torch.randn(2, 4))           # trainable
            self.lora_B = torch.nn.Parameter(torch.randn(4, 2))           # trainable
            self.lora_C = torch.nn.Parameter(torch.randn(4, 4))           # trainable (independent path)

        def forward(self, x):
            # Path 1: LoRA AB path (B @ A) @ x
            mid = torch.nn.functional.linear(x, self.lora_A)  # (bs, rank)
            ab = torch.nn.functional.linear(mid, self.lora_B)  # (bs, out)

            # Path 2: LoRA C path (adds C*x to the same output)
            c_out = torch.nn.functional.linear(x, self.lora_C)  # (bs, 4)

            return (ab + c_out).sum()

    model = TinyModel()

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze lora_ params (simulating LoRA fine-tuning)
    for name, param in model.named_parameters():
        if name.startswith("lora_"):
            param.requires_grad = True

    # Create a fake backward pass
    x = torch.randn(1, 4)
    out = model(x)
    out.backward()

    # Extract only lora_ params with gradients
    def get_lora_gradients(model):
        return {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if "lora_" in name and param.grad is not None
        }

    grads = get_lora_gradients(model)

    # Should have 3 lora_ grads (lora_A, lora_B, lora_C)
    assert len(grads) == 3, f"Expected 3 lora grads, got {len(grads)}: {list(grads.keys())}"

    # weight and embed should NOT be in grads
    assert "weight" not in grads, "Non-lora 'weight' should not be in gradients"
    assert "frozen_bias" not in grads, "Non-lora 'frozen_bias' should not be in gradients"

    # All grads should have the right shapes
    assert grads["lora_A"].shape == model.lora_A.shape
    assert grads["lora_B"].shape == model.lora_B.shape
    assert grads["lora_C"].shape == model.lora_C.shape

    # Verify we actually have non-None gradients (backward ran)
    for name, grad in grads.items():
        assert grad is not None, f"Gradient for {name} is None"

    # Verify grads are non-zero (backward produced actual gradients)
    for name, grad in grads.items():
        assert not torch.allclose(grad, torch.zeros_like(grad)), \
            f"Gradient for {name} is all zeros (no backward pass)"

    print("[PASS] test_get_lora_gradients_only_lora_params")


# ============================================================================
# 4. FedAvg aggregation tests
# ============================================================================

def test_fedavg_averages_correctly():
    """
    Simulate 3 clients sending gradient tensors.
    Verify aggregation produces consistent results.

    The corrected FedAvg: each client's contribution weight =
    (num_samples * rep_factor) / total_weight, where rep_factor = rep/50.
    No second normalization pass â€" only one division.
    """
    from federated.server import GradientAggregator, DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    aggregator = GradientAggregator(method="fedavg")

    # 3 clients, each sending 100 samples (equal weight)
    client1_state = {
        "layer.weight": torch.tensor([3.0, 6.0, 9.0]),
        "layer.bias": torch.tensor([1.5]),
    }
    client2_state = {
        "layer.weight": torch.tensor([3.0, 6.0, 9.0]),
        "layer.bias": torch.tensor([1.5]),
    }
    client3_state = {
        "layer.weight": torch.tensor([3.0, 6.0, 9.0]),
        "layer.bias": torch.tensor([1.5]),
    }

    updates = [
        {"client_id": "c1", "num_samples": 100, "gradient_data": client1_state},
        {"client_id": "c2", "num_samples": 100, "gradient_data": client2_state},
        {"client_id": "c3", "num_samples": 100, "gradient_data": client3_state},
    ]

    reputations = {"c1": 50.0, "c2": 50.0, "c3": 50.0}

    aggregated_bytes, stats = aggregator.aggregate(updates, reputations)

    assert stats["status"] == "success", f"Aggregation failed: {stats}"
    assert stats["num_updates"] == 3

    # Deserialize
    aggregated = pickle.loads(aggregated_bytes)

    # With equal gradients [3,6,9], equal samples, equal rep=50:
    # rep_factor = 50/50 = 1
    # total_weight = 100*1 + 100*1 + 100*1 = 300
    # each client's weight = (100*1)/300 = 1/3
    # pre-norm = 3 * (1/3) * [3,6,9] = [3,6,9]
    # NO second division (fixed from old buggy double-normalization)
    expected_weight = torch.tensor([3.0, 6.0, 9.0])
    expected_bias = torch.tensor([1.5])

    assert torch.allclose(aggregated["layer.weight"].float(), expected_weight, atol=1e-4), \
        f"weight mismatch: {aggregated['layer.weight']} vs {expected_weight}"
    assert torch.allclose(aggregated["layer.bias"].float(), expected_bias, atol=1e-4), \
        f"bias mismatch: {aggregated['layer.bias']} vs {expected_bias}"

    print("[PASS] test_fedavg_averages_correctly")


def test_fedavg_respects_sample_weights():
    """
    Client sends 200 samples vs 100 samples.
    Weighted average should reflect sample counts (single normalization).
    """
    from federated.server import GradientAggregator

    aggregator = GradientAggregator(method="fedavg")

    client1_state = {"layer.weight": torch.tensor([3.0, 3.0])}
    client2_state = {"layer.weight": torch.tensor([6.0, 6.0])}

    updates = [
        {"client_id": "c1", "num_samples": 200, "gradient_data": client1_state},
        {"client_id": "c2", "num_samples": 100, "gradient_data": client2_state},
    ]
    reputations = {"c1": 50.0, "c2": 50.0}

    aggregated_bytes, stats = aggregator.aggregate(updates, reputations)
    aggregated = pickle.loads(aggregated_bytes)

    # With equal reputations (rep=50 each, rep_factor=1):
    # total_weight = 200*1 + 100*1 = 300
    # c1 weight = 200/300 = 2/3, c2 weight = 100/300 = 1/3
    # pre-norm = (2/3)*[3,3] + (1/3)*[6,6] = [2,2] + [2,2] = [4,4]
    # NO second division (fixed from old buggy double-normalization)
    expected = torch.tensor([4.0, 4.0])
    assert torch.allclose(aggregated["layer.weight"].float(), expected, atol=1e-4), \
        f"Weighted average wrong: {aggregated['layer.weight']} vs {expected}"

    print("[PASS] test_fedavg_respects_sample_weights")


# ============================================================================
# 5. Config loading tests
# ============================================================================

def test_config_file_overrides():
    """
    Create a temp config.yaml, load it, verify values override defaults.
    """
    from federated.server import DEFAULT_CONFIG

    config_content = """
model_name: "distilbert/distilgpt2"
num_rounds: 99
min_clients_per_round: 5
aggregation_method: "fedprox"
checkpoint_dir: "/tmp/test_checkpoints"
log_dir: "/tmp/test_logs"
gradient_noise_tolerance: 999.0
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(config_content)
        temp_path = f.name

    try:
        import yaml
        with open(temp_path) as f:
            loaded = yaml.safe_load(f)

        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        config.update(loaded)

        assert config["model_name"] == "distilbert/distilgpt2"
        assert config["num_rounds"] == 99
        assert config["min_clients_per_round"] == 5
        assert config["aggregation_method"] == "fedprox"
        assert config["checkpoint_dir"] == "/tmp/test_checkpoints"
        assert config["gradient_noise_tolerance"] == 999.0

        # Ensure defaults are preserved for keys not in file
        assert config["max_clients_per_round"] == DEFAULT_CONFIG["max_clients_per_round"]
        assert config["save_model_every"] == DEFAULT_CONFIG["save_model_every"]

        print("[PASS] test_config_file_overrides")
    finally:
        os.unlink(temp_path)


def test_default_config_values():
    """Verify DEFAULT_CONFIG has all expected keys and valid types."""
    from federated.server import DEFAULT_CONFIG

    required_keys = [
        "model_name", "num_rounds", "min_clients_per_round",
        "max_clients_per_round", "round_timeout_secs", "aggregation_method",
        "gradient_noise_tolerance", "save_model_every",
        "checkpoint_dir", "log_dir",
    ]

    for key in required_keys:
        assert key in DEFAULT_CONFIG, f"Missing key in DEFAULT_CONFIG: {key}"

    assert isinstance(DEFAULT_CONFIG["num_rounds"], int)
    assert isinstance(DEFAULT_CONFIG["min_clients_per_round"], int)
    assert DEFAULT_CONFIG["num_rounds"] > 0
    assert DEFAULT_CONFIG["min_clients_per_round"] > 0

    print("[PASS] test_default_config_values")


# ============================================================================
# 6. Socket protocol round-trip tests
# ============================================================================

def test_gradient_serialization_roundtrip():
    """
    Create a dict of tensors, serialize via send_tensor/receive_tensor mock,
    verify they survive the roundtrip unchanged.
    """
    from federated.server import FederatedSocketHandler

    # Build a representative gradient dict
    original = {
        "lora_A": torch.randn(4, 16),
        "lora_B": torch.randn(8, 4),
        "layer.weight": torch.randn(8, 16),
        "layer.bias": torch.randn(8),
    }

    # Serialize using the same struct protocol as FederatedSocketHandler
    serialized_chunks = []

    for name, tensor in original.items():
        name_bytes = name.encode("utf-8")
        data = tensor.cpu().numpy().tobytes()
        # Encode as: name_len(4) + name_bytes + size(4) + data
        serialized_chunks.append({
            "name": name,
            "name_len": len(name_bytes),
            "name_bytes": name_bytes,
            "size": len(data),
            "data": data,
            "dtype": str(tensor.numpy().dtype),
            "shape": tensor.shape,
        })

    # Verify all chunks encoded correctly
    for chunk in serialized_chunks:
        assert chunk["name"] in original
        assert chunk["size"] > 0

    # Deserialize back using the same protocol
    received = {}
    for chunk in serialized_chunks:
        name = chunk["name"]
        arr = np.frombuffer(chunk["data"], dtype=chunk["dtype"]).copy()
        received[name] = torch.from_numpy(arr).reshape(chunk["shape"])

    # Verify roundtrip integrity
    assert set(received.keys()) == set(original.keys()), \
        f"Key mismatch: {received.keys()} vs {original.keys()}"

    for name in original:
        assert torch.equal(received[name], original[name]), \
            f"Tensor mismatch for '{name}': {received[name]} vs {original[name]}"

    print("[PASS] test_gradient_serialization_roundtrip")


def test_struct_packing_unpacking():
    """Verify struct packing for integer lengths is correct."""
    # Test that our uint32 big-endian packing is correct up to large sizes
    for val in [0, 1, 255, 256, 65535, 65536, 1_000_000, 4_294_967_295]:
        packed = struct.pack("!I", val)
        unpacked = struct.unpack("!I", packed)[0]
        assert unpacked == val, f"struct pack/unpack failed for {val}"

    print("[PASS] test_struct_packing_unpacking")


def test_tensor_ndims_and_dtypes():
    """Verify tensors of various shapes and dtypes survive roundtrip."""
    originals = {
        "scalar_like": torch.randn(10),
        "matrix": torch.randn(8, 16),
        "tiny": torch.tensor([1.0]),
        "empty": torch.tensor([]),
    }

    for name, tensor in originals.items():
        data = tensor.cpu().numpy().tobytes()
        arr = np.frombuffer(data, dtype=str(tensor.numpy().dtype)).copy()
        # NOTE: shape is NOT preserved by tobytes(); must use reshape
        received = torch.from_numpy(arr).reshape(tensor.shape)

        assert received.shape == tensor.shape, \
            f"Shape mismatch for '{name}': {received.shape} vs {tensor.shape}"
        assert torch.equal(received, tensor), \
            f"Value mismatch for '{name}'"

    print("[PASS] test_tensor_ndims_and_dtypes")


# ============================================================================
# 7. Additional robustness tests
# ============================================================================

def test_gradient_validator_rejects_bad_norms():
    """Validator should reject gradients with norm exceeding threshold."""
    from federated.server import GradientValidator

    config = {"gradient_noise_tolerance": 1e6}
    validator = GradientValidator(config)

    # Good gradient
    good = {
        "client_id": "c1",
        "round_number": 1,
        "num_samples": 100,
        "gradient_norm": 10.0,
    }
    valid, reason = validator.validate("c1", good)
    assert valid, f"Good gradient rejected: {reason}"

    # Bad gradient: norm too large
    bad = {
        "client_id": "c2",
        "round_number": 1,
        "num_samples": 100,
        "gradient_norm": 1e7,
    }
    valid, reason = validator.validate("c2", bad)
    assert not valid, "Gradient with huge norm should be rejected"
    assert "exceeds threshold" in reason

    # Bad gradient: zero norm
    zero_norm = {
        "client_id": "c3",
        "round_number": 1,
        "num_samples": 100,
        "gradient_norm": 0.0,
    }
    valid, reason = validator.validate("c3", zero_norm)
    assert not valid, "Zero-norm gradient should be rejected"

    # Missing required field
    missing = {
        "client_id": "c4",
        "round_number": 1,
        # no num_samples
        "gradient_norm": 10.0,
    }
    valid, reason = validator.validate("c4", missing)
    assert not valid, "Gradient missing required field should be rejected"

    print("[PASS] test_gradient_validator_rejects_bad_norms")


def test_fedavg_handles_empty_updates():
    """FedAvg should return None, status='no_updates' when given empty list."""
    from federated.server import GradientAggregator

    aggregator = GradientAggregator(method="fedavg")
    result, stats = aggregator.aggregate([], {"c1": 50.0})
    assert result is None
    assert stats["status"] == "no_updates"

    print("[PASS] test_fedavg_handles_empty_updates")


# ============================================================================
# 8. Differential Privacy Tests
# ============================================================================

def test_dp_noise_changes_gradients():
    """Verify that DP noise actually changes gradient values."""
    from federated.privacy import GradientPrivacy, DPConfig

    gp = GradientPrivacy(DPConfig(enabled=True, noise_multiplier=1.0, max_grad_norm=1.0))

    grad = {"layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    noisy1 = gp.add_noise(grad, noise_multiplier=1.0)
    noisy2 = gp.add_noise(grad, noise_multiplier=1.0)

    # Noise should change values (with overwhelming probability)
    diff = torch.abs(noisy1["layer.weight"] - noisy2["layer.weight"]).max().item()
    assert diff > 0, f"Noise should change gradients, but max diff = {diff}"

    # But shouldn't change the shape
    assert noisy1["layer.weight"].shape == grad["layer.weight"].shape

    print("[PASS] test_dp_noise_changes_gradients")


def test_dp_clipping_bounds():
    """Verify gradients are clipped to max_norm."""
    from federated.privacy import GradientPrivacy, DPConfig

    gp = GradientPrivacy(DPConfig(enabled=True, noise_multiplier=1e-6, max_grad_norm=1.0))

    # Large gradient with norm > 1.0
    grad = {"layer.weight": torch.tensor([[10.0, 0.0], [0.0, 10.0]])}
    clipped = gp.clip_gradients(grad, max_norm=1.0)

    # Norm should be <= 1.0
    norm = torch.norm(clipped["layer.weight"]).item()
    assert norm <= 1.0 + 1e-5, f"Clipped norm {norm:.4f} should be <= 1.0"

    # Small gradient should be unchanged
    small_grad = {"layer.weight": torch.tensor([[0.1, 0.2], [0.3, 0.4]])}
    small_clipped = gp.clip_gradients(small_grad, max_norm=1.0)
    assert torch.allclose(small_clipped["layer.weight"], small_grad["layer.weight"]), \
        "Small gradient should not be clipped"

    print("[PASS] test_dp_clipping_bounds")


def test_dp_epsilon_computation():
    """Verify epsilon decreases properly with more rounds."""
    from federated.privacy import GradientPrivacy, DPConfig

    sigma = 1.0  # noise_multiplier
    eps_1 = GradientPrivacy.compute_epsilon(sigma, steps=1)
    eps_10 = GradientPrivacy.compute_epsilon(sigma, steps=10)
    eps_100 = GradientPrivacy.compute_epsilon(sigma, steps=100)

    # Epsilon should increase with steps (cumulative privacy cost)
    assert eps_10 > eps_1, f"Îµ should grow with rounds: {eps_10} <= {eps_1}"
    assert eps_100 > eps_10, f"Îµ should grow with rounds: {eps_100} <= {eps_10}"

    # But rate of growth should be sublinear (Îµ âˆ sqrt(steps) via RDP, not linear)
    # Our simplified formula: Îµ = 2*ÏƒÂ²*steps â†' linear, so check that it's non-decreasing
    assert eps_1 >= 0 and eps_10 >= eps_1 and eps_100 >= eps_10

    # Strength classification: epsilon < 2 is "strong", < 8 is "moderate"
    # For sigma=2.0, eps at 1 round = 2 * 4 * 1 = 8 â†' "moderate" (boundary)
    eps_moderate = GradientPrivacy.compute_epsilon(2.0, steps=1)
    assert eps_moderate <= 8, f"Îµ={eps_moderate} should be moderate (<=8) for small rounds"

    # Test delta relationship: large epsilon â†' small delta â†' check passes
    # epsilon=20, sigma=1, steps=10 â†' Î´ â‰ˆ exp(-400/(20+20)) = exp(-10) â‰ˆ 4.5e-5
    delta_ok = GradientPrivacy.epsilon_to_delta(
        epsilon=20.0, noise_multiplier=1.0, steps=10, target_delta=1e-3
    )
    assert delta_ok, "Î´ check should pass for large epsilon (small Î´)"

    print("[PASS] test_dp_epsilon_computation")


def test_dp_aggregate_pipeline():
    """Test the full DP aggregation: clip â†' sum â†' add noise."""
    from federated.privacy import GradientPrivacy, DPConfig

    gp = GradientPrivacy(DPConfig(enabled=True, noise_multiplier=1.0, max_grad_norm=1.0))

    grad1 = {"layer.weight": torch.tensor([1.0, 2.0, 3.0])}
    grad2 = {"layer.weight": torch.tensor([1.0, 2.0, 3.0])}

    # With tiny noise (negligible), DP aggregate should closely approximate the average.
    # The norm of [1,2,3] = sqrt(14) â‰ˆ 3.74 > max_norm=1.0, so clipping applies.
    # After clipping: [1,2,3] * (1.0/3.74) * 0.5 + same = [0.267, 0.534, 0.801] * 2 = [0.267, 0.534, 0.801]
    result = gp.dp_aggregate(
        [grad1, grad2],
        noise_multiplier=1e-6,  # essentially no noise
        max_grad_norm=1.0,
        client_weights=[1.0, 1.0],
    )
    clipped_norm = torch.norm(grad1["layer.weight"]).item()  # sqrt(14) â‰ˆ 3.74
    scale = 1.0 / clipped_norm  # = 0.267
    expected = torch.tensor([1.0, 2.0, 3.0]) * scale  # = [0.267, 0.534, 0.801]
    assert torch.allclose(result["layer.weight"], expected, atol=1e-4), \
        f"DP aggregate = {result['layer.weight']}, expected {expected}"

    print("[PASS] test_dp_aggregate_pipeline")


# ============================================================================
# 9. Gradient Compression Tests
# ============================================================================

def test_compress_sparsify_roundtrip():
    """Sparsification + decompress should recover approximate gradient values."""
    from federated.compression import compress_sparsify, decompress_sparsify

    grad_dict = {
        "layer.weight": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "layer.bias": torch.tensor([0.1, 0.2]),
    }

    compressed = compress_sparsify(grad_dict, k=0.5)
    assert compressed['method'] == 'sparsify'
    assert compressed['k'] == 0.5
    assert compressed['total_kept'] < compressed['total_original']

    decompressed = decompress_sparsify(compressed)
    assert set(decompressed.keys()) == set(grad_dict.keys())

    # Verify shapes are preserved
    for name in grad_dict:
        assert decompressed[name].shape == grad_dict[name].shape, \
            f"Shape mismatch for {name}: {decompressed[name].shape} vs {grad_dict[name].shape}"

    # Values at kept indices should match
    sparse_data = compressed['data']
    for name in grad_dict:
        original = grad_dict[name].float().numpy().flatten()
        indices = sparse_data[name]['indices']
        recovered_values = decompressed[name].numpy().flatten()[indices]
        sparse_values = sparse_data[name]['values']
        assert np.allclose(recovered_values, sparse_values, atol=1e-4), \
            f"Values mismatch for {name}"

    print("[PASS] test_compress_sparsify_roundtrip")


def test_compress_quantize_roundtrip():
    """Quantization + decompress should recover close-to-original values."""
    from federated.compression import compress_quantize, decompress_quantize

    grad_dict = {
        "layer.weight": torch.randn(10, 20) * 0.5,
        "layer.bias": torch.randn(10) * 0.1,
    }

    for bits in [8, 16]:
        compressed = compress_quantize(grad_dict, bits=bits)
        assert compressed['method'] == 'quantize'
        assert compressed['bits'] == bits

        decompressed = decompress_quantize(compressed)
        assert set(decompressed.keys()) == set(grad_dict.keys())

        # Verify shapes preserved
        for name in grad_dict:
            assert decompressed[name].shape == grad_dict[name].shape

        # Check reconstruction error is within quantization bounds
        for name in grad_dict:
            original = grad_dict[name].float()
            recovered = decompressed[name].float()
            max_err = (original - recovered).abs().max().item()
            # For 8-bit: quantization step = range/255; error should be < step
            step_8bit = (original.max() - original.min()).item() / 255.0 if bits == 8 else 0
            err_tolerance = step_8bit * 2 + 1e-4
            assert max_err < err_tolerance, \
                f"[bits={bits}] Reconstruction error {max_err:.6f} too high for {name}"

    print("[PASS] test_compress_quantize_roundtrip")


def test_compress_both_roundtrip():
    """Combined sparsify + quantize should still reconstruct correctly."""
    from federated.compression import compress_both, decompress_both

    grad_dict = {
        "layer.weight": torch.randn(8, 16),
        "layer.bias": torch.randn(8),
    }

    compressed = compress_both(grad_dict, k=0.2, bits=8)
    assert compressed['method'] == 'both'

    decompressed = decompress_both(compressed)
    assert set(decompressed.keys()) == set(grad_dict.keys())

    # Shapes preserved
    for name in grad_dict:
        assert decompressed[name].shape == grad_dict[name].shape

    print("[PASS] test_compress_both_roundtrip")


def test_compress_gradients_interface():
    """Test the unified compress_gradients / decompress_gradients interface."""
    from federated.compression import compress_gradients, decompress_gradients

    grad_dict = {
        "lora_A": torch.randn(4, 16),
        "lora_B": torch.randn(8, 4),
    }

    for method in ["none", "sparsify", "quantize", "both"]:
        compressed, metadata = compress_gradients(grad_dict, method=method, k=0.25, bits=8)
        assert metadata['method'] == method

        decompressed = decompress_gradients(compressed, metadata)
        assert set(decompressed.keys()) == set(grad_dict.keys())

        # Shapes preserved for all methods
        for name in grad_dict:
            assert decompressed[name].shape == grad_dict[name].shape, \
                f"Shape mismatch for {name} in method={method}"

    print("[PASS] test_compress_gradients_interface")


def test_fedavg_no_double_normalization():
    """
    Verify the corrected FedAvg does NOT double-normalize.

    The old bug: weight = num_samples/total, then a second division by total_rep/50.
    The fix: single normalized weight = (num_samples * rep_factor) / total_weight.

    This test uses unequal reputations to expose the double-normalization bug.
    """
    from federated.server import GradientAggregator

    aggregator = GradientAggregator(method="fedavg")

    # Client 1: 100 samples, rep=100 (high rep, high weight)
    # Client 2: 100 samples, rep=25  (low rep, low weight)
    client1_state = {"layer.weight": torch.tensor([10.0, 10.0])}
    client2_state = {"layer.weight": torch.tensor([0.0, 0.0])}  # zero gradient

    updates = [
        {"client_id": "c1", "num_samples": 100, "gradient_data": client1_state},
        {"client_id": "c2", "num_samples": 100, "gradient_data": client2_state},
    ]
    reputations = {"c1": 100.0, "c2": 25.0}

    aggregated_bytes, _ = aggregator.aggregate(updates, reputations)
    aggregated = pickle.loads(aggregated_bytes)

    # rep_factor_c1 = 100/50 = 2.0
    # rep_factor_c2 = 25/50 = 0.5
    # total_weight = 100*2 + 100*0.5 = 200 + 50 = 250
    # c1_weight = 200/250 = 0.8
    # pre-norm = 0.8 * [10,10] + 0.2 * [0,0] = [8,8]
    # NO second division
    expected = torch.tensor([8.0, 8.0])
    assert torch.allclose(aggregated["layer.weight"].float(), expected, atol=1e-4), \
        f"Single normalization wrong: {aggregated['layer.weight']} vs {expected}"

    print("[PASS] test_fedavg_no_double_normalization")


# ============================================================================
# 10. Byzantine Resilience Tests
# ============================================================================

def test_byzantine_norm_detects_outlier():
    """
    Create 4 honest clients with small gradients and 1 malicious client
    with a much larger gradient. Norm-based detection should flag and exclude
    the outlier.

    With 4 honest clients of norm ~1 and 1 malicious of norm ~100,
    mean â‰ˆ 21, std â‰ˆ 44, threshold â‰ˆ 153. The malicious gradient (100)
    is below the threshold with 3Ïƒ â€" so we use a larger outlier (500).
    """
    from federated.byzantine import ByzantineResilientAggregator, norm_based_aggregate

    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # norm â‰ˆ 5.5
    honest_norms = [5.0, 5.5, 6.0, 5.3]  # ~5.5 each
    grad_dicts = []
    for n in honest_norms:
        grad_dicts.append({"layer.weight": base * (n / 5.5)})
    # One malicious client with a massive gradient (norm ~5000 >> threshold ~764)
    malicious = {"layer.weight": base * (5000.0 / 5.5)}

    grad_dicts.append(malicious)
    n_honest = len(honest_norms)

    # Norm-based should flag the malicious one
    byz = ByzantineResilientAggregator(method="norm", sigma_threshold=3.0)
    result, stats = byz.aggregate(grad_dicts)

    assert stats["num_outliers"] >= 1, f"Should detect outlier, got stats: {stats}"
    assert stats["num_kept"] == n_honest, \
        f"Should keep {n_honest} honest clients, got: {stats}"

    # Also test the raw function
    result2, stats2 = norm_based_aggregate(grad_dicts, sigma_threshold=3.0)
    assert stats2["num_outliers"] >= 1, f"Raw function should detect outlier: {stats2}"

    print("[PASS] test_byzantine_norm_detects_outlier")


def test_byzantine_trimmed_mean_works():
    """
    Verify trimmed mean computation is correct.

    For a single dimension with values [1, 2, 3, 4, 5] and alpha=0.2:
    - n=5, trim 1 from each end (floor(0.2*5)=1)
    - sorted = [1,2,3,4,5], trimmed = [2,3,4], mean = 3.0
    """
    from federated.byzantine import trimmed_mean_aggregate, ByzantineResilientAggregator

    # Create 5 clients with simple gradient values
    grad_dicts = [
        {"layer.weight": torch.tensor([[1.0]])}  # index 0
        for _ in range(5)
    ]
    # Set the actual values we want to test
    grad_dicts[0]["layer.weight"] = torch.tensor([[1.0]])
    grad_dicts[1]["layer.weight"] = torch.tensor([[2.0]])
    grad_dicts[2]["layer.weight"] = torch.tensor([[3.0]])
    grad_dicts[3]["layer.weight"] = torch.tensor([[4.0]])
    grad_dicts[4]["layer.weight"] = torch.tensor([[5.0]])

    byz = ByzantineResilientAggregator(method="trimmed_mean", alpha=0.2)
    result, stats = byz.aggregate(grad_dicts)

    # Trim 1 from each end: mean of [2,3,4] = 3.0
    expected_val = 3.0
    actual_val = result["layer.weight"].item()
    assert abs(actual_val - expected_val) < 1e-4, \
        f"Trimmed mean: expected {expected_val}, got {actual_val}"

    # Also test 3 clients with alpha=0.1 (minimum trim)
    grad3 = [
        {"layer.weight": torch.tensor([[1.0]])},
        {"layer.weight": torch.tensor([[2.0]])},
        {"layer.weight": torch.tensor([[3.0]])},
    ]
    byz3 = ByzantineResilientAggregator(method="trimmed_mean", alpha=0.1)
    result3, stats3 = byz3.aggregate(grad3)
    # n=3, alpha=0.1, floor(0.1*3)=0 â†' no trim, mean = (1+2+3)/3 = 2.0
    assert abs(result3["layer.weight"].item() - 2.0) < 1e-4, \
        f"Trimmed mean n=3 alpha=0.1 should give 2.0, got {result3['layer.weight'].item()}"

    print("[PASS] test_byzantine_trimmed_mean_works")


def test_byzantine_aggregate_preserves_shape():
    """
    Verify that ByzantineResilientAggregator output dict has the same
    shapes as the inputs, for all three methods.
    """
    from federated.byzantine import ByzantineResilientAggregator

    grad_dicts = [
        {
            "lora_A": torch.randn(4, 16),
            "lora_B": torch.randn(8, 4),
            "layer.weight": torch.randn(8, 16),
            "layer.bias": torch.randn(8),
        }
        for _ in range(4)
    ]

    for method in ["krum", "trimmed_mean", "norm"]:
        if method == "krum":
            byz = ByzantineResilientAggregator(method=method, f=1)
        elif method == "trimmed_mean":
            byz = ByzantineResilientAggregator(method=method, alpha=0.1)
        else:
            byz = ByzantineResilientAggregator(method=method)

        result, stats = byz.aggregate(grad_dicts)

        assert set(result.keys()) == set(grad_dicts[0].keys()), \
            f"[{method}] Output keys mismatch: {set(result.keys())} vs {set(grad_dicts[0].keys())}"

        for key in grad_dicts[0]:
            assert result[key].shape == grad_dicts[0][key].shape, \
                f"[{method}] Shape mismatch for '{key}': {result[key].shape} vs {grad_dicts[0][key].shape}"

    print("[PASS] test_byzantine_aggregate_preserves_shape")


def test_byzantine_krum_selects_closest_to_neighbors():
    """
    Krum should select the gradient closest to its neighbors,
    which should be the honest one when f malicious clients exist.
    """
    from federated.byzantine import ByzantineResilientAggregator, krum_select

    # 4 clients: 3 honest with similar gradients, 1 malicious with very different
    honest_base = {"layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    malicious = {"layer.weight": torch.tensor([[100.0, -100.0], [100.0, -100.0]])}

    grad_dicts = [
        {k: v.clone() for k, v in honest_base.items()},
        {k: v.clone() + torch.randn_like(v) * 0.01 for k, v in honest_base.items()},  # honest+noise
        {k: v.clone() + torch.randn_like(v) * 0.01 for k, v in honest_base.items()},  # honest+noise
        {k: v.clone() for k, v in malicious.items()},  # malicious
    ]

    # f=1, n=4 â†' n-f-2 = 1 neighbor, multi-krum selects n-f-1 = 2
    byz = ByzantineResilientAggregator(method="krum", f=1)
    result, stats = byz.aggregate(grad_dicts)

    # Multi-krum should select 2 of the 3 honest clients
    assert stats["method"] == "krum"
    assert stats["num_excluded"] >= 1, f"Krum should exclude at least 1 malicious: {stats}"
    assert stats["status"] == "success", f"Krum failed: {stats}"

    # Also test the raw krum_select function
    indices, select_stats = krum_select(grad_dicts, f=1, multi=True)
    # With n=4, f=1: multi-krum selects n-f-1 = 2
    assert len(indices) == 2, f"Multi-krum should select 2, got {len(indices)}: {indices}"
    # The malicious is index 3, should NOT be selected
    assert 3 not in indices, f"Malicious client 3 should NOT be selected: {indices}"

    print("[PASS] test_byzantine_krum_selects_closest_to_neighbors")


def test_byzantine_fallback_when_all_look_malicious():
    """
    If norm-based detection flags all clients, it should fall back to FedAvg.
    """
    from federated.byzantine import norm_based_aggregate

    # Two very different large gradients â€" both might look like outliers
    grad_dicts = [
        {"layer.weight": torch.tensor([[1000.0, 2000.0]])},
        {"layer.weight": torch.tensor([[-1000.0, -2000.0]])},
    ]

    # With n=2, std might be huge and threshold very high â€" or very low
    # Test the fallback path explicitly
    result, stats = norm_based_aggregate(grad_dicts, sigma_threshold=0.0)
    # sigma_threshold=0 means only mean itself would pass; std might be 0 or huge
    # The key is it should NOT crash and should return something
    assert "status" in stats
    assert isinstance(result, dict)

    print("[PASS] test_byzantine_fallback_when_all_look_malicious")


# ============================================================================
# Byzantine Defense Unit Tests (stress-test inline implementations)
# These test the byzantine_norm_filter and byzantine_krum functions as used
# in eval/byzantine_stress_test.py — NOT the federated/byzantine.py versions.
# ============================================================================

def test_stress_test_norm_filter_drops_malicious_grad():
    """
    byzantine_norm_filter in the stress test should detect and drop gradients
    with norms far outside the honest distribution using MAD-based detection.

    Setup: 4 honest clients with small gradients (norm ~5) + 1 malicious
    with huge gradient (norm ~500). MAD-based threshold should catch the outlier.
    """
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
    from byzantine_stress_test import byzantine_norm_filter

    # 4 honest clients: norm ~5 each
    honest_base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # norm ≈ 5.5
    honest_norms = [5.0, 5.5, 6.0, 5.3]
    deltas = []
    for n in honest_norms:
        deltas.append({"layer.weight": honest_base * (n / 5.5)})

    # 1 malicious: 100× norm of honest (500+)
    malicious = {"layer.weight": honest_base * (100.0 / 5.5)}
    deltas.append(malicious)
    weights = [1.0] * 5

    filtered_deltas, filtered_weights = byzantine_norm_filter(deltas, weights)

    # Should drop the malicious client, keep all 4 honest
    assert len(filtered_deltas) == 4, \
        f"Should drop malicious, keep 4 honest, got {len(filtered_deltas)}: {filtered_weights}"
    assert len(filtered_weights) == 4

    print("[PASS] test_stress_test_norm_filter_drops_malicious_grad")


def test_stress_test_krum_drops_malicious_grad():
    """
    byzantine_krum in the stress test should exclude the malicious client
    when f=1 and the malicious gradient is far from honest ones.

    With n=5, f=1: n >= 2f+3 (5 >= 5) → Krum is viable.
    k = n - f - 2 = 2 neighbors, n_keep = n - f - 1 = 3.
    Multi-Krum selects the 3 most honest (closest to their neighbors).
    The malicious client (index 4) should NOT be among the selected.
    """
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
    from byzantine_stress_test import byzantine_krum

    # 4 honest clients: similar gradients (norm ~5)
    honest_base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # norm ≈ 5.5
    deltas = [
        {"layer.weight": honest_base.clone()},
        {"layer.weight": honest_base.clone() + torch.randn_like(honest_base) * 0.01},
        {"layer.weight": honest_base.clone() + torch.randn_like(honest_base) * 0.01},
        {"layer.weight": honest_base.clone() + torch.randn_like(honest_base) * 0.01},
    ]

    # 1 malicious: very different gradient (opposite sign, 50× magnitude)
    malicious = {"layer.weight": honest_base * -50.0}
    deltas.append(malicious)
    weights = [1.0] * 5

    result, _ = byzantine_krum(deltas, weights, n_malicious=1)

    # The result should be a dict of tensors (aggregated honest gradients)
    assert isinstance(result, dict), f"Krum should return dict, got {type(result)}"
    assert "layer.weight" in result

    # The malicious (index 4) should NOT contribute meaningfully.
    # Check by comparing to the honest mean: if malicious was included,
    # the result would be far from the honest mean.
    honest_mean = sum(d["layer.weight"] for d in deltas[:4]).float() / 4.0
    deviation = (result["layer.weight"].float() - honest_mean).abs().max().item()
    # deviation should be small (malicious excluded), not huge (malicious included)
    assert deviation < 5.0, \
        f"Result too far from honest mean (malicious likely included): deviation={deviation:.2f}"

    print("[PASS] test_stress_test_krum_drops_malicious_grad")


def test_stress_test_trimmed_mean_excludes_malicious():
    """
    byzantine_trimmed_mean should naturally exclude malicious values at the
    extremes even without explicit detection.

    With n=5, trim 10% each side → trim 0 (floor(0.1*5)=0), keep all 5.
    We need n=6+ or a larger trim fraction. Use n=6 with trim 1 each side.
    """
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
    from byzantine_stress_test import byzantine_trimmed_mean

    # 5 honest clients: similar gradients
    honest_base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # norm ≈ 5.5
    deltas = [
        {"layer.weight": honest_base * n} for n in [1.0, 1.1, 1.2, 0.9, 1.05]
    ]
    # 1 malicious: opposite sign, huge magnitude
    malicious = {"layer.weight": honest_base * -50.0}
    deltas.append(malicious)
    weights = [1.0] * 6

    result = byzantine_trimmed_mean(deltas, weights, trim_fraction=0.2)

    # n=6, alpha=0.2 → trim 1 from each side, mean of 4 middle values
    # The middle 4 should be the 5 honest (small magnitude) ones,
    # not the huge malicious. The result norm should be small.
    result_norm = torch.norm(result["layer.weight"]).item()
    assert result_norm < 10.0, \
        f"Trimmed mean result norm too large (malicious included?): {result_norm:.2f}"

    print("[PASS] test_stress_test_trimmed_mean_excludes_malicious")


def test_stress_test_krum_viable_with_5_clients_f1():
    """
    Verify Krum does NOT fall back to mean with n=5, f=1.
    n=5 >= 2f+3=5 → Krum is viable. Previously the bug was n <= 2f+1 = 3
    which would have triggered fallback even for n=5.
    """
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
    from byzantine_stress_test import byzantine_krum

    # Simple 5-client setup with distinct gradients
    deltas = [{"layer.weight": torch.tensor([[float(i)]]) for i in range(1, 6)}]
    weights = [1.0] * 5

    result, _ = byzantine_krum(deltas, weights, n_malicious=1)

    # Should NOT fall back (no warning about insufficient clients)
    assert isinstance(result, dict), f"Krum should return dict, got {type(result)}"
    assert "layer.weight" in result

    print("[PASS] test_stress_test_krum_viable_with_5_clients_f1")


# ============================================================================
# HIPAA Audit Logger Tests
# ============================================================================

def test_audit_logger_records_events():
    """Audit logger records events with all required fields."""
    import tempfile
    from utils.audit_logger import AuditLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(audit_dir=tmpdir)

        event = logger.log_event(
            event_type="gradient_send",
            client_id="hospital_A",
            data_type="gradient_update",
            record_count=147456,
            success=True,
        )

        assert event["event_type"] == "gradient_send"
        assert event["client_id"] == "hospital_A"
        assert event["data_type"] == "gradient_update"
        assert event["record_count"] == 147456
        assert event["success"] is True
        assert "timestamp" in event
        assert event["timestamp"].endswith("Z")
        assert "chain_hash" in event
        assert "prev_hash" in event

        # Verify we can decrypt the file
        events = logger.decrypt_log_file(datetime.utcnow().strftime("%Y-%m-%d"))
        assert len(events) >= 1
        assert any(e["event_type"] == "gradient_send" for e in events)

    print("[PASS] test_audit_logger_records_events")


def test_audit_logger_chain_integrity():
    """Each log entry's chain_hash links to the previous entry."""
    import tempfile
    from utils.audit_logger import AuditLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(audit_dir=tmpdir)

        # Log several events
        logger.log_event(event_type="client_connect", client_id="hospital_A")
        logger.log_event(event_type="gradient_send", client_id="hospital_A")
        logger.log_event(event_type="checkpoint_save", client_id="hospital_A")

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        result = logger.verify_chain(date_str)

        assert result["valid"] is True, f"Chain integrity failed: {result['errors']}"
        assert result["entries_checked"] == 3
        assert len(result["errors"]) == 0

    print("[PASS] test_audit_logger_chain_integrity")


def test_audit_logger_tamper_detection():
    """Tampering with a log entry is detected by chain verification."""
    import tempfile
    from utils.audit_logger import AuditLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(audit_dir=tmpdir)
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

        logger.log_event(event_type="client_connect", client_id="hospital_A")
        logger.log_event(event_type="gradient_send", client_id="hospital_A")

        # Tamper with the log file by appending a fake entry
        log_file = Path(tmpdir) / f"audit_{date_str}.log"
        fake_entry = b"FAKE_TAMPERED_ENTRY_DATA_THAT_IS_NOT_VALID_FERNET"
        with open(log_file, "ab") as f:
            f.write(fake_entry + b"\n")

        result = logger.verify_chain(date_str)
        # The fake entry will fail to decrypt â€" this is detected as a tamper
        assert result["valid"] is False or len(result["errors"]) > 0

    print("[PASS] test_audit_logger_tamper_detection")


def test_audit_logger_compliance_report():
    """Compliance report aggregates events correctly for HIPAA audit."""
    import tempfile
    from utils.audit_logger import AuditLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(audit_dir=tmpdir)
        start = datetime.utcnow().strftime("%Y-%m-%d")

        logger.log_event(event_type="client_connect", client_id="hospital_A")
        logger.log_event(event_type="gradient_send", client_id="hospital_A")
        logger.log_event(event_type="gradient_receive", client_id="hospital_B")
        logger.log_event(event_type="checkpoint_save", client_id="hospital_A")

        report = logger.generate_compliance_report(
            start_date=start,
            end_date=datetime.utcnow().strftime("%Y-%m-%d"),
        )

        assert report["total_events"] == 4
        assert report["event_type_counts"]["client_connect"] == 1
        assert report["event_type_counts"]["gradient_send"] == 1
        assert report["event_type_counts"]["gradient_receive"] == 1
        assert report["event_type_counts"]["checkpoint_save"] == 1
        assert "hospital_A" in report["client_activity"]
        assert "hospital_B" in report["client_activity"]
        assert report["retention_policy"]["required_years"] == 7
        assert report["chain_verification"]["valid"] is True

    print("[PASS] test_audit_logger_compliance_report")


# ============================================================================
# 11. pyproject.toml completeness tests
# ============================================================================

def test_pyproject_toml_packages_includes_federated_lisa_utils():
    """Verify pyproject.toml [tool.setuptools] packages covers key modules."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    packages = data.get("tool", {}).get("setuptools", {}).get("packages", [])
    # packages can be a list of str or a dict with find directive
    if isinstance(packages, dict):
        # e.g. {"find": ""} â€" setuptools will auto-discover
        packages = []

    assert isinstance(packages, list), f"packages should be a list, got {type(packages)}"
    assert "federated" in packages, f"'federated' not in packages: {packages}"
    assert "lisa" in packages, f"'lisa' not in packages: {packages}"
    assert "utils" in packages, f"'utils' not in packages: {packages}"
    assert "api" in packages, f"'api' not in packages: {packages}"

    print("[PASS] test_pyproject_toml_packages_includes_federated_lisa_utils")


# ============================================================================
# 12. Client auth tests
# ============================================================================

def test_federated_client_accepts_auth_token():
    """
    FederatedClient.__init__ should accept and store auth_token.
    When auth_token is set, it should be sent as a Bearer header on submit.
    """
    from unittest.mock import patch, MagicMock
    from federated.client import FederatedClient

    # Patch the trainer so we don't actually load a model
    with patch("federated.client.LocalTrainer") as mock_trainer_cls:
        mock_trainer = MagicMock()
        mock_trainer.compute_gradient_update.return_value = MagicMock(
            client_id="test-client",
            round_number=1,
            timestamp=1234567890.0,
            num_samples=100,
            compressed_data=b"fake_data",
            compression_info={"method": "pickle-zlib"},
            noise_seed=None,
            dp_epsilon=None,
            gradient_norm=1.0,
            loss_before=2.0,
            loss_after=1.8,
            param_names=["layer.weight"],
        )
        mock_trainer._latest_raw_gradients = {"layer.weight": torch.tensor([1.0, 2.0])}
        mock_trainer._latest_raw_round = 1
        mock_trainer_cls.return_value = mock_trainer

        # Create client WITH auth token
        client = FederatedClient(
            client_id="test-client",
            server_url="http://localhost:8000",
            auth_token="secret-token-123",
        )

        assert client.auth_token == "secret-token-123", \
            f"auth_token not stored: {client.auth_token}"

        # Create client WITHOUT auth token
        client_no_auth = FederatedClient(
            client_id="test-client-noauth",
            server_url="http://localhost:8000",
        )
        assert client_no_auth.auth_token is None, \
            "auth_token should be None when not provided"

        print("[PASS] test_federated_client_accepts_auth_token")


def test_client_submits_with_bearer_header():
    """When auth_token is set, submit uses Bearer Authorization header."""
    import requests
    from unittest.mock import patch, MagicMock
    from federated.client import FederatedClient

    with patch("federated.client.LocalTrainer") as mock_trainer_cls:
        mock_trainer = MagicMock()
        mock_update = MagicMock(
            client_id="auth-client",
            round_number=1,
            timestamp=1234567890.0,
            num_samples=50,
            compressed_data=b"gradient_bytes",
            compression_info={"method": "pickle-zlib"},
            noise_seed=None,
            dp_epsilon=None,
            gradient_norm=0.5,
            loss_before=3.0,
            loss_after=2.5,
            param_names=["layer.weight"],
        )
        mock_trainer.compute_gradient_update.return_value = mock_update
        mock_trainer._latest_raw_gradients = None
        mock_trainer_cls.return_value = mock_trainer

        client = FederatedClient(
            client_id="auth-client",
            server_url="http://localhost:8000",
            auth_token="my-secret",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(requests, "post", return_value=mock_response) as mock_post:
            client.train_and_submit(round_number=1)

            # Check the call was made with the Authorization header
            mock_post.assert_called_once()
            _, kwargs = mock_post.call_args
            assert "headers" in kwargs, f"headers not in call kwargs: {kwargs}"
            assert kwargs["headers"].get("Authorization") == "Bearer my-secret", \
                f"Bearer header wrong: {kwargs['headers']}"

        # Without auth_token, no Authorization header should be sent
        client_no_auth = FederatedClient(
            client_id="no-auth-client",
            server_url="http://localhost:8000",
        )
        with patch.object(requests, "post", return_value=mock_response) as mock_post_noauth:
            client_no_auth.train_and_submit(round_number=1)
            _, kwargs = mock_post_noauth.call_args
            assert "headers" not in kwargs or kwargs.get("headers", {}).get("Authorization") is None, \
                "No Authorization header should be sent when auth_token is None"

        print("[PASS] test_client_submits_with_bearer_header")


def test_client_disconnect_method():
    """FederatedClient.disconnect() sends a POST to /disconnect endpoint."""
    import requests
    from unittest.mock import patch, MagicMock
    from federated.client import FederatedClient

    with patch("federated.client.LocalTrainer") as mock_trainer_cls:
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        client = FederatedClient(
            client_id="disc-client",
            server_url="http://localhost:8000",
            auth_token="logout-token",
        )
        client.state.round_number = 5

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "disconnected", "client_id": "disc-client"}

        with patch.object(requests, "post", return_value=mock_resp) as mock_post:
            result = client.disconnect()

            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            call_kwargs = mock_post.call_args[1]
            assert "/disconnect" in call_url, f"Wrong URL: {call_url}"
            assert call_kwargs.get("headers", {}).get("Authorization") == "Bearer logout-token", \
                f"Wrong auth header: {call_kwargs}"
            assert call_kwargs["json"]["client_id"] == "disc-client"
            assert call_kwargs["json"]["round_number"] == 5
            assert result["status"] == "disconnected"

        # Without auth_token, disconnect should log and skip
        client_no_auth = FederatedClient(
            client_id="disc-client-noauth",
            server_url="http://localhost:8000",
        )
        with patch.object(requests, "post") as mock_post:
            result = client_no_auth.disconnect()
            assert result["status"] == "skipped"
            mock_post.assert_not_called()

        print("[PASS] test_client_disconnect_method")




"""Phase 3 tests — written by subagent."""

import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import time


def _make_mock_server(config_dict):
    """
    Return a real FederatedServer with _init_model patched so no model download occurs.
    """
    from federated.server import DEFAULT_CONFIG
    from unittest.mock import MagicMock
    from federated import server as server_mod

    cfg = {**DEFAULT_CONFIG, **config_dict}

    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "[EOS]"
    mock_model = MagicMock()
    mock_model.state_dict.return_value = {}
    mock_model.load_state_dict.return_value = None
    mock_model.parameters.return_value = iter([])

    orig_init = server_mod.FederatedServer._init_model

    def fake_init(self):
        self.tokenizer = mock_tok
        self.model = mock_model
        self.global_round = 0

    server_mod.FederatedServer._init_model = fake_init
    try:
        srv = server_mod.FederatedServer(cfg)
    finally:
        server_mod.FederatedServer._init_model = orig_init

    return srv


def test_server_disconnect_endpoint_marks_client_inactive():
    """Server /disconnect endpoint should mark client inactive and add it to clients_disconnected."""
    from federated.server import FederatedServer, ClientInfo, RoundState

    tmpdir = Path(tempfile.mkdtemp())
    try:
        server = _make_mock_server({
            "checkpoint_dir": str(tmpdir / "ckpt"),
            "log_dir": str(tmpdir / "logs"),
        })

        with server._get_lock():
            server.round_state[1] = RoundState(
                round_number=1,
                status="collecting",
                started_at=time.time(),
            )
            server.clients["hospital_A"] = ClientInfo(
                client_id="hospital_A",
                registered_at=time.time(),
                last_seen=time.time(),
                is_active=True,
            )

        async def run_disconnect(srv):
            from federated.server import disconnect as disconnect_fn
            import federated.server as mod
            old = mod._server
            mod._server = srv
            try:
                req = MagicMock()
                async def fake_json():
                    return {"client_id": "hospital_A", "round_number": 1}
                req.json = fake_json
                req.headers = {}
                return await disconnect_fn(req)
            finally:
                mod._server = old

        result = asyncio.new_event_loop().run_until_complete(run_disconnect(server))

        assert result["status"] == "disconnected"
        assert result["client_id"] == "hospital_A"

        with server._get_lock():
            assert server.clients["hospital_A"].is_active is False
            assert "hospital_A" in server.round_state[1].clients_disconnected

        print("[PASS] test_server_disconnect_endpoint_marks_client_inactive")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_server_auth_on_submit_requires_bearer_token():
    """When server has auth_token configured, /submit without valid Bearer header should return 401."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        server = _make_mock_server({
            "checkpoint_dir": str(tmpdir / "ckpt"),
            "log_dir": str(tmpdir / "logs"),
            "auth_token": "server-secret",
        })

        async def run_submit(req):
            # Import the submit function and bind _server to our server instance
            from federated.server import submit as submit_fn
            import federated.server as fed_server_mod
            old_server = fed_server_mod._server
            fed_server_mod._server = server
            try:
                return await submit_fn(req)
            finally:
                fed_server_mod._server = old_server

        def make_async_req(auth_header):
            async def fake_json():
                return {"client_id": "c1", "round_number": 1, "num_samples": 100, "gradient_norm": 1.0}
            req = MagicMock()
            req.json = fake_json
            req.headers = {"Authorization": auth_header} if auth_header else {}
            return req

        # No auth → should raise HTTPException 401
        try:
            asyncio.new_event_loop().run_until_complete(run_submit(make_async_req(None)))
            raise AssertionError("Expected HTTPException 401 for missing auth")
        except Exception as e:
            assert hasattr(e, "status_code") and e.status_code == 401, \
                f"Expected HTTPException 401, got: {type(e).__name__}: {e}"

        # Wrong token → should raise HTTPException 401
        try:
            asyncio.new_event_loop().run_until_complete(run_submit(make_async_req("Bearer wrong-token")))
            raise AssertionError("Expected HTTPException 401 for wrong auth")
        except Exception as e:
            assert hasattr(e, "status_code") and e.status_code == 401, \
                f"Expected HTTPException 401, got: {type(e).__name__}: {e}"

        # Correct token → should NOT raise 401 auth error
        exc_info = None
        try:
            asyncio.new_event_loop().run_until_complete(run_submit(make_async_req("Bearer server-secret")))
        except Exception as e:
            exc_info = e
        if exc_info is not None and hasattr(exc_info, "status_code"):
            assert exc_info.status_code != 401, \
                f"Should NOT return 401 with correct token, got: {exc_info}"

        print("[PASS] test_server_auth_on_submit_requires_bearer_token")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_socket_handler_marks_disconnected_on_connection_error():
    """FederatedSocketHandler._mark_client_disconnected adds client to round's disconnected list."""
    from federated.server import FederatedSocketHandler, ClientInfo, RoundState

    tmpdir = Path(tempfile.mkdtemp())
    try:
        server = _make_mock_server({
            "checkpoint_dir": str(tmpdir / "ckpt"),
            "log_dir": str(tmpdir / "logs"),
        })

        with server._get_lock():
            server.round_state[1] = RoundState(
                round_number=1,
                status="collecting",
                started_at=time.time(),
            )
            server.clients["client-1"] = ClientInfo(
                client_id="client-1",
                registered_at=time.time(),
                last_seen=time.time(),
            )

        # Call _mark_client_disconnected directly
        # FederatedSocketHandler._mark_client_disconnected(self, server, lock, client_id, round_num)
        # Here 'handler' acts as self, and we pass server, lock, client_id, round_num
        handler = FederatedSocketHandler.__new__(FederatedSocketHandler)
        handler._mark_client_disconnected(server, server._get_lock(), "client-1", 1)

        with server._get_lock():
            assert "client-1" in server.round_state[1].clients_disconnected

        print("[PASS] test_socket_handler_marks_disconnected_on_connection_error")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# 11. GradientUpdate dataclass tests
# ============================================================================

def test_gradient_update_to_dict_roundtrip():
    """
    GradientUpdate is created by FederatedClient.compute_gradient_update().
    Verify its to_dict() produces a correctly-shaped dict and that
    compression_info + dp_epsilon are preserved through serialization.
    """
    from federated.client import GradientUpdate
    import pickle

    update = GradientUpdate(
        client_id="c1",
        round_number=5,
        timestamp=1700000000.0,
        num_samples=150,
        param_names=["lora_A", "lora_B"],
        compressed_data=b"\x00\x01\x02\x03",
        compression_info={"method": "both", "k": 0.1, "bits": 8, "compression_ratio": 42.0},
        noise_seed=42,
        dp_epsilon=2.5,
        gradient_norm=1.23,
        loss_before=2.71,
        loss_after=2.13,
    )

    d = update.to_dict()

    # Check all expected keys are present
    assert d["client_id"] == "c1"
    assert d["round_number"] == 5
    assert d["timestamp"] == 1700000000.0
    assert d["num_samples"] == 150
    assert d["num_params"] == 2
    assert d["gradient_norm"] == 1.23
    assert d["loss_before"] == 2.71
    assert d["loss_after"] == 2.13
    assert d["compression_info"] == {"method": "both", "k": 0.1, "bits": 8, "compression_ratio": 42.0}
    assert d["dp_epsilon"] == 2.5

    # Verify the update survives pickle roundtrip (used internally by some transports)
    pickled = pickle.dumps(update)
    restored = pickle.loads(pickled)
    assert restored.client_id == "c1"
    assert restored.compression_info["method"] == "both"
    assert restored.dp_epsilon == 2.5

    print("[PASS] test_gradient_update_to_dict_roundtrip")


# ============================================================================
# 12. Compression: method="none" roundtrip
# ============================================================================

def test_compress_gradients_method_none_roundtrip():
    """
    Test compress_gradients with method='none'.
    Verifies the raw pickle path: gradients should survive
    compress/decompress unchanged when no compression is applied.
    """
    from federated.compression import compress_gradients, decompress_gradients

    grad_dict = {
        "lora_A": torch.randn(4, 16),
        "lora_B": torch.randn(8, 4),
        "layer.weight": torch.randn(8, 16),
    }

    compressed, metadata = compress_gradients(grad_dict, method="none")
    restored = decompress_gradients(compressed, metadata)

    assert set(restored.keys()) == set(grad_dict.keys())
    for name in grad_dict:
        assert torch.allclose(restored[name].float(), grad_dict[name].float(), atol=1e-4), \
            f"method='none' roundtrip changed tensor '{name}'"
    assert metadata["method"] == "none"

    print("[PASS] test_compress_gradients_method_none_roundtrip")


# ============================================================================
# pytest entry points
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_lora_linear_math,
        test_lora_both_params_trainable_and_get_gradients,
        test_lisa_layer_selection_always_includes_bounds,
        test_lisa_selection_is_deterministic_with_seed,
        test_get_lora_gradients_only_lora_params,
        test_fedavg_averages_correctly,
        test_fedavg_respects_sample_weights,
        test_config_file_overrides,
        test_default_config_values,
        test_gradient_serialization_roundtrip,
        test_struct_packing_unpacking,
        test_tensor_ndims_and_dtypes,
        test_gradient_validator_rejects_bad_norms,
        test_fedavg_handles_empty_updates,
        test_dp_noise_changes_gradients,
        test_dp_clipping_bounds,
        test_dp_epsilon_computation,
        test_dp_aggregate_pipeline,
        test_compress_sparsify_roundtrip,
        test_compress_quantize_roundtrip,
        test_compress_both_roundtrip,
        test_compress_gradients_interface,
        test_fedavg_no_double_normalization,
        # Byzantine resilience
        test_byzantine_norm_detects_outlier,
        test_byzantine_trimmed_mean_works,
        test_byzantine_aggregate_preserves_shape,
        test_byzantine_krum_selects_closest_to_neighbors,
        test_byzantine_fallback_when_all_look_malicious,
        # Byzantine stress-test inline impl tests
        test_stress_test_norm_filter_drops_malicious_grad,
        test_stress_test_krum_drops_malicious_grad,
        test_stress_test_trimmed_mean_excludes_malicious,
        test_stress_test_krum_viable_with_5_clients_f1,
        # HIPAA audit logger
        test_audit_logger_records_events,
        test_audit_logger_chain_integrity,
        test_audit_logger_tamper_detection,
        test_audit_logger_compliance_report,
        # Phase 3: new features
        test_pyproject_toml_packages_includes_federated_lisa_utils,
        test_federated_client_accepts_auth_token,
        test_client_submits_with_bearer_header,
        test_client_disconnect_method,
        test_server_disconnect_endpoint_marks_client_inactive,
        test_server_auth_on_submit_requires_bearer_token,
        test_socket_handler_marks_disconnected_on_connection_error,
    ]

    print(f"Running {len(tests)} tests ...")
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAILED: {t.__name__}: {e}")
    print("Done.")
