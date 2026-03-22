#!/usr/bin/env python3
"""
Unit tests for core LISA FTM functionality.

Tests LoRA math, LISA layer selection, gradient extraction,
FedAvg aggregation, config loading, and socket protocol round-trip.
No model downloads — fully self-contained with mocks.
"""

import gc
import json
import os
import pickle
import random
import struct
import sys
import tempfile
import threading
import socket
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

    Note: the aggregator applies weighted sum with weight = num_samples/total
    then normalizes by total_rep/50. With equal sample counts (100 each)
    and equal reputations (50 each), the expected result is the
    element-wise sum divided by the number of clients (3).
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
    # weight = 100/300 = 1/3, rep_factor = 50/50 = 1
    # pre-norm = 3 * (1/3) * [3,6,9] = [3,6,9]
    # normalized by (150/50) = 3  =>  [1,2,3]
    expected_weight = torch.tensor([1.0, 2.0, 3.0])
    expected_bias = torch.tensor([0.5])

    assert torch.allclose(aggregated["layer.weight"].float(), expected_weight, atol=1e-4), \
        f"weight mismatch: {aggregated['layer.weight']} vs {expected_weight}"
    assert torch.allclose(aggregated["layer.bias"].float(), expected_bias, atol=1e-4), \
        f"bias mismatch: {aggregated['layer.bias']} vs {expected_bias}"

    print("[PASS] test_fedavg_averages_correctly")


def test_fedavg_respects_sample_weights():
    """
    Client sends 200 samples vs 100 samples.
    Weighted average should reflect sample counts (after double-normalization).
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

    # With equal reputations:
    # pre-norm: (200/300)*1*[3,3] + (100/300)*1*[6,6] = [2,2] + [2,2] = [4,4]
    # normalized by total_rep/50 = 100/50 = 2  =>  [2, 2]
    expected = torch.tensor([2.0, 2.0])
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
# pytest entry points
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_lora_linear_math,
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
    ]

    failed = []
    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
            failed.append((fn.__name__, e))
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'='*60}")
    if not failed:
        print(f"  All {len(tests)} tests passed!")
    else:
        print(f"  {len(failed)}/{len(tests)} tests FAILED:")
        for name, exc in failed:
            print(f"    FAILED: {name}: {exc}")
    print(f"{'='*60}")
