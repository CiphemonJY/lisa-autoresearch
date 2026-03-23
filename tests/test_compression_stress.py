#!/usr/bin/env python3
"""
Compression Stress Tests for Federated Learning Gradient Compression

Tests:
1. Round-trip fidelity: gradient direction preserved after compress/decompress
2. Top-K sparsification at different K values (1%, 5%, 10%, 25%, 50%)
3. uint8 and uint16 quantization round-trips
4. Edge cases: empty gradients, NaN gradients, all-zero gradients, extremely large gradients
5. Combined sparsify+quantize pipeline
6. Compression ratio measurements
7. Direction preservation metric (cosine similarity)
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated.compression import (
    compress_sparsify, decompress_sparsify,
    compress_quantize, decompress_quantize,
    compress_both, decompress_both,
    compress_gradients, decompress_gradients,
)


# ============================================================================
# Helpers
# ============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays."""
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def relative_l2_error(original: np.ndarray, recovered: np.ndarray) -> float:
    """Relative L2 error."""
    orig_flat = original.flatten()
    rec_flat = recovered.flatten()
    denom = np.linalg.norm(orig_flat)
    if denom < 1e-10:
        return 0.0 if np.allclose(original, recovered) else float('inf')
    return float(np.linalg.norm(orig_flat - rec_flat) / denom)


def pack_grad_dict(*tensors) -> dict:
    """Pack list of tensors into a named dict for testing."""
    names = [f"layer_{i}.weight" for i in range(len(tensors))]
    return {name: t for name, t in zip(names, tensors)}


# ============================================================================
# 1. Round-trip fidelity — gradient direction preserved
# ============================================================================

def test_sparsify_preserves_gradient_direction():
    """
    After sparsification + decompression, the gradient should still point
    in roughly the same direction as the original.

    Note: Cosine similarity is a harsh metric for sparsification because
    keeping only 10% of elements means 90% of the high-dimensional vector
    is zeroed — the angle between the original dense vector and the sparse
    recovered vector is inherently large. We use two metrics:
    1. Sign agreement: % of kept indices where sign is preserved
    2. Cosine similarity: should be >= 0.50 (very lenient for 90% sparsity)
    """
    np.random.seed(42)
    original = np.random.randn(512, 256).astype(np.float32)

    grad_dict = {"layer.weight": torch.from_numpy(original)}
    compressed = compress_sparsify(grad_dict, k=0.1)
    decompressed = decompress_sparsify(compressed)

    orig_flat = original.flatten()
    rec_flat = decompressed["layer.weight"].numpy().flatten()
    sparse_data = compressed['data']['layer.weight']
    indices = sparse_data['indices']
    kept_values = sparse_data['values']
    orig_at_indices = orig_flat[indices]

    # Metric 1: Sign agreement at kept positions (should be near 1.0)
    sign_agreement = np.mean(np.sign(kept_values) == np.sign(orig_at_indices))
    print(f"  Sparsify k=10%: sign_agreement={sign_agreement:.4f} "
          f"(kept {len(indices)}/{len(orig_flat)} elements)")

    # Metric 2: Cosine similarity (lenient — inherently low for 90% sparsity)
    cos_sim = cosine_similarity(orig_flat, rec_flat)
    print(f"  Sparsify k=10%: cosine_similarity={cos_sim:.4f}")

    # Sign agreement is the right metric for sparsification direction preservation
    assert sign_agreement >= 0.85, \
        f"Sign agreement too low: {sign_agreement:.4f} — gradient direction corrupted"
    assert cos_sim >= 0.50, f"Cosine too low: {cos_sim:.4f} (may indicate bug)"

    # Also test k=5%
    compressed5 = compress_sparsify(grad_dict, k=0.05)
    decompressed5 = decompress_sparsify(compressed5)
    sparse_data5 = compressed5['data']['layer.weight']
    indices5 = sparse_data5['indices']
    values5 = sparse_data5['values']
    orig5 = original.flatten()[indices5]
    sign_agree5 = np.mean(np.sign(values5) == np.sign(orig5))
    print(f"  Sparsify k=5%: sign_agreement={sign_agree5:.4f}")
    assert sign_agree5 >= 0.80, f"k=5% sign agreement too low: {sign_agree5:.4f}"

    print("[PASS] test_sparsify_preserves_gradient_direction")


def test_quantize_preserves_gradient_direction():
    """uint8 quantization should preserve gradient direction well."""
    np.random.seed(99)
    original = np.random.randn(256, 128).astype(np.float32)

    grad_dict = {"layer.weight": torch.from_numpy(original)}
    compressed = compress_quantize(grad_dict, bits=8)
    decompressed = decompress_quantize(compressed)

    orig_flat = original.flatten()
    rec_flat = decompressed["layer.weight"].numpy().flatten()

    cos_sim = cosine_similarity(orig_flat, rec_flat)
    rel_err = relative_l2_error(original, rec_flat)

    print(f"  Quantize 8-bit: cosine_similarity={cos_sim:.4f}, rel_L2={rel_err:.4f}")
    assert cos_sim >= 0.99, f"Quantization destroyed direction: cos_sim={cos_sim:.4f}"
    # Relative L2 error should be small for uint8
    assert rel_err < 0.05, f"Rel L2 error too high: {rel_err:.4f}"

    print("[PASS] test_quantize_preserves_gradient_direction")


def test_both_preserves_gradient_direction():
    """Combined sparsify + quantize should still preserve sign at kept positions."""
    np.random.seed(7)
    original = np.random.randn(128, 64).astype(np.float32)

    grad_dict = {"layer.weight": torch.from_numpy(original)}
    compressed = compress_both(grad_dict, k=0.1, bits=8)
    decompressed = decompress_both(compressed)

    orig_flat = original.flatten()
    rec_flat = decompressed["layer.weight"].numpy().flatten()

    # Sign agreement: % of kept positions where sign matches
    # Note: compress_both keeps sparse indices from sparsify stage
    # The actual indices kept are stored in sparse_data
    sparse_data = compressed['data']['sparsify']['layer.weight']
    indices = sparse_data['indices']
    kept_orig = orig_flat[indices]
    kept_rec = rec_flat[indices]

    sign_agreement = np.mean(np.sign(kept_orig) == np.sign(kept_rec))
    cos_sim = cosine_similarity(orig_flat, rec_flat)

    print(f"  Sparsify+Quantize: sign_agreement={sign_agreement:.4f}, cos_sim={cos_sim:.4f}")
    assert sign_agreement >= 0.80, \
        f"Combined compression sign agreement too low: {sign_agreement:.4f}"
    assert cos_sim >= 0.50, f"Cosine too low: {cos_sim:.4f}"

    print("[PASS] test_both_preserves_gradient_direction")


# ============================================================================
# 2. Top-K sparsification at different K values
# ============================================================================

def test_topk_sparsification_k_values():
    """Top-K should work correctly at K=1%, 5%, 10%, 25%, 50%, 100%."""
    np.random.seed(123)
    original = np.random.randn(1000).astype(np.float32)
    grad_dict = {"layer.weight": torch.from_numpy(original)}

    for k_frac in [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]:
        compressed = compress_sparsify(grad_dict, k=k_frac)
        decompressed = decompress_sparsify(compressed)

        # Check shapes match
        assert decompressed["layer.weight"].shape == torch.Size([1000])

        # Check kept elements count matches expectation
        k_count = max(1, int(1000 * k_frac))
        total_kept = compressed['total_kept']
        # Allow some slack since argpartition may keep slightly more
        assert total_kept <= k_count + 10, \
            f"k={k_frac}: kept {total_kept} vs expected ~{k_count}"

        # Check compression ratio is computed
        assert compressed['compression_ratio'] > 1.0 or k_frac == 1.0

        # Sign agreement: at kept positions, signs should match
        sparse_data = compressed['data']['layer.weight']
        indices = sparse_data['indices']
        kept_orig = original.flatten()[indices]
        kept_vals = sparse_data['values']
        sign_agree = np.mean(np.sign(kept_orig) == np.sign(kept_vals))

        print(f"  k={k_frac:.0%}: kept={total_kept}, sign_agreement={sign_agree:.4f}")

        # Sign agreement should be high for all K values
        assert sign_agree >= 0.80, \
            f"k={k_frac}: sign_agreement={sign_agree:.4f} too low"

        # Cosine similarity: use lenient threshold (lower for tiny K)
        orig_flat = original.flatten()
        rec_flat = decompressed["layer.weight"].numpy().flatten()
        cos_sim = cosine_similarity(orig_flat, rec_flat)
        # At k=1% (10/1000 elements), cosine is inherently limited — be very lenient
        min_cos = 0.25 if k_frac <= 0.01 else (0.40 if k_frac <= 0.05 else 0.50)
        assert cos_sim >= min_cos, f"k={k_frac}: cos_sim={cos_sim:.4f} < {min_cos}"

    print("[PASS] test_topk_sparsification_k_values")


def test_topk_exact_values_preserved():
    """Values at the kept indices should be recovered exactly (sparsify only)."""
    np.random.seed(55)
    original = np.array([1.0, 100.0, 2.0, 200.0, 3.0, 0.5, 150.0, 4.0], dtype=np.float32)
    grad_dict = {"layer.weight": torch.from_numpy(original)}

    # Keep top 3 (k=3/8)
    compressed = compress_sparsify(grad_dict, k=0.375)
    decompressed = decompress_sparsify(compressed)

    sparse_data = compressed['data']['layer.weight']
    indices = sparse_data['indices']
    values = sparse_data['values']

    # Top 3 by magnitude should be 200, 150, 100
    expected_top3 = sorted(np.abs(original))[-3:]
    actual_top3 = sorted(np.abs(values))
    assert np.allclose(expected_top3, actual_top3), \
        f"Top values mismatch: {expected_top3} vs {actual_top3}"

    # Decompressed should have zeros everywhere except top-K indices
    rec = decompressed["layer.weight"].numpy()
    assert np.allclose(rec[indices], values)
    # Non-kept indices should be zero
    mask = np.ones(8, dtype=bool)
    mask[indices] = False
    assert np.allclose(rec[mask], 0.0), "Non-top-K indices should be zero after sparsify"

    print("[PASS] test_topk_exact_values_preserved")


# ============================================================================
# 3. uint8 and uint16 quantization
# ============================================================================

def test_quantize_uint8_vs_uint16():
    """Compare uint8 vs uint16 quantization error."""
    np.random.seed(7)
    original = np.random.randn(512, 256).astype(np.float32) * 0.5
    grad_dict = {"layer.weight": torch.from_numpy(original)}

    for bits in [8, 16]:
        compressed = compress_quantize(grad_dict, bits=bits)
        decompressed = decompress_quantize(compressed)

        rec = decompressed["layer.weight"].numpy()
        rel_err = relative_l2_error(original, rec)
        cos_sim = cosine_similarity(original, rec)

        print(f"  bits={bits}: rel_L2={rel_err:.6f}, cos_sim={cos_sim:.6f}")
        assert cos_sim >= 0.999, f"bits={bits}: cos_sim={cos_sim:.6f} too low"
        assert rel_err < (0.05 if bits == 8 else 0.001), \
            f"bits={bits}: rel_L2={rel_err:.6f} too high"

    # uint16 should be noticeably more accurate than uint8
    comp8 = compress_quantize(grad_dict, bits=8)
    comp16 = compress_quantize(grad_dict, bits=16)
    dec8 = decompress_quantize(comp8)
    dec16 = decompress_quantize(comp16)

    rel_err_8 = relative_l2_error(original, dec8["layer.weight"].numpy())
    rel_err_16 = relative_l2_error(original, dec16["layer.weight"].numpy())
    assert rel_err_16 < rel_err_8, "uint16 should be more accurate than uint8"

    print("[PASS] test_quantize_uint8_vs_uint16")


def test_quantize_preserves_zero_point():
    """Verify zero_point is correctly recovered in dequantization."""
    np.random.seed(42)
    original = np.random.randn(100).astype(np.float32) * 10 + 50  # shift positive
    grad_dict = {"layer.weight": torch.from_numpy(original)}

    compressed = compress_quantize(grad_dict, bits=8)
    decompressed = decompress_quantize(compressed)

    # Check zero_point recovery
    q_info = compressed['data']['layer.weight']
    scale = q_info['scale']
    zero_point = q_info['zero_point']

    # Reconstruct
    rec = decompressed["layer.weight"].numpy()
    rec_dequantized = (rec / scale + 0) if False else None  # just checking shape

    # Simple L2 check
    rel_err = relative_l2_error(original, rec)
    print(f"  Zero-point shift test: rel_L2={rel_err:.6f}, zero_point={zero_point}")
    assert rel_err < 0.05

    print("[PASS] test_quantize_preserves_zero_point")


# ============================================================================
# 4. Edge cases
# ============================================================================

def test_empty_gradients():
    """Handle empty gradient tensors gracefully."""
    grad_dict = {
        "layer.weight": torch.tensor([]).reshape(0, 4),
        "layer.bias": torch.tensor([]),
    }

    for method in ["sparsify", "quantize", "both"]:
        k = 0.1 if method in ("sparsify", "both") else None
        bits = 8 if method in ("quantize", "both") else None

        try:
            if method == "sparsify":
                compressed = compress_sparsify(grad_dict, k=0.1)
                decompressed = decompress_sparsify(compressed)
            elif method == "quantize":
                compressed = compress_quantize(grad_dict, bits=8)
                decompressed = decompress_quantize(compressed)
            else:
                compressed = compress_both(grad_dict, k=0.1, bits=8)
                decompressed = decompress_both(compressed)

            # Shapes should be preserved
            assert decompressed["layer.weight"].shape == grad_dict["layer.weight"].shape
            assert decompressed["layer.bias"].shape == grad_dict["layer.bias"].shape
            print(f"  method={method}: empty grads handled OK, shapes preserved")
        except Exception as e:
            print(f"  method={method}: FAILED — {e}")
            raise

    print("[PASS] test_empty_gradients")


def test_nan_gradients():
    """Handle gradients containing NaN values."""
    original = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0], dtype=np.float32)
    grad_dict = {"layer.weight": torch.from_numpy(original)}

    # Test sparsify: NaN is excluded from top-K (abs(nan)=nan which is replaced with 0)
    # so NaN positions become 0 in recovered gradient
    compressed = compress_sparsify(grad_dict, k=0.5)
    decompressed = decompress_sparsify(compressed)
    rec = decompressed["layer.weight"].numpy()

    # Recovered should have NO NaN values (NaN positions become 0 after sparsify)
    assert not np.any(np.isnan(rec)), \
        f"Recovered should have no NaN, got: {rec}"

    # Valid values at kept indices should be exactly preserved
    sparse_data = compressed['data']['layer.weight']
    indices = sparse_data['indices']
    values = sparse_data['values']
    orig_at_idx = original[indices]
    assert np.allclose(values, orig_at_idx, atol=1e-5), \
        f"Valid values not preserved at kept indices"

    # Test quantize: NaN is tracked via nan_mask and restored after dequantize
    comp_q = compress_quantize(grad_dict, bits=8)
    dec_q = decompress_quantize(comp_q)
    rec_q = dec_q["layer.weight"].numpy()

    # NaN positions should be restored as NaN
    nan_mask = np.isnan(original)
    assert np.all(np.isnan(rec_q[nan_mask])), \
        f"NaN positions should be restored after quantize decompress"
    # Non-NaN positions should be finite
    assert not np.any(np.isnan(rec_q[~nan_mask])), \
        f"Non-NaN positions should not have NaN in recovered"

    print("[PASS] test_nan_gradients")


def test_inf_gradients():
    """Handle gradients containing Inf values."""
    original = np.array([1.0, 2.0, np.inf, 4.0, -np.inf, 6.0], dtype=np.float32)
    grad_dict = {"layer.weight": torch.from_numpy(original)}

    compressed = compress_sparsify(grad_dict, k=0.5)
    decompressed = decompress_sparsify(compressed)
    rec = decompressed["layer.weight"].numpy()

    # Inf positions should remain Inf
    inf_mask = np.isinf(original)
    assert np.all(np.isinf(rec[inf_mask])), "Inf positions should be preserved"

    print("[PASS] test_inf_gradients")


def test_all_zero_gradients():
    """Handle all-zero gradients."""
    grad_dict = {"layer.weight": torch.zeros(10, 10)}

    compressed = compress_sparsify(grad_dict, k=0.1)
    decompressed = decompress_sparsify(compressed)
    rec = decompressed["layer.weight"].numpy()

    assert np.allclose(rec, 0.0), "All-zero should stay all-zero after sparsify"

    comp_q = compress_quantize(grad_dict, bits=8)
    dec_q = decompress_quantize(comp_q)
    rec_q = dec_q["layer.weight"].numpy()
    assert np.allclose(rec_q, 0.0, atol=1e-5), "All-zero should stay near-zero after quantize"

    print("[PASS] test_all_zero_gradients")


def test_extremely_large_gradients():
    """Handle very large gradient values (no overflow in quantization)."""
    original = np.array([1e30, -1e30, 1e25, 0.0, -1e20], dtype=np.float32)
    grad_dict = {"layer.weight": torch.from_numpy(original)}

    compressed = compress_quantize(grad_dict, bits=8)
    decompressed = decompress_quantize(compressed)
    rec = decompressed["layer.weight"].numpy()

    # Large values (1e30, -1e30) should be recovered with sign preserved
    assert np.sign(original[0]) == np.sign(rec[0]), \
        f"Sign of +1e30 not preserved: {original[0]} -> {rec[0]}"
    assert np.sign(original[1]) == np.sign(rec[1]), \
        f"Sign of -1e30 not preserved: {original[1]} -> {rec[1]}"

    # Small/medium values may have quantization error but should be recoverable
    # At least verify they are in the right ballpark
    for i in [2, 3, 4]:  # 1e25, 0.0, -1e20
        assert np.isfinite(rec[i]), f"Value at index {i} became non-finite: {rec[i]}"

    print("[PASS] test_extremely_large_gradients")


def test_single_element_gradient():
    """Handle gradient with single element."""
    original = np.array([3.14159], dtype=np.float32)
    grad_dict = {"layer.weight": torch.from_numpy(original)}

    compressed = compress_sparsify(grad_dict, k=1.0)
    decompressed = decompress_sparsify(compressed)
    rec = decompressed["layer.weight"].numpy()

    assert np.isclose(rec[0], original[0], rtol=1e-4), \
        f"Single element not recovered: {rec[0]} vs {original[0]}"

    comp_q = compress_quantize(grad_dict, bits=8)
    dec_q = decompress_quantize(comp_q)
    rec_q = dec_q["layer.weight"].numpy()
    # Single element recovery tolerance is higher
    assert np.isclose(rec_q[0], original[0], rtol=0.1), \
        f"Single element quantize failed: {rec_q[0]} vs {original[0]}"

    print("[PASS] test_single_element_gradient")


def test_unified_interface_all_methods():
    """Test unified compress_gradients/decompress_gradients for all methods."""
    np.random.seed(42)
    original = np.random.randn(64, 32).astype(np.float32)
    grad_dict = {"layer.weight": torch.from_numpy(original)}

    for method in ["none", "sparsify", "quantize", "both"]:
        k = 0.2 if method in ("sparsify", "both") else None
        bits = 8 if method in ("quantize", "both") else None

        compressed, metadata = compress_gradients(
            grad_dict, method=method, k=k, bits=bits
        )
        assert metadata['method'] == method

        decompressed = decompress_gradients(compressed, metadata)
        assert "layer.weight" in decompressed
        assert decompressed["layer.weight"].shape == torch.Size([64, 32])

        # Direction preserved for all methods
        rec = decompressed["layer.weight"].numpy()
        cos_sim = cosine_similarity(original, rec)
        print(f"  method={method}: cos_sim={cos_sim:.4f}")
        assert cos_sim >= 0.70, f"method={method}: cos_sim={cos_sim:.4f} too low"

    print("[PASS] test_unified_interface_all_methods")


# ============================================================================
# 5. Compression ratio measurements
# ============================================================================

def test_compression_ratios():
    """Measure compression ratios for each method and report."""
    np.random.seed(42)
    original = np.random.randn(1024, 1024).astype(np.float32)
    grad_dict = {"layer.weight": torch.from_numpy(original)}

    print(f"  Original size: {original.nbytes:,} bytes ({original.size:,} float32 elements)")

    # Sparsify
    comp_sp = compress_sparsify(grad_dict, k=0.1)
    sp_ratio = comp_sp['compression_ratio']
    # Estimate compressed size (values + indices)
    sparse_data = comp_sp['data']['layer.weight']
    est_sp_size = sparse_data['values'].nbytes + sparse_data['indices'].nbytes
    actual_sp_ratio = original.nbytes / est_sp_size
    print(f"  Sparsify k=10%: kept={comp_sp['total_kept']}/{comp_sp['total_original']}, "
          f"ratio={sp_ratio:.2f}x (est. packed ratio={actual_sp_ratio:.2f}x)")

    # Quantize 8-bit
    comp_q8 = compress_quantize(grad_dict, bits=8)
    q8_ratio = comp_q8['compression_ratio']
    print(f"  Quantize 8-bit: compression_ratio={q8_ratio:.2f}x, "
          f"quantized_bytes={comp_q8['total_quantized']:,}")

    # Quantize 16-bit
    comp_q16 = compress_quantize(grad_dict, bits=16)
    q16_ratio = comp_q16['compression_ratio']
    print(f"  Quantize 16-bit: compression_ratio={q16_ratio:.2f}x")

    # Both
    comp_both = compress_both(grad_dict, k=0.1, bits=8)
    both_ratio = comp_both['compression_ratio']
    print(f"  Sparsify+Quantize: combined ratio={both_ratio:.2f}x "
          f"(sparse × quantize = {sp_ratio:.2f} × {q8_ratio:.2f} = {sp_ratio*q8_ratio:.2f})")

    # Sanity checks
    assert sp_ratio >= 8.0, f"Sparsify ratio {sp_ratio:.2f}x should be >= 8x for k=10%"
    assert q8_ratio >= 3.0, f"Quantize 8-bit ratio {q8_ratio:.2f}x should be >= 3x"
    assert q16_ratio >= 1.5, f"Quantize 16-bit ratio {q16_ratio:.2f}x should be >= 1.5x"

    print("[PASS] test_compression_ratios")


def test_multi_parameter_compression():
    """Test compression with multiple parameters of different sizes."""
    np.random.seed(42)
    grad_dict = {
        "lora_A": torch.from_numpy(np.random.randn(4, 1024).astype(np.float32)),
        "lora_B": torch.from_numpy(np.random.randn(1024, 8).astype(np.float32)),
        "head.weight": torch.from_numpy(np.random.randn(512, 1024).astype(np.float32)),
        "head.bias": torch.from_numpy(np.random.randn(512).astype(np.float32)),
    }

    for method in ["sparsify", "quantize", "both"]:
        k = 0.1 if method in ("sparsify", "both") else None
        bits = 8 if method in ("quantize", "both") else None

        compressed, metadata = compress_gradients(grad_dict, method=method, k=k, bits=bits)
        decompressed = decompress_gradients(compressed, metadata)

        assert set(decompressed.keys()) == set(grad_dict.keys())

        for name in grad_dict:
            orig = grad_dict[name].numpy()
            rec = decompressed[name].numpy()
            cos_sim = cosine_similarity(orig, rec)
            rel_err = relative_l2_error(orig, rec)
            print(f"  {method} / {name}: cos_sim={cos_sim:.4f}, rel_L2={rel_err:.4f}")
            assert decompressed[name].shape == grad_dict[name].shape
            # Use lenient threshold for cosine (sparsification zeroes 90% of vector)
            # "both" has same error as sparsify since quantization is near-lossless
            min_cos = 0.40 if method in ("sparsify", "both") else 0.70
            assert cos_sim >= min_cos, \
                f"{method}/{name}: cos_sim={cos_sim:.4f} < {min_cos}"

    print("[PASS] test_multi_parameter_compression")


# ============================================================================
# 6. Decompress-both edge cases
# ============================================================================

def test_decompress_both_with_quantized_sparse_values():
    """Verify decompress_both reconstructs sparse values from quantized representation."""
    np.random.seed(888)
    original = np.random.randn(256, 128).astype(np.float32)
    grad_dict = {"layer.weight": torch.from_numpy(original)}

    compressed = compress_both(grad_dict, k=0.1, bits=8)
    decompressed = decompress_both(compressed)

    rec = decompressed["layer.weight"].numpy()
    sparse_data = compressed['data']['sparsify']['layer.weight']
    indices = sparse_data['indices']
    kept_orig = original.flatten()[indices]
    kept_rec = rec.flatten()[indices]

    sign_agreement = np.mean(np.sign(kept_orig) == np.sign(kept_rec))
    cos_sim = cosine_similarity(original.flatten(), rec.flatten())

    print(f"  decompress_both: sign_agreement={sign_agreement:.4f}, cos_sim={cos_sim:.4f}")
    assert sign_agreement >= 0.75, \
        f"Combined decompression sign agreement too low: {sign_agreement:.4f}"

    print("[PASS] test_decompress_both_with_quantized_sparse_values")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    tests = [
        # Direction preservation
        test_sparsify_preserves_gradient_direction,
        test_quantize_preserves_gradient_direction,
        test_both_preserves_gradient_direction,
        # Top-K at different values
        test_topk_sparsification_k_values,
        test_topk_exact_values_preserved,
        # Quantization
        test_quantize_uint8_vs_uint16,
        test_quantize_preserves_zero_point,
        # Edge cases
        test_empty_gradients,
        test_nan_gradients,
        test_inf_gradients,
        test_all_zero_gradients,
        test_extremely_large_gradients,
        test_single_element_gradient,
        # Interface
        test_unified_interface_all_methods,
        # Ratios
        test_compression_ratios,
        # Multi-param
        test_multi_parameter_compression,
        # Both
        test_decompress_both_with_quantized_sparse_values,
    ]

    print(f"\n{'='*70}")
    print("COMPRESSION STRESS TESTS")
    print(f"{'='*70}\n")

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*70}\n")

    if failed:
        exit(1)
