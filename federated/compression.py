#!/usr/bin/env python3
"""
Gradient Compression for Federated Learning

Implements two compression strategies:
1. Top-K Sparsification - keep only the K% largest-magnitude gradients
2. Tensor Quantization - reduce precision from float32 to uint8/uint16

These reduce bandwidth when sending gradients over the network.
"""

import json
import logging
from typing import Dict, Any, Tuple, List

import numpy as np
import torch

logger = logging.getLogger("compression")


# ============================================================================
# Top-K Sparsification
# ============================================================================

def compress_sparsify(grad_dict: Dict[str, torch.Tensor], k: float = 0.1) -> Dict[str, Any]:
    """
    Compress gradients by keeping only the top K% largest-magnitude elements.

    Args:
        grad_dict: Dict mapping parameter names to gradient tensors
        k: Fraction of elements to keep (0.0 to 1.0)

    Returns:
        Dict containing:
          - 'data': {name: {'values': ..., 'indices': ..., 'shape': ...}} for sparse tensors
          - 'original_sizes': {name: original_num_elements}
          - 'k': the k value used
          - 'method': 'sparsify'
    """
    sparse_data = {}
    original_sizes = {}
    total_elements = 0
    kept_elements = 0

    for name, tensor in grad_dict.items():
        if not isinstance(tensor, (torch.Tensor, np.ndarray)):
            continue

        # Convert to numpy float32
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().float().numpy()
        else:
            arr = tensor.astype(np.float32)

        original_sizes[name] = arr.size
        total_elements += arr.size

        # Flatten and find threshold
        flat = arr.flatten()
        k_count = max(1, int(len(flat) * k))

        # Get indices of top K% by magnitude, excluding NaN/Inf
        abs_flat = np.abs(flat)
        # Replace NaN/Inf with 0 so they are NOT selected as "largest"
        abs_flat = np.where(np.isfinite(abs_flat), abs_flat, 0.0)

        # Use argpartition for efficiency (O(n) vs O(n log n))
        if k_count < len(flat):
            threshold_idx = np.argpartition(abs_flat, -k_count)[-k_count]
            threshold = abs_flat[threshold_idx]
        else:
            threshold = 0.0

        # Keep elements above threshold OR that are Inf (Inf is always "large")
        has_inf = np.isinf(flat)
        mask = (abs_flat >= threshold) | has_inf
        indices = np.where(mask)[0]
        values = flat[indices]

        kept_elements += len(values)

        sparse_data[name] = {
            'values': values.astype(np.float32),
            'indices': indices.astype(np.int32),
            'shape': arr.shape,
        }

    compression_ratio = total_elements / max(kept_elements, 1)

    return {
        'data': sparse_data,
        'original_sizes': original_sizes,
        'total_original': total_elements,
        'total_kept': kept_elements,
        'k': k,
        'method': 'sparsify',
        'compression_ratio': compression_ratio,
    }


def decompress_sparsify(sparse_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Reconstruct full gradient tensors from sparsified representation.

    Args:
        sparse_dict: Output of compress_sparsify()

    Returns:
        Dict mapping parameter names to reconstructed full gradient tensors
    """
    result = {}
    sparse_data = sparse_dict['data']

    for name, sparse_info in sparse_data.items():
        values = sparse_info['values']
        indices = sparse_info['indices']
        shape = tuple(sparse_info['shape'])

        # Reconstruct full array
        full = np.zeros(np.prod(shape), dtype=np.float32)
        full[indices] = values

        # Reshape to original shape
        result[name] = torch.from_numpy(full.reshape(shape))

    return result


# ============================================================================
# Tensor Quantization
# ============================================================================

def compress_quantize(grad_dict: Dict[str, torch.Tensor], bits: int = 8) -> Dict[str, Any]:
    """
    Compress gradients by quantizing float32 to low-precision integers.

    Args:
        grad_dict: Dict mapping parameter names to gradient tensors
        bits: Number of bits for quantization (8 or 16)

    Returns:
        Dict containing:
          - 'data': {name: {'quantized': ..., 'scale': ..., 'zero_point': ..., 'shape': ...}}
          - 'original_sizes': {name: original_size_in_bytes}
          - 'bits': quantization bits used
          - 'method': 'quantize'
    """
    if bits == 8:
        dtype = np.uint8
        max_val = 255
    elif bits == 16:
        dtype = np.uint16
        max_val = 65535
    else:
        raise ValueError(f"bits must be 8 or 16, got {bits}")

    quantized_data = {}
    original_sizes = {}
    total_original = 0
    total_quantized = 0

    for name, tensor in grad_dict.items():
        if not isinstance(tensor, (torch.Tensor, np.ndarray)):
            continue

        # Convert to numpy float32
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().float().numpy()
        else:
            arr = tensor.astype(np.float32)

        original_sizes[name] = arr.nbytes
        total_original += arr.nbytes

        flat = arr.flatten()

        # Handle empty arrays
        if flat.size == 0:
            quantized_data[name] = {
                'quantized': np.array([], dtype=dtype),
                'scale': 1.0,
                'zero_point': 0,
                'shape': arr.shape,
                'dtype': str(dtype),
            }
            total_quantized += 8  # just scale + zero_point
            continue

        # Handle NaN values: track positions and replace with fill value for quantization
        nan_mask = np.isnan(flat)
        num_nan = np.sum(nan_mask)
        if num_nan > 0:
            fill_val = 0.0  # replace NaN with 0 for quantization
            flat = flat.copy()
            flat[nan_mask] = fill_val

        # Compute scale and zero point for quantization
        min_val = float(flat.min())
        max_val_arr = float(flat.max())

        if max_val_arr - min_val > 1e-8:
            scale = (max_val_arr - min_val) / max_val
            zero_point = int(-min_val / scale)
        else:
            scale = 1.0
            zero_point = 128

        # Quantize
        quantized = np.clip((flat / scale + zero_point), 0, max_val).astype(dtype)

        total_quantized += quantized.nbytes + 8  # +8 for scale and zero_point

        # Store NaN mask so we can restore NaN positions during decompression
        nan_mask_bytes = nan_mask.tobytes() if num_nan > 0 else b''
        nan_mask_shape = arr.shape if num_nan > 0 else ()

        quantized_data[name] = {
            'quantized': quantized,
            'scale': float(scale),
            'zero_point': int(zero_point),
            'shape': arr.shape,
            'dtype': str(dtype),
            'nan_mask': nan_mask_bytes,
            'nan_mask_shape': nan_mask_shape,
            'nan_count': int(num_nan),
        }

    compression_ratio = total_original / max(total_quantized, 1)

    return {
        'data': quantized_data,
        'original_sizes': original_sizes,
        'total_original': total_original,
        'total_quantized': total_quantized,
        'bits': bits,
        'method': 'quantize',
        'compression_ratio': compression_ratio,
    }


def decompress_quantize(quantized_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Reconstruct full float32 tensors from quantized representation.

    Args:
        quantized_dict: Output of compress_quantize()

    Returns:
        Dict mapping parameter names to reconstructed full gradient tensors
    """
    result = {}
    quantized_data = quantized_dict['data']

    for name, q_info in quantized_data.items():
        quantized = q_info['quantized']
        scale = q_info['scale']
        zero_point = q_info['zero_point']
        shape = tuple(q_info['shape'])

        # Convert to float32
        if isinstance(quantized, np.ndarray):
            flat = quantized.astype(np.float32)
        else:
            flat = np.frombuffer(bytes(quantized), dtype=q_info.get('dtype', 'uint8')).astype(np.float32)

        # Dequantize in float64 to avoid overflow with extreme values
        # (float32 has max ~3.4e38, but intermediate (val-zero)*scale can overflow
        #  when scale ~1e27 and val ~1e30, even though result fits in float32)
        dequantized = ((flat.astype(np.float64) - zero_point) * float(scale)).astype(np.float32)

        # Restore NaN positions if we stored a NaN mask
        nan_mask_bytes = q_info.get('nan_mask', b'')
        nan_count = q_info.get('nan_count', 0)
        nan_mask_shape = q_info.get('nan_mask_shape', shape)
        if nan_count > 0 and nan_mask_bytes:
            # Use uint8 for cross-platform-safe mask deserialization (np.bool_ is deprecated)
            nan_mask = np.frombuffer(nan_mask_bytes, dtype=np.uint8).astype(np.bool_)
            nan_mask = nan_mask.reshape(nan_mask_shape).flatten()
            dequantized[nan_mask] = float('nan')

        # Reshape to original shape
        result[name] = torch.from_numpy(dequantized.reshape(shape))

    return result


# ============================================================================
# Combined Compression Pipeline
# ============================================================================

def compress_both(grad_dict: Dict[str, torch.Tensor], k: float = 0.1, bits: int = 8) -> Dict[str, Any]:
    """
    Apply both sparsification and quantization for maximum compression.

    First sparsify (keep top K%), then quantize the sparse representation.
    """
    # First: sparsify
    sparse = compress_sparsify(grad_dict, k=k)

    # Convert sparse values to tensors for quantization
    sparse_values_dict = {}
    for name, info in sparse['data'].items():
        sparse_values_dict[name] = torch.from_numpy(info['values'])

    # Second: quantize the sparse representation
    quantized = compress_quantize(sparse_values_dict, bits=bits)

    # Combine metadata
    return {
        'data': {
            'sparsify': sparse['data'],
            'quantize': quantized['data'],
        },
        'sparse_k': k,
        'quantize_bits': bits,
        'method': 'both',
        'compression_ratio': sparse['compression_ratio'] * quantized['compression_ratio'],
        'total_original': sparse['total_original'],
        'total_kept_after_sparse': sparse['total_kept'],
        'total_quantized_bytes': quantized['total_quantized'],
    }


def decompress_both(combined_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Decompress from combined sparsification + quantization.
    """
    k = combined_dict.get('sparse_k', 0.1)
    bits = combined_dict.get('quantize_bits', 8)

    sparse_data = combined_dict['data']['sparsify']
    quantized_data = combined_dict['data']['quantize']

    # Reconstruct sparse values from quantized
    reconstructed_sparse = decompress_quantize({'data': quantized_data, 'bits': bits, 'method': 'quantize'})

    # Reconstruct full gradients from sparse
    sparse_reconstruct = {}
    for name in sparse_data:
        sparse_reconstruct[name] = {
            'values': reconstructed_sparse.get(name, torch.tensor([])).numpy(),
            'indices': sparse_data[name]['indices'],
            'shape': sparse_data[name]['shape'],
        }

    return decompress_sparsify({'data': sparse_reconstruct})


# ============================================================================
# Unified Compress/Decompress Interface
# ============================================================================

def compress_gradients(
    grad_dict: Dict[str, torch.Tensor],
    method: str = "none",
    k: float = 0.1,
    bits: int = 8,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compress gradients using the specified method.

    Args:
        grad_dict: Parameter name -> gradient tensor
        method: 'none', 'sparsify', 'quantize', or 'both'
        k: Sparsification ratio (for sparsify/both)
        bits: Quantization bits (for quantize/both)

    Returns:
        (compressed_dict, metadata_dict)
    """
    if method == "none":
        # No compression - just serialize
        import pickle
        data = pickle.dumps(grad_dict)
        return {'raw': data}, {'method': 'none', 'original_size': len(data)}

    elif method == "sparsify":
        compressed = compress_sparsify(grad_dict, k=k)
        metadata = {
            'method': 'sparsify',
            'k': k,
            'compression_ratio': compressed['compression_ratio'],
            'total_original': compressed['total_original'],
            'total_kept': compressed['total_kept'],
        }
        return compressed, metadata

    elif method == "quantize":
        compressed = compress_quantize(grad_dict, bits=bits)
        metadata = {
            'method': 'quantize',
            'bits': bits,
            'compression_ratio': compressed['compression_ratio'],
            'total_original': compressed['total_original'],
            'total_quantized': compressed['total_quantized'],
        }
        return compressed, metadata

    elif method == "both":
        compressed = compress_both(grad_dict, k=k, bits=bits)
        metadata = {
            'method': 'both',
            'k': k,
            'bits': bits,
            'compression_ratio': compressed['compression_ratio'],
            'total_original': compressed['total_original'],
        }
        return compressed, metadata

    else:
        raise ValueError(f"Unknown compression method: {method}")


def decompress_gradients(
    compressed: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Decompress gradients using the method specified in metadata.
    """
    method = metadata.get('method', 'none')

    if method == "none":
        import pickle
        return pickle.loads(compressed.get('raw', compressed))

    elif method == "sparsify":
        return decompress_sparsify(compressed)

    elif method == "quantize":
        return decompress_quantize(compressed)

    elif method == "both":
        return decompress_both(compressed)

    else:
        raise ValueError(f"Unknown compression method: {method}")
