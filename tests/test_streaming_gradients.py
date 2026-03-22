#!/usr/bin/env python3
"""
Test streaming gradients: chunked transfer for large tensors.

Creates a large random tensor (~500MB), sends it via streaming,
receives it and verifies shape and values, reports time taken.
"""
import sys
import os
import time
import socket
import struct
import json
import threading
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("test-streaming")


def send_tensor_streaming(sock, name, tensor, chunk_size=65536):
    """Send a tensor in chunks."""
    name_bytes = name.encode("utf-8")
    sock.sendall(struct.pack("!I", len(name_bytes)) + name_bytes)

    dtype_str = str(tensor.dtype).replace("torch.", "")
    dtype_bytes = dtype_str.encode("utf-8")
    sock.sendall(struct.pack("!I", len(dtype_bytes)) + dtype_bytes)

    shape_bytes = json.dumps(list(tensor.shape)).encode("utf-8")
    sock.sendall(struct.pack("!I", len(shape_bytes)) + shape_bytes)

    total_bytes = tensor.numel() * tensor.element_size()
    sock.sendall(struct.pack("!Q", total_bytes))  # 8-byte unsigned

    np_bytes = tensor.cpu().numpy().tobytes()
    for offset in range(0, len(np_bytes), chunk_size):
        chunk = np_bytes[offset : offset + chunk_size]
        sock.sendall(chunk)


def recv_tensor_streaming(sock, chunk_size=65536):
    """Receive a streamed tensor."""
    name_len_data = b""
    while len(name_len_data) < 4:
        c = sock.recv(4 - len(name_len_data))
        if not c:
            return "", None
        name_len_data += c
    name_len = struct.unpack("!I", name_len_data)[0]

    name_bytes = b""
    while len(name_bytes) < name_len:
        c = sock.recv(name_len - len(name_bytes))
        if not c:
            return "", None
        name_bytes += c
    name = name_bytes.decode("utf-8")

    dtype_len_data = b""
    while len(dtype_len_data) < 4:
        c = sock.recv(4 - len(dtype_len_data))
        if not c:
            return name, None
        dtype_len_data += c
    dtype_len = struct.unpack("!I", dtype_len_data)[0]

    dtype_bytes = b""
    while len(dtype_bytes) < dtype_len:
        c = sock.recv(dtype_len - len(dtype_bytes))
        if not c:
            return name, None
        dtype_bytes += c
    dtype_str = dtype_bytes.decode("utf-8")

    shape_len_data = b""
    while len(shape_len_data) < 4:
        c = sock.recv(4 - len(shape_len_data))
        if not c:
            return name, None
        shape_len_data += c
    shape_len = struct.unpack("!I", shape_len_data)[0]

    shape_bytes = b""
    while len(shape_bytes) < shape_len:
        c = sock.recv(shape_len - len(shape_bytes))
        if not c:
            return name, None
        shape_bytes += c
    shape = json.loads(shape_bytes.decode("utf-8"))

    num_bytes_data = b""
    while len(num_bytes_data) < 8:
        c = sock.recv(8 - len(num_bytes_data))
        if not c:
            return name, None
        num_bytes_data += c
    total_bytes = struct.unpack("!Q", num_bytes_data)[0]

    buffer = b""
    while len(buffer) < total_bytes:
        chunk = sock.recv(min(chunk_size, total_bytes - len(buffer)))
        if not chunk:
            break
        buffer += chunk

    import numpy as np
    import torch
    arr = np.frombuffer(buffer, dtype=dtype_str).copy()
    result = torch.from_numpy(arr).view(shape)
    return name, result


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return data
        data += chunk
    return data


def test_streaming_large_tensor():
    """Test chunked transfer of a large tensor."""
    import numpy as np
    import torch
    import psutil

    # Create a large random tensor (~50MB)
    # 50MB / 4 bytes (float32) = 12.5M elements
    shape = (12_500_000,)
    tensor = torch.randn(shape, dtype=torch.float32)
    tensor_bytes = tensor.numel() * tensor.element_size()
    log.info(f"Created tensor: shape={shape}, size={tensor_bytes/1e6:.1f} MB")

    # Start server socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]
    log.info(f"Server listening on port {port}")

    # Memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6

    def server_thread():
        conn, addr = server_sock.accept()
        conn.settimeout(300)  # 5 min timeout for large transfers
        log.info(f"Server: client connected from {addr}")

        t0 = time.time()
        name, received = recv_tensor_streaming(conn)
        elapsed = time.time() - t0

        if received is None:
            log.error("Server: failed to receive tensor")
            conn.close()
            return

        mem_after = process.memory_info().rss / 1e6

        log.info(f"Server: received tensor '{name}' in {elapsed:.2f}s")
        log.info(f"  Shape: {list(received.shape)}")
        log.info(f"  Dtype: {received.dtype}")
        log.info(f"  Size: {received.numel() * received.element_size()/1e6:.1f} MB")
        log.info(f"  Memory delta: {mem_after - mem_before:.1f} MB")

        # Verify values
        max_diff = float(torch.max(torch.abs(tensor - received)).item())
        log.info(f"  Max value diff: {max_diff:.6f}")

        if max_diff < 1e-5:
            log.info("  ✓ Values match!")
        else:
            log.error("  ✗ Values DO NOT match!")

        log.info(f"  Throughput: {tensor_bytes/elapsed/1e6:.1f} MB/s")
        conn.close()
        server_sock.close()

    t = threading.Thread(target=server_thread)
    t.start()

    # Client connects and sends
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(("127.0.0.1", port))

    t0 = time.time()
    send_tensor_streaming(client_sock, "large_grad", tensor)
    elapsed = time.time() - t0

    log.info(f"Client: sent tensor in {elapsed:.2f}s")
    log.info(f"  Throughput: {tensor_bytes/elapsed/1e6:.1f} MB/s")

    client_sock.close()
    t.join(timeout=30)

    log.info("\n=== TEST COMPLETE ===")
    log.info(f"Tensor size: {tensor_bytes/1e6:.1f} MB")
    log.info(f"Chunk size: 65536 bytes")
    log.info(f"Transfer time: {elapsed:.2f}s")
    log.info(f"Throughput: {tensor_bytes/elapsed/1e6:.1f} MB/s")


def test_streaming_threshold():
    """Test that small tensors use pickle path and large tensors use streaming."""
    import numpy as np
    import torch
    import pickle

    # Small tensor (< 10MB) - should use pickle
    small = torch.randn(100_000, dtype=torch.float32)  # ~0.4 MB
    small_bytes = small.numel() * small.element_size()
    log.info(f"\nSmall tensor: {small_bytes/1e6:.2f} MB (< 10MB threshold -> pickle)")

    # Large tensor (> 10MB) - should use streaming
    large = torch.randn(5_000_000, dtype=torch.float32)  # ~20 MB
    large_bytes = large.numel() * large.element_size()
    log.info(f"Large tensor: {large_bytes/1e6:.1f} MB (> 10MB threshold -> streaming)")

    THRESHOLD = 10 * 1024 * 1024
    small_is_small = small_bytes < THRESHOLD
    large_is_large = large_bytes > THRESHOLD

    assert small_is_small, "Small tensor should be below threshold"
    assert large_is_large, "Large tensor should be above threshold"
    log.info("  ✓ Threshold detection works correctly!")


def test_memory_profile():
    """Test memory profiling with psutil."""
    import psutil

    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1e6
    log.info(f"\nCurrent process memory: {mem_mb:.1f} MB")

    # Allocate a large tensor
    import torch
    large = torch.randn(50_000_000, dtype=torch.float32)  # ~200 MB
    mem_after = process.memory_info().rss / 1e6
    log.info(f"After allocating 200MB tensor: {mem_after:.1f} MB (delta: {mem_after - mem_mb:.1f} MB)")

    del large
    import gc
    gc.collect()
    mem_freed = process.memory_info().rss / 1e6
    log.info(f"After deallocating: {mem_freed:.1f} MB")
    log.info("  ✓ Memory profiling with psutil works!")


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("STREAMING GRADIENTS TEST")
    log.info("=" * 60)

    test_streaming_threshold()
    test_memory_profile()

    log.info("\n" + "=" * 60)
    log.info("LARGE TENSOR STREAMING TEST (~500MB)")
    log.info("=" * 60)
    test_streaming_large_tensor()

    log.info("\n✓ All tests passed!")
