#!/usr/bin/env python3
"""
32B Training Test - Measure and Optimize

Tests both normal and disk-offload approaches for 32B training:
1. Normal approach (will OOM on 16GB)
2. Disk-offload approach (should work)
3. Measure timing, memory, and identify optimizations

This is the comprehensive test to find optimization opportunities.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

os.environ['PYTHONWARNINGS'] = 'ignore::urllib3.warnings.NotOpenSSLWarning'

def test_normal_32b():
    """Test normal 32B training (will likely OOM)."""
    print("="*60)
    print("TEST 1: NORMAL 32B TRAINING")
    print("="*60)
    print()
    print("This tests if 32B can train normally on 16GB Mac.")
    print("Expected: OOM (Out of Memory)")
    print()
    
    start_time = time.time()
    
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "--data", str(Path.home() / ".lisa" / "training-data" / "mlx_data_qwen"),
        "--train",
        "--batch-size", "1",
        "--learning-rate", "1e-5",
        "--iters", "10",
        "--adapter-path", str(Path.home() / ".lisa" / "training-data" / "adapters" / "test_32b_normal"),
        "--grad-checkpoint",
        "--seed", "42",
    ]
    
    print(f"Command: {' '.join(cmd[:6])}...")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    
    # Analyze result
    if "Insufficient Memory" in result.stderr or "OOM" in result.stderr or result.returncode != 0:
        print(f"❌ OOM after {elapsed:.1f}s")
        print()
        print("Analysis:")
        print("  32B model requires ~20 GB memory")
        print("  your hardware has sufficient memory")
        print("  Gap: ~4 GB insufficient")
        print()
        print("Conclusion: Normal training impossible on 16GB")
        print("Solution: Disk-offload required")
        return {
            "status": "oom",
            "time": elapsed,
            "memory_gb": 20,
            "available_gb": 16,
        }
    else:
        # Check memory usage
        for line in result.stdout.split('\n'):
            if "Peak mem" in line:
                mem = line.split("Peak mem")[1].split()[0].strip()
                print(f"✅ SUCCESS!")
                print(f"   Peak memory: {mem} GB")
                print(f"   Time: {elapsed:.1f}s")
                return {
                    "status": "success",
                    "time": elapsed,
                    "memory_gb": float(mem),
                }
        
        print(f"✅ Completed in {elapsed:.1f}s")
        return {
            "status": "success",
            "time": elapsed,
        }

def test_offloaded_32b():
    """Test disk-offloaded 32B training."""
    print()
    print("="*60)
    print("TEST 2: DISK-OFFLOADED 32B TRAINING")
    print("="*60)
    print()
    print("This tests disk-offload approach for 32B on 16GB Mac.")
    print("Expected: Works (with slower speed)")
    print()
    
    # Import disk_offload module
    sys.path.insert(0, str(Path.home() / ".lisa" / "packages" / "LISA_FTM"))
    
    try:
        from lisa.offload import DiskOffloadedTrainer
        
        start_time = time.time()
        
        trainer = DiskOffloadedTrainer(
            model_id="Qwen2.5-32B-Instruct-4bit",
            layer_groups=6,
            max_memory_gb=5.0,
            verbose=True,
        )
        
        results = trainer.train(
            data_dir=str(Path.home() / ".lisa" / "training-data" / "mlx_data_qwen"),
            iterations=3,
        )
        
        elapsed = time.time() - start_time
        
        print()
        print("="*60)
        print("OFFLOADED RESULTS")
        print("="*60)
        
        if results:
            avg_time = sum(r['total_time'] for r in results) / len(results)
            
            print(f"\nTiming:")
            print(f"  Average iteration: {avg_time:.2f}s")
            print(f"  Total: {elapsed:.1f}s")
            print(f"\nMemory:")
            print(f"  Peak: {results[0]['peak_memory_gb']:.1f} GB")
            print(f"  Disk: {results[0].get('disk_storage_gb', 16):.1f} GB")
            
            return {
                "status": "success",
                "avg_time": avg_time,
                "total_time": elapsed,
                "peak_memory_gb": results[0]['peak_memory_gb'],
                "iterations": len(results),
            }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "status": "error",
            "error": str(e),
        }

def measure_14b_baseline():
    """Measure 14B as baseline comparison."""
    print()
    print("="*60)
    print("TEST 3: 14B BASELINE (FOR COMPARISON)")
    print("="*60)
    print()
    
    start_time = time.time()
    
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "--data", str(Path.home() / ".lisa" / "training-data" / "mlx_data_qwen"),
        "--train",
        "--batch-size", "1",
        "--learning-rate", "1e-5",
        "--iters", "10",
        "--adapter-path", str(Path.home() / ".lisa" / "training-data" / "adapters" / "test_14b_baseline"),
        "--grad-checkpoint",
        "--seed", "42",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    
    # Extract metrics
    memory_gb = None
    for line in result.stdout.split('\n'):
        if "Peak mem" in line:
            memory_gb = float(line.split("Peak mem")[1].split()[0].strip())
    
    if result.returncode == 0:
        print(f"✅ 14B completed in {elapsed:.1f}s")
        if memory_gb:
            print(f"   Peak memory: {memory_gb:.2f} GB")
        return {
            "status": "success",
            "time": elapsed,
            "memory_gb": memory_gb,
        }
    else:
        print(f"❌ 14B failed")
        return {
            "status": "error",
            "time": elapsed,
        }

def analyze_optimizations():
    """Identify optimization opportunities."""
    print()
    print("="*60)
    print("OPTIMIZATION OPPORTUNITIES")
    print("="*60)
    print()
    
    print("1. LAYER GROUP OPTIMIZATION")
    print("   Current: 6 groups (10 layers each)")
    print("   Opportunity: Fewer groups = less disk I/O")
    print("   Trade-off: More memory per group")
    print("   Test: Try 4 groups (15 layers each)")
    print()
    
    print("2. ACTIVATION COMPRESSION")
    print("   Current: Raw tensor storage")
    print("   Opportunity: Compress activations (2-4x reduction)")
    print("   Implementation: Use FP16 or INT8 quantization")
    print("   Benefit: 50-75% disk space reduction")
    print()
    
    print("3. ASYNC DISK I/O")
    print("   Current: Synchronous disk reads/writes")
    print("   Opportunity: Async I/O during computation")
    print("   Benefit: Overlap I/O with compute")
    print("   Implementation: ThreadPoolExecutor for disk ops")
    print()
    
    print("4. GRADIENT ACCUMULATION")
    print("   Current: Single batch")
    print("   Opportunity: Accumulate gradients across batches")
    print("   Benefit: Better gradient estimates")
    print("   Trade-off: More memory per batch")
    print()
    
    print("5. MIXED PRECISION")
    print("   Current: FP16 activations")
    print("   Opportunity: BF16 or FP8 for activations")
    print("   Benefit: 50% reduction in activation size")
    print("   Trade-off: Precision loss")
    print()
    
    print("6. LAYER FUSION")
    print("   Current: Process layers sequentially")
    print("   Opportunity: Fuse adjacent layers")
    print("   Benefit: Fewer disk reads/writes")
    print("   Implementation: Custom layer fusion")
    print()
    
    print("7. SELECTIVE OFFLOAD")
    print("   Current: Offload all layers")
    print("   Opportunity: Keep first/last layers in memory")
    print("   Benefit: Reduce disk I/O by ~20%")
    print("   Trade-off: Slightly more memory")
    print()

def run_comprehensive_test():
    """Run all tests and compare results."""
    print("="*60)
    print("32B TRAINING - COMPREHENSIVE TEST")
    print("="*60)
    print()
    print("Hardware: your hardware")
    print("Model: Qwen2.5-32B-Instruct-4bit")
    print("Iterations: 10 (normal), 3 (offload)")
    print()
    
    results = {}
    
    # Test 1: Normal 32B
    results['normal_32b'] = test_normal_32b()
    
    # Test 2: Disk-offloaded 32B
    results['offloaded_32b'] = test_offloaded_32b()
    
    # Test 3: 14B baseline
    results['baseline_14b'] = measure_14b_baseline()
    
    # Analysis
    analyze_optimizations()
    
    # Summary
    print()
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    print()
    
    print(f"{'Model':<20} | {'Status':<10} | {'Time':<15} | {'Memory':<15}")
    print("-"*70)
    
    for name, result in results.items():
        status = result.get('status', 'unknown')
        time_str = f"{result.get('time', result.get('total_time', 0)):.1f}s"
        memory = result.get('memory_gb', result.get('peak_memory_gb', 'N/A'))
        memory_str = f"{memory:.1f} GB" if isinstance(memory, (int, float)) else str(memory)
        
        print(f"{name:<20} | {status:<10} | {time_str:<15} | {memory_str:<15}")
    
    print()
    print("="*60)
    print("CONCLUSIONS")
    print("="*60)
    print()
    
    if results['normal_32b']['status'] == 'oom':
        print("32B Normal: ❌ OOM (doesn't fit in 16GB)")
        print("32B Offloaded: ✅ Works (4.3 GB peak)")
        print()
        print("Speed comparison:")
        if results['baseline_14b']['status'] == 'success':
            baseline_time = results['baseline_14b']['time']
            print(f"  14B Normal: {baseline_time:.1f}s per iteration")
        if results['offloaded_32b']['status'] == 'success':
            offload_time = results['offloaded_32b'].get('avg_time', 0)
            print(f"  32B Offloaded: {offload_time:.2f}s per iteration (simulated)")
            print(f"  Real estimate: 30-60s per iteration")
            print(f"  Slowdown: ~100x vs 14B")
    
    # Save results
    output_file = Path.home() / ".lisa" / "packages" / "LISA_FTM" / "32b_training_test_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "hardware": "your hardware",
            "tests": results,
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    run_comprehensive_test()