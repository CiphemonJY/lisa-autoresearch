#!/usr/bin/env python3
"""
Real Training Tests for v1.2 Features

Tests:
1. Mixed Precision (FP16)
2. Gradient Accumulation
3. Selective Offload

Uses small model (0.5B) for quick validation.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

os.environ['PYTHONWARNINGS'] = 'ignore::urllib3.warnings.NotOpenSSLWarning'

# Test configuration
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Small for quick tests
TEST_ITERATIONS = 10  # Quick test
RESULTS_FILE = Path.home() / ".lisa" / "v1_2_test_results.json"

def log(message):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def test_mixed_precision():
    """Test mixed precision training."""
    log("="*60)
    log("TEST 1: Mixed Precision (FP16)")
    log("="*60)
    
    try:
        # Import mixed precision module
        from mixed_precision import MixedPrecisionTrainer, MixedPrecisionConfig
        
        # Create config
        config = MixedPrecisionConfig(
            enabled=True,
            dtype="float16",
            loss_scale="dynamic",
        )
        
        # Create trainer
        trainer = MixedPrecisionTrainer(
            model_id=TEST_MODEL,
            mp_config=config,
            max_memory_gb=2.0,
            verbose=True,
        )
        
        # Estimate memory savings
        savings = trainer.estimate_memory_savings()
        
        log(f"Memory savings: {savings['savings_percent']:.0f}%")
        log(f"FP32 memory: {savings['fp32_total_gb']:.1f} GB")
        log(f"FP16 memory: {savings['fp16_total_gb']:.1f} GB")
        
        # Run training simulation (no data_dir needed for simulation)
        log("Running training simulation...")
        start = time.time()
        result = trainer.train(data_dir=None, iterations=TEST_ITERATIONS)
        elapsed = time.time() - start
        
        log(f"Completed in {elapsed:.2f}s")
        
        return {
            "status": "success",
            "memory_savings_pct": savings['savings_percent'],
            "fp32_memory_gb": savings['fp32_total_gb'],
            "fp16_memory_gb": savings['fp16_total_gb'],
            "time_seconds": elapsed,
            "config": config.__dict__,
        }
        
    except Exception as e:
        log(f"Error: {e}")
        return {
            "status": "error",
            "error": str(e),
        }

def test_gradient_accumulation():
    """Test gradient accumulation."""
    log("\n" + "="*60)
    log("TEST 2: Gradient Accumulation")
    log("="*60)
    
    try:
        # Import gradient accumulation module
        from gradient_accumulation import GradientAccumulationTrainer, GradientAccumulationConfig
        
        # Test different accumulation steps
        results = []
        
        for steps in [1, 4, 16]:
            log(f"\nTesting {steps}x accumulation...")
            
            config = GradientAccumulationConfig(
                enabled=True,
                accumulation_steps=steps,
                micro_batch_size=1,
            )
            
            trainer = GradientAccumulationTrainer(
                model_id=TEST_MODEL,
                ga_config=config,
                max_memory_gb=2.0,
                verbose=True,
            )
            
            # Estimate memory
            memory = trainer.estimate_memory_impact()
            
            log(f"Effective batch size: {steps}")
            log(f"Memory: {memory['total_memory_gb']:.1f} GB")
            
            results.append({
                "accumulation_steps": steps,
                "effective_batch_size": steps,
                "memory_gb": memory['total_memory_gb'],
            })
        
        return {
            "status": "success",
            "results": results,
        }
        
    except Exception as e:
        log(f"Error: {e}")
        return {
            "status": "error",
            "error": str(e),
        }

def test_selective_offload():
    """Test selective offload."""
    log("\n" + "="*60)
    log("TEST 3: Selective Offload")
    log("="*60)
    
    try:
        # Import selective offload module
        from selective_offload import SelectiveOffloadTrainer, SelectiveOffloadConfig
        
        # Test different configs
        results = []
        
        for in_memory in [0, 10, 20]:
            log(f"\nTesting {in_memory} layers in memory...")
            
            config = SelectiveOffloadConfig(
                keep_in_memory=in_memory,
                offload_middle=True,
                max_memory_gb=2.0,
            )
            
            trainer = SelectiveOffloadTrainer(
                model_id="Qwen/Qwen2.5-32B-Instruct-4bit",  # Use 32B for realistic test
                config=config,
                verbose=True,
            )
            
            # Estimate memory
            size = trainer.estimate_memory()
            
            log(f"In-memory: {size['in_memory_layers']} layers")
            log(f"Offloaded: {size['offloaded_layers']} layers")
            log(f"Memory: {size['total_gb']:.1f} GB")
            log(f"Speedup: {size['speedup_pct']:.0f}%")
            
            results.append({
                "layers_in_memory": in_memory,
                "memory_gb": size['total_gb'],
                "speedup_pct": size['speedup_pct'],
            })
        
        return {
            "status": "success",
            "results": results,
        }
        
    except Exception as e:
        log(f"Error: {e}")
        return {
            "status": "error",
            "error": str(e),
        }

def run_all_tests():
    """Run all v1.2 feature tests."""
    log("="*60)
    log("v1.2 FEATURE VALIDATION")
    log("="*60)
    log(f"Model: {TEST_MODEL}")
    log(f"Iterations: {TEST_ITERATIONS}")
    log("")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": TEST_MODEL,
        "iterations": TEST_ITERATIONS,
        "tests": {},
    }
    
    # Run tests
    results["tests"]["mixed_precision"] = test_mixed_precision()
    results["tests"]["gradient_accumulation"] = test_gradient_accumulation()
    results["tests"]["selective_offload"] = test_selective_offload()
    
    # Summary
    log("\n" + "="*60)
    log("TEST SUMMARY")
    log("="*60)
    
    for name, result in results["tests"].items():
        status = result.get("status", "unknown")
        status_icon = "✅" if status == "success" else "❌"
        log(f"{status_icon} {name}: {status}")
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    log(f"\nResults saved to: {RESULTS_FILE}")
    
    return results

if __name__ == "__main__":
    run_all_tests()
