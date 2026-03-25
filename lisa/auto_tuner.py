#!/usr/bin/env python3
"""
Auto-Tuner - Automatically optimizes federated learning configuration
Based on device profiles, finds best layer groups, batch size, etc.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger("auto-tuner")


@dataclass
class TuningConfig:
    """Auto-tuned configuration for a device."""
    device_id: str
    model_name: str
    
    # Layer configuration
    layer_groups: int
    groups_per_pass: int  # How many groups to process before saving
    
    # Batch configuration
    batch_size: int
    gradient_accumulation_steps: int
    
    # Memory management
    offload_to_disk: bool
    disk_cache_dir: str
    max_memory_gb: float
    
    # Performance tuning
    dataloader_workers: int
    pin_memory: bool
    use_mixed_precision: bool
    
    # Learning rate (may be auto-adjusted)
    learning_rate: float
    
    # Meta
    estimated_round_time_sec: float
    auto_tuned: bool = True
    
    def to_dict(self):
        d = asdict(self)
        d['auto_tuned'] = self.auto_tuned
        return d


class FederatedAutoTuner:
    """
    Automatically tunes federated learning for best performance.
    
    Uses device profiles and benchmark data to find optimal configuration.
    """
    
    # Model size estimates (GB at fp16)
    MODEL_SIZES = {
        "1b": 2.0,
        "3b": 6.0,
        "7b": 14.0,
        "14b": 28.0,
        "32b": 64.0,
        "60b": 120.0,
    }
    
    # Model sizes at 4-bit quantization
    MODEL_SIZES_4BIT = {
        "1b": 0.5,
        "3b": 1.5,
        "7b": 3.5,
        "14b": 7.0,
        "32b": 16.0,
        "60b": 30.0,
    }
    
    def __init__(self):
        self.profiles: Dict[str, dict] = {}
        self.best_configs: Dict[str, TuningConfig] = {}
    
    def add_device(self, profile) -> None:
        """Add a device profile to the tuner."""
        # Handle both dict and DeviceProfile object
        if hasattr(profile, 'to_dict'):
            profile = profile.to_dict()
        self.profiles[profile['device_id']] = profile
        logger.info(f"Added device: {profile['device_id']}")
        logger.info(f"  RAM: {profile['ram_available_gb']:.1f}GB, SSD: {profile['ssd_read_speed_gbps']:.1f}GB/s")
        logger.info(f"  GPU: {profile['gpu_available']}, MLX: {profile['mlx_available']}")
    
    def configure_for_model(
        self,
        model_name: str,
        quantization: str = "4bit",
        target_rounds: int = 10
    ) -> Dict[str, TuningConfig]:
        """
        Generate optimized configuration for all devices for a given model.
        
        Returns dict of device_id -> TuningConfig
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"AUTO-TUNING for {model_name} ({quantization})")
        logger.info(f"{'='*60}")
        
        # Determine model size
        model_size_gb = self._estimate_model_size(model_name, quantization)
        logger.info(f"Model size: {model_size_gb:.1f}GB")
        
        configs = {}
        
        for device_id, profile in self.profiles.items():
            logger.info(f"\n📱 Tuning for {device_id}...")
            config = self._tune_device(profile, model_name, model_size_gb)
            configs[device_id] = config
            
            logger.info(f"   → Groups: {config.layer_groups}, Batch: {config.batch_size}")
            logger.info(f"   → Est. round time: {config.estimated_round_time_sec / 60:.1f} min")
        
        self.best_configs = configs
        return configs
    
    def _estimate_model_size(self, model_name: str, quantization: str) -> float:
        """Estimate model size in GB."""
        name_lower = model_name.lower()
        
        # Check for known models
        if "0.5b" in name_lower or "0_5b" in name_lower:
            size = 1.0 if quantization == "4bit" else 2.0
        elif "1b" in name_lower or "1_1b" in name_lower:
            size = 0.5 if quantization == "4bit" else 2.0
        elif "3b" in name_lower or "3_0b" in name_lower:
            size = 1.5 if quantization == "4bit" else 6.0
        elif "7b" in name_lower or "7_0b" in name_lower:
            size = 3.5 if quantization == "4bit" else 14.0
        elif "14b" in name_lower or "14_0b" in name_lower:
            size = 7.0 if quantization == "4bit" else 28.0
        elif "32b" in name_lower or "32_0b" in name_lower:
            size = 16.0 if quantization == "4bit" else 64.0
        elif "60b" in name_lower or "60_0b" in name_lower:
            size = 30.0 if quantization == "4bit" else 120.0
        else:
            # Default: try to parse number
            logger.warning(f"Unknown model size for {model_name}, assuming 7B")
            size = 3.5 if quantization == "4bit" else 14.0
        
        return size
    
    def _tune_device(self, profile: dict, model_name: str, model_size_gb: float) -> TuningConfig:
        """Auto-tune configuration for a specific device."""
        
        device_id = profile['device_id']
        ram_gb = profile['ram_available_gb']
        ssd_speed = profile['ssd_read_speed_gbps']
        gpu_available = profile['gpu_available']
        gpu_mem_gb = profile['gpu_memory_gb']
        mlx_available = profile['mlx_available']
        cpu_cores = profile['cpu_cores']
        
        # Determine if we need disk offloading
        needs_offload = model_size_gb > ram_gb * 0.7
        
        if mlx_available:
            # Apple Silicon - best performance, unified memory
            return self._tune_mlx_device(profile, model_name, model_size_gb)
        elif gpu_available and gpu_mem_gb >= model_size_gb:
            # NVIDIA GPU with enough VRAM
            return self._tune_gpu_device(profile, model_name, model_size_gb)
        elif needs_offload:
            # Need disk offloading
            return self._tune_offload_device(profile, model_name, model_size_gb)
        else:
            # Fits in RAM with room for batch
            return self._tune_ram_device(profile, model_name, model_size_gb)
    
    def _tune_mlx_device(self, profile: dict, model_name: str, model_size_gb: float) -> TuningConfig:
        """Tune for Apple Silicon with MLX."""
        
        device_id = profile['device_id']
        ram_gb = profile['ram_available_gb']
        
        # MLX is fast - can load more groups
        layer_groups = max(2, int(model_size_gb / 4))  # 4GB per group
        layer_groups = min(layer_groups, 8)  # Cap at 8
        
        # Batch size based on RAM
        available_for_batch = ram_gb - (model_size_gb / layer_groups) - 1.0  # 1GB overhead
        batch_size = max(1, int(available_for_batch / 0.001))  # ~1MB per sample
        batch_size = min(batch_size, 16)  # Cap
        
        # MLX doesn't need gradient accumulation for throughput
        grad_accum = 1
        
        # Round time estimate
        # MLX is fast: ~50 tokens/sec, 200 samples/round = 4 sec inference
        est_time = 300 / layer_groups + 10  # sec
        
        return TuningConfig(
            device_id=device_id,
            model_name=model_name,
            layer_groups=layer_groups,
            groups_per_pass=layer_groups,  # All at once
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            offload_to_disk=False,
            disk_cache_dir="",
            max_memory_gb=ram_gb * 0.8,
            dataloader_workers=min(profile['cpu_cores'], 4),
            pin_memory=False,
            use_mixed_precision=False,
            learning_rate=2e-4,
            estimated_round_time_sec=est_time
        )
    
    def _tune_gpu_device(self, profile: dict, model_name: str, model_size_gb: float) -> TuningConfig:
        """Tune for NVIDIA GPU."""
        
        device_id = profile['device_id']
        gpu_mem_gb = profile['gpu_memory_gb']
        cpu_cores = profile['cpu_cores']
        
        # GPU can handle larger batches
        layer_groups = max(1, int(model_size_gb / gpu_mem_gb))
        layer_groups = min(layer_groups, 4)
        
        # Batch size
        available = gpu_mem_gb - (model_size_gb / layer_groups) - 1.0
        batch_size = max(1, int(available / 0.002))  # 2MB per sample
        batch_size = min(batch_size, 32)
        
        est_time = 400 / layer_groups + 15
        
        return TuningConfig(
            device_id=device_id,
            model_name=model_name,
            layer_groups=layer_groups,
            groups_per_pass=layer_groups,
            batch_size=batch_size,
            gradient_accumulation_steps=1,
            offload_to_disk=False,
            disk_cache_dir="",
            max_memory_gb=gpu_mem_gb * 0.9,
            dataloader_workers=cpu_cores,
            pin_memory=True,
            use_mixed_precision=True,
            learning_rate=2e-4,
            estimated_round_time_sec=est_time
        )
    
    def _tune_offload_device(self, profile: dict, model_name: str, model_size_gb: float) -> TuningConfig:
        """Tune for disk offloading (low RAM or large model)."""
        
        device_id = profile['device_id']
        ram_gb = profile['ram_available_gb']
        ssd_speed = profile['ssd_read_speed_gbps']
        ssd_free_gb = profile['ssd_available_gb']
        cpu_cores = profile['cpu_cores']
        
        # Calculate layer groups to keep RAM usage low
        # Peak RAM = model_size / groups + activations + overhead
        overhead_gb = 1.0  # System + activations
        available_for_model = ram_gb * 0.5 - overhead_gb  # Conservative 50%
        
        min_groups = max(1, int(model_size_gb / available_for_model))
        
        # More groups = less RAM per group but more disk I/O
        # Tradeoff: groups vs I/O time
        # Each group loads model_size/groups from SSD at ssd_speed GB/s
        # Plus compute time
        best_groups = min_groups
        best_score = float('inf')
        
        # Binary search-like evaluation
        for groups in range(max(min_groups, 2), min(min_groups * 4, 32) + 1, 2):
            # Time to load all groups: groups * (model_size/groups) / ssd_speed = model_size / ssd_speed
            # This is constant! The difference is in how many fits in RAM for activations
            
            # But with gradient accumulation, more groups = more accumulation overhead
            score = groups * 0.5  # Overhead factor
            
            if score < best_score:
                best_score = score
                best_groups = groups
        
        layer_groups = best_groups
        
        # Batch size (limited by RAM)
        available = ram_gb * 0.4  # Very conservative
        batch_size = max(1, int(available / 0.001))
        batch_size = min(batch_size, 4)  # Small batches for offload
        
        # Gradient accumulation to compensate for small batches
        grad_accum = max(1, 8 // batch_size)
        
        # Disk cache
        cache_dir = f"/tmp/lisa_offload_{device_id}"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Estimate time
        # Load all groups: model_size / ssd_speed (constant regardless of groups)
        load_time = model_size_gb / ssd_speed
        # Plus compute
        compute_time = layer_groups * 5  # 5 sec per group estimate
        est_time = load_time + compute_time + 60
        
        return TuningConfig(
            device_id=device_id,
            model_name=model_name,
            layer_groups=layer_groups,
            groups_per_pass=1,  # Load one at a time
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            offload_to_disk=True,
            disk_cache_dir=cache_dir,
            max_memory_gb=ram_gb * 0.5,
            dataloader_workers=min(cpu_cores, 2),  # Limited by disk
            pin_memory=False,
            use_mixed_precision=False,
            learning_rate=1e-4,  # Lower for offload stability
            estimated_round_time_sec=est_time
        )
    
    def _tune_ram_device(self, profile: dict, model_name: str, model_size_gb: float) -> TuningConfig:
        """Tune for device with enough RAM but no GPU."""
        
        device_id = profile['device_id']
        ram_gb = profile['ram_available_gb']
        cpu_cores = profile['cpu_cores']
        
        layer_groups = max(1, int(model_size_gb / (ram_gb * 0.3)))
        layer_groups = min(layer_groups, 8)
        
        available = ram_gb * 0.5
        batch_size = max(1, int(available / 0.001))
        batch_size = min(batch_size, 8)
        
        est_time = 600 / layer_groups + 20
        
        return TuningConfig(
            device_id=device_id,
            model_name=model_name,
            layer_groups=layer_groups,
            groups_per_pass=layer_groups,
            batch_size=batch_size,
            gradient_accumulation_steps=1,
            offload_to_disk=False,
            disk_cache_dir="",
            max_memory_gb=ram_gb * 0.6,
            dataloader_workers=cpu_cores,
            pin_memory=False,
            use_mixed_precision=False,
            learning_rate=2e-4,
            estimated_round_time_sec=est_time
        )
    
    def estimate_total_round_time(self, configs: Dict[str, TuningConfig]) -> float:
        """Estimate total round time across all devices."""
        if not configs:
            return 0
        
        # Round time is determined by slowest device
        max_time = max(c.estimated_round_time_sec for c in configs.values())
        
        # But we can parallelize
        return max_time * 1.1  # 10% overhead
    
    def print_summary(self, configs: Dict[str, TuningConfig]) -> None:
        """Print a summary of the auto-tuned configuration."""
        print(f"\n{'='*60}")
        print("AUTO-TUNED CONFIGURATION SUMMARY")
        print(f"{'='*60}")
        
        total_round = self.estimate_total_round_time(configs)
        print(f"\n📊 Estimated total round time: {total_round / 60:.1f} min")
        print(f"📊 Number of devices: {len(configs)}")
        
        print("\n📱 Per-Device Configuration:")
        for device_id, config in configs.items():
            print(f"\n  {device_id}:")
            print(f"    Model: {config.model_name}")
            print(f"    Layer groups: {config.layer_groups} (offload: {config.offload_to_disk})")
            print(f"    Batch size: {config.batch_size}, Grad accum: {config.gradient_accumulation_steps}")
            print(f"    Est. round time: {config.estimated_round_time_sec / 60:.1f} min")
            print(f"    LR: {config.learning_rate}")
        
        print(f"\n{'='*60}\n")
    
    def save_configs(self, path: str) -> None:
        """Save all configs to JSON."""
        data = {k: v.to_dict() for k, v in self.best_configs.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved configs to: {path}")
    
    def load_configs(self, path: str) -> Dict[str, TuningConfig]:
        """Load configs from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        configs = {}
        for device_id, d in data.items():
            configs[device_id] = TuningConfig(**d)
        
        logger.info(f"Loaded {len(configs)} configs from: {path}")
        return configs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-tune federated learning")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--quantization", default="4bit", choices=["fp16", "4bit", "8bit"])
    parser.add_argument("--devices", nargs="+", help="Device IDs to tune for")
    parser.add_argument("--load-profiles", help="Load profiles from JSON")
    parser.add_argument("--save-configs", help="Save configs to JSON")
    parser.add_argument("--summary", action="store_true", help="Print summary")
    args = parser.parse_args()
    
    tuner = FederatedAutoTuner()
    
    # Load profiles if provided
    if args.load_profiles:
        with open(args.load_profiles, 'r') as f:
            data = json.load(f)
        # Handle both formats: {"device_id": {...}} or {...} (direct)
        if 'device_id' in data:
            # Direct format
            tuner.add_device(data)
        else:
            # Dict format
            for device_id, profile in data.items():
                tuner.add_device(profile)
    
    # Or use a default
    if not tuner.profiles:
        # Add current device
        from lisa.device_profiler import DeviceProfiler
        profiler = DeviceProfiler()
        profile = profiler.profile()
        tuner.add_device(profile.to_dict())
    
    # Tune
    configs = tuner.configure_for_model(args.model, args.quantization)
    
    if args.summary:
        tuner.print_summary(configs)
    
    if args.save_configs:
        tuner.save_configs(args.save_configs)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
