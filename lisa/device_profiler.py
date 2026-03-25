#!/usr/bin/env python3
"""
Device Profiler - Auto-tuning Federated Learning
Profiles device capabilities and returns optimized config suggestions.
"""

import os
import sys
import time
import shutil
import tempfile
import platform
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class DeviceProfile:
    """Complete device capability profile."""
    device_id: str
    hostname: str
    platform: str  # darwin, linux, windows
    
    # Memory
    ram_total_gb: float
    ram_available_gb: float
    swap_total_gb: float
    
    # Storage
    ssd_available_gb: float
    ssd_read_speed_gbps: float  # GB/s
    ssd_write_speed_gbps: float
    
    # Compute
    cpu_cores: int
    gpu_available: bool
    gpu_memory_gb: float
    gpu_name: str
    
    # Framework advantages
    mlx_available: bool  # Apple Silicon
    cuda_available: bool
    
    # Calculated
    recommended_layer_groups: int
    recommended_batch_size: int
    estimated_throughput_toks_per_sec: float
    
    def to_dict(self):
        return asdict(self)
    
    def summary(self) -> str:
        lines = [
            f"  Platform: {self.platform}",
            f"  RAM: {self.ram_available_gb:.1f}GB / {self.ram_total_gb:.1f}GB",
            f"  SSD: {self.ssd_available_gb:.0f}GB free, {self.ssd_read_speed_gbps:.1f} GB/s read",
            f"  CPU: {self.cpu_cores} cores",
        ]
        if self.gpu_available:
            lines.append(f"  GPU: {self.gpu_name} ({self.gpu_memory_gb:.1f}GB)")
        if self.mlx_available:
            lines.append(f"  MLX: Available (Apple Silicon - fast!)")
        lines.append(f"  → Recommended: {self.recommended_layer_groups} groups, batch={self.recommended_batch_size}")
        return "\n".join(lines)


class DeviceProfiler:
    """Profiles a device for federated learning capabilities."""
    
    PROFILE_VERSION = "1.0"
    
    def __init__(self, device_id: Optional[str] = None):
        self.device_id = device_id or platform.node()
    
    def profile(self, temp_dir: Optional[str] = None) -> DeviceProfile:
        """Run all benchmarks and return complete profile."""
        print(f"🔍 Profiling device: {self.device_id}")
        print("=" * 50)
        
        # Use provided temp dir or create one
        use_temp = tempfile.mkdtemp(prefix="lisa_profile_")
        temp_file = os.path.join(use_temp, "speed_test.bin")
        
        try:
            ram = self._profile_memory()
            ssd = self._profile_storage(use_temp)
            compute = self._profile_compute()
            frameworks = self._profile_frameworks()
            
            # Calculate recommendations
            groups, batch, throughput = self._calculate_recommendations(
                ram['available_gb'],
                ssd['read_speed_gbps'],
                compute['gpu_available'],
                compute.get('gpu_memory_gb', 0),
                frameworks['mlx_available'],
                frameworks['cuda_available']
            )
            
            profile = DeviceProfile(
                device_id=self.device_id,
                hostname=platform.node(),
                platform=platform.system().lower(),
                ram_total_gb=ram['total_gb'],
                ram_available_gb=ram['available_gb'],
                swap_total_gb=ram['swap_gb'],
                ssd_available_gb=ssd['available_gb'],
                ssd_read_speed_gbps=ssd['read_speed_gbps'],
                ssd_write_speed_gbps=ssd['write_speed_gbps'],
                cpu_cores=compute['cpu_cores'],
                gpu_available=compute['gpu_available'],
                gpu_memory_gb=compute.get('gpu_memory_gb', 0),
                gpu_name=compute.get('gpu_name', 'Unknown'),
                mlx_available=frameworks['mlx_available'],
                cuda_available=frameworks['cuda_available'],
                recommended_layer_groups=groups,
                recommended_batch_size=batch,
                estimated_throughput_toks_per_sec=throughput
            )
            
            print("=" * 50)
            print(profile.summary())
            return profile
            
        finally:
            # Cleanup temp dir
            if not temp_dir:
                shutil.rmtree(use_temp, ignore_errors=True)
    
    def _profile_memory(self) -> Dict:
        """Profile RAM and swap."""
        print("  📊 Profiling memory...")
        
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            total_bytes = int(result.stdout.strip())
            total_gb = total_bytes / 1e9
            
            # Get available memory
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True, text=True
            )
            # Parse vm_stat output
            lines = result.stdout.strip().split('\n')
            free = 0
            inactive = 0
            for line in lines:
                if 'Pages free:' in line:
                    free = int(line.split(':')[1].strip().rstrip('.')) * 4096
                if 'Pages inactive:' in line:
                    inactive = int(line.split(':')[1].strip().rstrip('.')) * 4096
            available_gb = (free + inactive) / 1e9
            
        else:  # Linux (including Jetson)
            # Read /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            
            mem = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    val = int(parts[1]) * 1024  # Convert KB to bytes
                    mem[key] = val
            
            total_gb = mem.get('MemTotal', 0) / 1e9
            free_gb = mem.get('MemFree', 0) / 1e9
            buffers_gb = mem.get('Buffers', 0) / 1e9
            cached_gb = mem.get('Cached', 0) / 1e9
            # Available = free + buffers + cached (best estimate)
            available_gb = free_gb + buffers_gb + cached_gb
            
            # Swap
            swap_total_gb = mem.get('SwapTotal', 0) / 1e9
        
        return {
            'total_gb': total_gb,
            'available_gb': available_gb,
            'swap_gb': swap_total_gb if platform.system() != "Darwin" else 0
        }
    
    def _profile_storage(self, temp_dir: str) -> Dict:
        """Profile SSD/storage speed."""
        print("  💾 Profiling storage...")
        
        # Use temp dir for testing
        os.makedirs(temp_dir, exist_ok=True)
        test_file = os.path.join(temp_dir, "speed_test.bin")
        test_size_mb = 256  # 256MB test
        
        # Check available space (use parent dir since file doesn't exist yet)
        stat = shutil.disk_usage(temp_dir)
        available_gb = stat.total / 1e9
        
        # Benchmark read/write speed
        test_size_bytes = test_size_mb * 1024 * 1024
        
        # Write benchmark
        print(f"    Writing {test_size_mb}MB...")
        data = os.urandom(test_size_bytes)
        start = time.perf_counter()
        with open(test_file, 'wb') as f:
            f.write(data)
        write_time = time.perf_counter() - start
        write_speed_gbps = (test_size_mb / 1024) / write_time
        
        # Read benchmark
        print(f"    Reading {test_size_mb}MB...")
        start = time.perf_counter()
        with open(test_file, 'rb') as f:
            f.read()
        read_time = time.perf_counter() - start
        read_speed_gbps = (test_size_mb / 1024) / read_time
        
        # Cleanup
        try:
            os.remove(test_file)
        except:
            pass
        
        print(f"    SSD: {available_gb:.0f}GB free, {read_speed_gbps:.2f} GB/s read, {write_speed_gbps:.2f} GB/s write")
        
        return {
            'available_gb': available_gb,
            'read_speed_gbps': read_speed_gbps,
            'write_speed_gbps': write_speed_gbps
        }
    
    def _profile_compute(self) -> Dict:
        """Profile CPU and GPU."""
        print("  ⚡ Profiling compute...")
        
        cpu_cores = os.cpu_count() or 4
        
        result = {
            'cpu_cores': cpu_cores,
            'gpu_available': False,
            'gpu_memory_gb': 0,
            'gpu_name': 'Unknown'
        }
        
        if platform.system() == "Darwin":
            # Check for Apple Silicon
            try:
                result_chip = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                )
                chip = result_chip.stdout.strip()
                if "Apple" in chip:
                    # Apple Silicon has unified memory
                    # Estimate ~50% of RAM for GPU
                    ram_result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True
                    )
                    total_bytes = int(ram_result.stdout.strip())
                    result['gpu_available'] = True
                    result['gpu_memory_gb'] = total_bytes / 1e9 * 0.5  # Estimate
                    result['gpu_name'] = chip
                    print(f"    GPU: {chip} (estimated {result['gpu_memory_gb']:.1f}GB unified)")
            except:
                pass
        
        elif platform.system() == "Linux":
            # Check for NVIDIA GPU
            try:
                result_gpu = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if result_gpu.returncode == 0:
                    gpu_info = result_gpu.stdout.strip().split(',')
                    result['gpu_available'] = True
                    result['gpu_name'] = gpu_info[0].strip()
                    result['gpu_memory_gb'] = float(gpu_info[1].strip().split()[0]) / 1024
                    print(f"    GPU: {result['gpu_name']} ({result['gpu_memory_gb']:.1f}GB)")
            except Exception as e:
                print(f"    GPU: Not available ({e})")
        
        return result
    
    def _profile_frameworks(self) -> Dict:
        """Check available ML frameworks."""
        print("  🔧 Profiling frameworks...")
        
        result = {
            'mlx_available': False,
            'cuda_available': False
        }
        
        # Check MLX (macOS)
        if platform.system() == "Darwin":
            try:
                import mlx.core as mlx
                result['mlx_available'] = True
                print("    MLX: Available (Apple Silicon acceleration)")
            except ImportError:
                print("    MLX: Not available")
        
        # Check CUDA
        try:
            import torch
            result['cuda_available'] = torch.cuda.is_available()
            if result['cuda_available']:
                print(f"    CUDA: Available (torch {torch.__version__})")
        except ImportError:
            print("    CUDA: PyTorch not available")
        
        return result
    
    def _calculate_recommendations(
        self,
        ram_gb: float,
        ssd_speed_gbps: float,
        gpu_available: bool,
        gpu_memory_gb: float,
        mlx_available: bool,
        cuda_available: bool
    ) -> tuple:
        """
        Calculate optimal layer groups and batch size based on device profile.
        
        Returns: (layer_groups, batch_size, estimated_throughput)
        """
        print("  🎯 Calculating optimal configuration...")
        
        # Base calculations
        # Model sizes at 4-bit (approximate)
        # 7B = 3.5GB, 14B = 7GB, 32B = 16GB, 60B = 30GB
        
        # For a typical 7B model at 4-bit
        model_size_gb = 3.5  # 7B at 4-bit
        activation_overhead_gb = 0.5
        optimizer_overhead_gb = 0.5
        
        # Effective bandwidth (SSD or GPU RAM)
        if gpu_available and gpu_memory_gb > 0 and not mlx_available:
            # NVIDIA GPU - can keep model in GPU RAM
            effective_bandwidth = 200  # GB/s (GPU RAM)
            memory_budget_gb = gpu_memory_gb * 0.8  # Leave headroom
        elif mlx_available:
            # Apple Silicon - unified memory
            effective_bandwidth = 100  # GB/s (unified memory)
            memory_budget_gb = ram_gb * 0.5  # Conservative
        else:
            # CPU + SSD offloading
            effective_bandwidth = ssd_speed_gbps
            memory_budget_gb = ram_gb * 0.6  # Leave room for OS
        
        # Calculate layer groups
        # Each group needs: model_size/groups + activations + optimizer
        # But with disk offload, we load one group at a time
        # So peak memory = model_size/groups * some_factor + activation_overhead
        
        # For offloading: peak ≈ (model_size / groups) + overhead
        # We want peak < memory_budget_gb
        # So: groups > model_size / (memory_budget_gb - overhead)
        
        min_groups_float = model_size_gb / max(memory_budget_gb - activation_overhead_gb - optimizer_overhead_gb, 0.5)
        min_groups = max(1, int(min_groups_float) + 1)  # Round up, minimum 1
        
        # But we want more groups for better memory efficiency
        # Optimal is often 4-16 depending on device
        if memory_budget_gb < 4:
            # Low RAM device (Jetson)
            recommended_groups = max(min_groups, 8)  # At least 8 for good granularity
        elif memory_budget_gb < 8:
            # Medium device
            recommended_groups = max(min_groups, 4)
        elif memory_budget_gb < 16:
            # Good device (Mac Mini)
            recommended_groups = max(min_groups, 2)
        else:
            # High-end device
            recommended_groups = max(min_groups, 1)
        
        # Cap at reasonable maximum
        recommended_groups = min(recommended_groups, 32)
        
        # Calculate batch size
        # Larger batches need more memory but are more efficient
        # Each batch item needs ~activation_size
        activation_per_sample_gb = 0.001  # 1MB per sample estimate
        
        available_for_batch = memory_budget_gb - (model_size_gb / recommended_groups) - activation_overhead_gb - optimizer_overhead_gb
        batch_size = max(1, int(available_for_batch / activation_per_sample_gb))
        
        # Cap batch size
        if memory_budget_gb < 4:
            batch_size = min(batch_size, 4)
        elif memory_budget_gb < 8:
            batch_size = min(batch_size, 8)
        else:
            batch_size = min(batch_size, 16)
        
        # Estimate throughput
        # Base throughput on compute capability
        if mlx_available:
            base_throughput = 50  # tokens/sec
        elif gpu_available:
            base_throughput = 40
        else:
            base_throughput = 5
        
        # Adjust for layer groups (more groups = more overhead)
        group_overhead_factor = 1.0 / (1.0 + 0.02 * recommended_groups)
        throughput = base_throughput * group_overhead_factor
        
        print(f"    → {recommended_groups} groups, batch={batch_size}, ~{throughput:.0f} tokens/sec")
        
        return recommended_groups, batch_size, throughput
    
    def save_profile(self, profile: DeviceProfile, path: str) -> None:
        """Save profile to JSON file."""
        with open(path, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)
        print(f"  💾 Profile saved to: {path}")
    
    @classmethod
    def load_profile(cls, path: str) -> DeviceProfile:
        """Load profile from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return DeviceProfile(**data)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Profile device for federated learning")
    parser.add_argument("--device-id", help="Device identifier")
    parser.add_argument("--output", "-o", help="Output profile path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()
    
    profiler = DeviceProfiler(args.device_id)
    profile = profiler.profile()
    
    if args.output:
        profiler.save_profile(profile, args.output)
    
    if not args.quiet:
        print("\n📋 Full Profile:")
        print(json.dumps(profile.to_dict(), indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
