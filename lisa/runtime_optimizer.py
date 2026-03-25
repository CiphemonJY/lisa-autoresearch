#!/usr/bin/env python3
"""
Runtime Optimizer - Monitors training and auto-adjusts configuration
for best performance based on actual metrics.
"""

import os
import sys
import time
import json
import threading
import logging
import statistics
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from collections import deque
from datetime import datetime


logger = logging.getLogger("runtime-optimizer")


@dataclass
class TrainingMetrics:
    """Snapshot of training metrics."""
    timestamp: float
    step: int
    loss: float
    throughput_toks_per_sec: float
    memory_used_gb: float
    memory_available_gb: float
    disk_io_wait_pct: float
    gpu_util_pct: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    batch_size: int = 0
    layer_groups: int = 0
    step_time_ms: float = 0
    forward_time_ms: float = 0
    backward_time_ms: float = 0


@dataclass
class ConfigChange:
    """Recommended configuration change."""
    reason: str
    change_type: str  # 'increase', 'decrease', 'adjust'
    parameter: str
    old_value: Any
    new_value: Any
    priority: str = "low"  # 'low', 'medium', 'high'
    estimated_impact_pct: float = 0


@dataclass 
class RuntimeConfig:
    """Current runtime configuration."""
    layer_groups: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_memory_gb: float
    dataloader_workers: int
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class RuntimeOptimizer:
    """
    Monitors training metrics and auto-adjusts configuration.
    
    Uses a feedback loop to optimize for:
    - Maximum throughput
    - Minimum memory pressure
    - Stable training
    """
    
    def __init__(
        self,
        initial_config: RuntimeConfig,
        target_throughput: Optional[float] = None,
        max_memory_pct: float = 0.8,
        adjustment_interval_steps: int = 50,
        smoothing_window: int = 10
    ):
        self.config = initial_config
        self.target_throughput = target_throughput
        self.max_memory_pct = max_memory_pct
        self.adjustment_interval = adjustment_interval_steps
        self.smoothing_window = smoothing_window
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=smoothing_window * 3)
        self.step_times: deque = deque(maxlen=smoothing_window)
        self.throughputs: deque = deque(maxlen=smoothing_window)
        self.memory_usage: deque = deque(maxlen=smoothing_window)
        
        # Tracking
        self.current_step = 0
        self.start_time = time.time()
        self.running = False
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_config_change: Optional[Callable[[ConfigChange], None]] = None
        
        # Thresholds
        self.THROUGHPUT_DROP_THRESHOLD = 0.15  # 15% drop triggers review
        self.MEMORY_HIGH_THRESHOLD = 0.85  # 85% memory usage
        self.MEMORY_CRITICAL_THRESHOLD = 0.95  # 95% = OOM risk
        self.DISK_IO_THRESHOLD = 0.30  # 30% disk wait
        
        # Adjustment rules
        self._adjustment_cooldown = 0
        self._last_adjustment_step = 0
        
        logger.info(f"RuntimeOptimizer initialized")
        logger.info(f"  Initial config: groups={initial_config.layer_groups}, "
                   f"batch={initial_config.batch_size}")
        logger.info(f"  Target throughput: {target_throughput or 'auto'}")
    
    def set_on_config_change(self, callback: Callable[[ConfigChange], None]) -> None:
        """Set callback for configuration changes."""
        self._on_config_change = callback
    
    def record_metrics(self, metrics: TrainingMetrics) -> None:
        """Record metrics snapshot and check for adjustments."""
        with self._lock:
            self.metrics_history.append(metrics)
            self.step_times.append(metrics.step_time_ms)
            self.throughputs.append(metrics.throughput_toks_per_sec)
            self.memory_usage.append(metrics.memory_used_gb / metrics.memory_available_gb)
            
            self.current_step = metrics.step
            
            # Check for adjustments
            if (self.current_step - self._last_adjustment_step >= self.adjustment_interval
                and self._adjustment_cooldown <= 0):
                changes = self._check_and_adjust()
                if changes:
                    self._last_adjustment_step = self.current_step
                    self._adjustment_cooldown = self.adjustment_interval * 2
    
    def _check_and_adjust(self) -> List[ConfigChange]:
        """Analyze metrics and determine if config should change."""
        if len(self.step_times) < self.smoothing_window:
            return []
        
        changes = []
        
        # Check 1: Memory pressure
        mem_changes = self._check_memory_pressure()
        changes.extend(mem_changes)
        
        # Check 2: Throughput
        throughput_changes = self._check_throughput()
        changes.extend(throughput_changes)
        
        # Check 3: Disk I/O
        io_changes = self._check_disk_io()
        changes.extend(io_changes)
        
        # Apply highest priority change
        if changes:
            # Sort by priority then impact
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            changes.sort(key=lambda c: (priority_order[c.priority], -c.estimated_impact_pct))
            
            best = changes[0]
            self._apply_change(best)
            return [best]
        
        return []
    
    def _check_memory_pressure(self) -> List[ConfigChange]:
        """Check for memory issues."""
        if not self.memory_usage:
            return []
        
        avg_mem = statistics.mean(self.memory_usage)
        changes = []
        
        if avg_mem > self.MEMORY_CRITICAL_THRESHOLD:
            # Critical - reduce immediately
            if self.config.layer_groups < 32:
                changes.append(ConfigChange(
                    reason=f"Memory critical: {avg_mem:.1%} > {self.MEMORY_CRITICAL_THRESHOLD:.1%}",
                    change_type="increase",
                    parameter="layer_groups",
                    old_value=self.config.layer_groups,
                    new_value=self.config.layer_groups + 2,
                    priority="high",
                    estimated_impact_pct=15
                ))
            elif self.config.batch_size > 1:
                changes.append(ConfigChange(
                    reason=f"Memory critical: {avg_mem:.1%} > {self.MEMORY_CRITICAL_THRESHOLD:.1%}",
                    change_type="decrease",
                    parameter="batch_size",
                    old_value=self.config.batch_size,
                    new_value=max(1, self.config.batch_size // 2),
                    priority="high",
                    estimated_impact_pct=20
                ))
        
        elif avg_mem > self.MEMORY_HIGH_THRESHOLD:
            if self.config.batch_size > 1:
                changes.append(ConfigChange(
                    reason=f"Memory high: {avg_mem:.1%} > {self.MEMORY_HIGH_THRESHOLD:.1%}",
                    change_type="decrease",
                    parameter="batch_size",
                    old_value=self.config.batch_size,
                    new_value=max(1, self.config.batch_size - 1),
                    priority="medium",
                    estimated_impact_pct=10
                ))
        
        return changes
    
    def _check_throughput(self) -> List[ConfigChange]:
        """Check throughput and optimize."""
        if len(self.throughputs) < self.smoothing_window:
            return []
        
        avg_throughput = statistics.mean(self.throughputs)
        changes = []
        
        # If we have a target and we're below it
        if self.target_throughput:
            ratio = avg_throughput / self.target_throughput
            if ratio < 0.8:
                # Significant underperformance
                if self.config.layer_groups > 1:
                    changes.append(ConfigChange(
                        reason=f"Throughput {avg_throughput:.0f} < target {self.target_throughput:.0f}",
                        change_type="decrease",
                        parameter="layer_groups",
                        old_value=self.config.layer_groups,
                        new_value=max(1, self.config.layer_groups - 1),
                        priority="medium",
                        estimated_impact_pct=15
                    ))
        
        # Check for throughput drop
        if len(self.throughputs) >= self.smoothing_window * 2:
            recent = statistics.mean(list(self.throughputs)[-self.smoothing_window:])
            older = statistics.mean(list(self.throughputs)[:-self.smoothing_window])
            if older > 0 and recent < older * (1 - self.THROUGHPUT_DROP_THRESHOLD):
                if self.config.layer_groups > 2:
                    changes.append(ConfigChange(
                        reason=f"Throughput dropped {((1 - recent/older) * 100):.0f}%",
                        change_type="decrease",
                        parameter="layer_groups",
                        old_value=self.config.layer_groups,
                        new_value=self.config.layer_groups - 1,
                        priority="medium",
                        estimated_impact_pct=10
                    ))
        
        return changes
    
    def _check_disk_io(self) -> List[ConfigChange]:
        """Check disk I/O wait time."""
        # This would need metrics.disk_io_wait_pct
        # Simplified for now
        return []
    
    def _apply_change(self, change: ConfigChange) -> None:
        """Apply a configuration change."""
        logger.info(f"⚙️  Applying config change: {change.parameter}")
        logger.info(f"   {change.reason}")
        logger.info(f"   {change.parameter}: {change.old_value} → {change.new_value}")
        
        old_config = self.config.to_dict()
        
        if change.parameter == "layer_groups":
            self.config.layer_groups = change.new_value
        elif change.parameter == "batch_size":
            self.config.batch_size = change.new_value
        elif change.parameter == "gradient_accumulation_steps":
            self.config.gradient_accumulation_steps = change.new_value
        elif change.parameter == "learning_rate":
            self.config.learning_rate = change.new_value
        
        self._adjustment_cooldown = self.adjustment_interval * 2
        
        if self._on_config_change:
            try:
                self._on_config_change(change)
            except Exception as e:
                logger.error(f"Config change callback failed: {e}")
    
    def get_current_config(self) -> RuntimeConfig:
        """Get current runtime configuration."""
        with self._lock:
            return self.config
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        with self._lock:
            stats = {
                'step': self.current_step,
                'uptime_sec': time.time() - self.start_time,
                'config': self.config.to_dict(),
            }
            
            if self.step_times:
                stats['avg_step_time_ms'] = statistics.mean(self.step_times)
                stats['recent_step_time_ms'] = statistics.mean(list(self.step_times)[-5:])
            if self.throughputs:
                stats['avg_throughput'] = statistics.mean(self.throughputs)
            if self.memory_usage:
                stats['avg_memory_pct'] = statistics.mean(self.memory_usage) * 100
            
            return stats
    
    def print_stats(self) -> None:
        """Print current stats."""
        stats = self.get_stats()
        print(f"\n📊 Runtime Stats (step {stats['step']}):")
        print(f"   Config: groups={stats['config']['layer_groups']}, "
              f"batch={stats['config']['batch_size']}")
        if 'avg_step_time_ms' in stats:
            print(f"   Step time: {stats['avg_step_time_ms']:.0f}ms")
        if 'avg_throughput' in stats:
            print(f"   Throughput: {stats['avg_throughput']:.1f} tok/s")
        if 'avg_memory_pct' in stats:
            print(f"   Memory: {stats['avg_memory_pct']:.1f}%")
    
    def save_metrics(self, path: str) -> None:
        """Save metrics history to JSON."""
        with open(path, 'w') as f:
            json.dump({
                'config': self.config.to_dict(),
                'stats': self.get_stats(),
                'metrics': [asdict(m) for m in self.metrics_history]
            }, f, indent=2)
        logger.info(f"Saved metrics to: {path}")


class MultiDeviceOptimizer:
    """
    Coordinates optimization across multiple devices.
    Balances workload based on device capabilities.
    """
    
    def __init__(self):
        self.device_optimizers: Dict[str, RuntimeOptimizer] = {}
        self.sync_interval = 100  # Sync configs every N steps
        self.current_step = 0
    
    def add_device(self, device_id: str, initial_config: RuntimeConfig) -> RuntimeOptimizer:
        """Add a device with its initial config."""
        opt = RuntimeOptimizer(initial_config)
        self.device_optimizers[device_id] = opt
        logger.info(f"Added device optimizer for: {device_id}")
        return opt
    
    def record_device_metrics(self, device_id: str, metrics: TrainingMetrics) -> None:
        """Record metrics for a specific device."""
        if device_id in self.device_optimizers:
            self.device_optimizers[device_id].record_metrics(metrics)
            self.current_step = metrics.step
    
    def should_rebalance(self) -> bool:
        """Check if we should rebalance workloads."""
        if not self.device_optimizers:
            return False
        
        # Check if throughputs vary significantly
        throughputs = []
        for opt in self.device_optimizers.values():
            if opt.throughputs:
                throughputs.append(statistics.mean(opt.throughputs))
        
        if len(throughputs) < 2:
            return False
        
        avg = statistics.mean(throughputs)
        max_diff = max(abs(t - avg) / avg for t in throughputs)
        
        return max_diff > 0.3  # 30% variation
    
    def rebalance_configs(self) -> Dict[str, ConfigChange]:
        """Rebalance configs across devices to minimize total time."""
        if not self.should_rebalance():
            return {}
        
        # Find fastest and slowest devices
        device_times = {}
        for device_id, opt in self.device_optimizers.items():
            if opt.throughputs:
                device_times[device_id] = 1.0 / statistics.mean(opt.throughputs)  # Inverse = time
        
        if len(device_times) < 2:
            return {}
        
        # Give slower devices fewer groups, faster devices more
        changes = {}
        
        # Sort by speed
        sorted_devices = sorted(device_times.items(), key=lambda x: x[1])
        slowest = sorted_devices[0][0]
        fastest = sorted_devices[-1][0]
        
        slow_opt = self.device_optimizers[slowest]
        fast_opt = self.device_optimizers[fastest]
        
        # If slowest has more groups, reduce and give to fastest
        if slow_opt.config.layer_groups > fast_opt.config.layer_groups:
            new_slow = max(1, slow_opt.config.layer_groups - 1)
            new_fast = fast_opt.config.layer_groups + 1
            
            changes[slowest] = ConfigChange(
                reason="Rebalancing: slowest device reduced groups",
                change_type="decrease",
                parameter="layer_groups",
                old_value=slow_opt.config.layer_groups,
                new_value=new_slow,
                priority="medium",
                estimated_impact_pct=10
            )
            
            changes[fastest] = ConfigChange(
                reason="Rebalancing: fastest device increased groups",
                change_type="increase",
                parameter="layer_groups",
                old_value=fast_opt.config.layer_groups,
                new_value=new_fast,
                priority="low",
                estimated_impact_pct=5
            )
        
        return changes
    
    def print_summary(self) -> None:
        """Print summary of all devices."""
        print(f"\n{'='*60}")
        print("MULTI-DEVICE OPTIMIZATION SUMMARY")
        print(f"{'='*60}")
        
        for device_id, opt in self.device_optimizers.items():
            stats = opt.get_stats()
            print(f"\n📱 {device_id}:")
            print(f"   Config: groups={stats['config']['layer_groups']}, "
                  f"batch={stats['config']['batch_size']}")
            if 'avg_throughput' in stats:
                print(f"   Throughput: {stats['avg_throughput']:.1f} tok/s")
            if 'avg_memory_pct' in stats:
                print(f"   Memory: {stats['avg_memory_pct']:.1f}%")


def main():
    # Simple test/demo
    print("Runtime Optimizer - Demo")
    print("=" * 50)
    
    initial = RuntimeConfig(
        layer_groups=4,
        batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_memory_gb=8.0,
        dataloader_workers=2
    )
    
    optimizer = RuntimeOptimizer(initial, target_throughput=50.0)
    
    # Simulate some training steps
    import random
    for step in range(1, 201):
        metrics = TrainingMetrics(
            timestamp=time.time(),
            step=step,
            loss=2.0 - step * 0.01 + random.uniform(-0.1, 0.1),
            throughput_toks_per_sec=50 + random.uniform(-5, 5),
            memory_used_gb=5.0 + random.uniform(-0.5, 0.5),
            memory_available_gb=8.0,
            disk_io_wait_pct=0.1,
            batch_size=4,
            layer_groups=4,
            step_time_ms=100 + random.uniform(-10, 10),
            forward_time_ms=60,
            backward_time_ms=40
        )
        optimizer.record_metrics(metrics)
    
    optimizer.print_stats()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
