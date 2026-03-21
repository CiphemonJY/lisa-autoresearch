"""Utilities Module - Benchmark, Mixed Precision, Production"""

__all__ = [
    "BenchmarkSuite",
]

def get_benchmark():
    from .benchmark import BenchmarkSuite
    return BenchmarkSuite
