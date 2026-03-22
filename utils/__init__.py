"""Utilities Module - Benchmark, Mixed Precision, Production, HIPAA Audit"""

__all__ = [
    "BenchmarkSuite",
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
]

def get_benchmark():
    from .benchmark import BenchmarkSuite
    return BenchmarkSuite

from .audit_logger import AuditLogger, AuditEvent, AuditEventType
