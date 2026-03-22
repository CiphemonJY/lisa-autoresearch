#!/usr/bin/env python3
"""
Generate a HIPAA compliance audit report from audit logs.

Usage:
    python scripts/generate_audit_report.py --start 2026-01-01 --end 2026-03-21
    python scripts/generate_audit_report.py --start 2026-01-01 --end 2026-03-21 --output compliance_report.json
    python scripts/generate_audit_report.py --verify-only  # just verify chain integrity
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.audit_logger import AuditLogger


def main():
    parser = argparse.ArgumentParser(description="Generate HIPAA compliance audit report")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: 90 days ago)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--audit-dir",
        type=str,
        default="audit_logs",
        help="Directory containing audit log files",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify chain integrity, don't generate full report",
    )
    parser.add_argument(
        "--decrypt",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Decrypt and print events for a specific date (for HIPAA officer review)",
    )
    args = parser.parse_args()

    logger = AuditLogger(audit_dir=args.audit_dir)

    # Decrypt-only mode: for HIPAA officers
    if args.decrypt:
        events = logger.decrypt_log_file(args.decrypt)
        print(f"=== Audit Log for {args.decrypt} ({len(events)} events) ===")
        print("(CONTAINS PHI — ACCESS RESTRICTED TO HIPAA COMPLIANCE OFFICERS)")
        print("-" * 60)
        for event in events:
            print(json.dumps(event, indent=2, default=str))
            print("-" * 40)
        return

    # Verify-only mode
    if args.verify_only:
        result = logger.verify_all_chains(args.start, args.end)
        print(json.dumps(result, indent=2, default=str))
        if not result.get("valid"):
            print("CHAIN INTEGRITY VIOLATION DETECTED", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"All chains valid. Total entries verified: {result['total_entries']}")
        return

    # Full compliance report
    report = logger.generate_compliance_report(
        start_date=args.start,
        end_date=args.end,
    )

    output = json.dumps(report, indent=2, default=str)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Compliance report written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
