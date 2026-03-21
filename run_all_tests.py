#!/usr/bin/env python3
"""
Nightly test runner - runs all modules, logs results, fixes errors.

Run directly: python run_all_tests.py
Or imported as a module by the cron agent.
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
os.chdir(str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_runner.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("test-runner")


def run_cmd(cmd, timeout=300):
    """Run a command, return (success, stdout+stderr)."""
    log.info(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=True if isinstance(cmd, str) else False,
        )
        elapsed = time.time() - t0
        combined = result.stdout + "\n" + result.stderr
        log.info(f"  Finished in {elapsed:.1f}s, exit={result.returncode}")
        return result.returncode == 0, combined, elapsed
    except subprocess.TimeoutExpired:
        log.error(f"  TIMEOUT after {timeout}s")
        return False, "TIMEOUT", timeout
    except Exception as e:
        log.error(f"  FAILED: {e}")
        return False, str(e), 0


def check_errors(output):
    """Extract error lines from output."""
    errors = []
    for line in output.split("\n"):
        if any(kw in line.lower() for kw in ["error", "exception", "traceback", "failed", "failures:", "error:"]):
            errors.append(line.strip())
    return errors[:20]  # First 20 error lines


def fix_error(output, cmd):
    """Attempt to fix common errors automatically."""
    fixes = []

    # Error: empty parameter list -> LoRA fallback was triggered
    if "empty parameter list" in output or "ValueError" in output:
        fixes.append("LORA_FALLBACK_NEEDED")

    # Error: can't import
    if "ImportError" in output or "ModuleNotFoundError" in output:
        for line in output.split("\n"):
            if "ModuleNotFoundError" in line or "ImportError" in line:
                pkg = line.split("'")[-2] if "'" in line else line.split()[-1]
                fixes.append(f"INSTALL:{pkg}")

    # Error: No module named 'mlx' (expected on Windows)
    if "No module named 'mlx'" in output:
        fixes.append("MLX_NOT_AVAILABLE")

    return fixes


RESULTS = []


def test(name, cmd, timeout=300, expect_fail=False):
    """Run a single test."""
    log.info("")
    log.info("=" * 60)
    log.info(f"TEST: {name}")
    log.info("=" * 60)

    success, output, elapsed = run_cmd(cmd, timeout=timeout)
    errors = check_errors(output)
    result = {
        "name": name,
        "success": success,
        "expect_fail": expect_fail,
        "elapsed": elapsed,
        "errors": errors,
    }

    if expect_fail:
        result["status"] = "xfail" if not success else "UNEXPECTED_PASS"
    else:
        result["status"] = "PASS" if success else "FAIL"

    if not success and errors:
        log.warning(f"  Errors found:")
        for e in errors[:5]:
            log.warning(f"    {e}")

    RESULTS.append(result)

    status_icon = "[PASS]" if (success == (not expect_fail)) else "[FAIL]"
    log.info(f"  Result: {status_icon} ({elapsed:.1f}s)")

    return success, errors, output


def main():
    log.info("")
    log.info("=" * 70)
    log.info(f"  LISA-AutoResearch Test Runner")
    log.info(f"  Started: {datetime.now()}")
    log.info("=" * 70)

    all_passed = True

    # ── Test 1: Unit tests ──────────────────────────────────────────
    ok, errors, output = test(
        "test_federated (25 unit tests)",
        ["python", "test_federated.py"],
        timeout=300,
    )
    if not ok:
        all_passed = False
        log.warning(f"  test_federated failed with {len(errors)} error lines")

    # ── Test 2: HTTP integration ──────────────────────────────────────
    ok, errors, output = test(
        "test_http_integration (server+client HTTP)",
        ["python", "test_http_integration.py"],
        timeout=300,
    )
    if not ok:
        all_passed = False

    # ── Test 3: Federated simulation ────────────────────────────────
    ok, errors, output = test(
        "main.py --mode simulate (3 clients, 2 rounds)",
        ["python", "main.py", "--mode", "simulate", "--clients", "3", "--rounds", "2"],
        timeout=300,
    )
    if not ok:
        all_passed = False

    # ── Test 4: LISA training ───────────────────────────────────────
    ok, errors, output = test(
        "main.py --mode train (LISA PyTorch, 5 iters)",
        ["python", "main.py", "--mode", "train",
         "--model", "distilbert/distilgpt2",
         "--iters", "5",
         "--bottom", "1", "--top", "1", "--middle", "0",
         "--output", "output/test_train"],
        timeout=300,
    )
    if not ok:
        all_passed = False

    # ── Test 5: Hardware detection ──────────────────────────────────
    ok, errors, output = test(
        "main.py --mode hardware",
        ["python", "main.py", "--mode", "hardware"],
        timeout=30,
    )
    if not ok:
        all_passed = False

    # ── Test 6: Remaining modules ───────────────────────────────────
    ok, errors, output = test(
        "test_remaining.py (11 modules, 101 checks)",
        ["python", "test_remaining.py"],
        timeout=120,
    )
    if not ok:
        all_passed = False

    # ── Summary ─────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("  SUMMARY")
    log.info("=" * 70)

    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    xfailed = sum(1 for r in RESULTS if r["status"] == "xfail")

    for r in RESULTS:
        icon = "[PASS]" if r["status"] == "PASS" else "[FAIL]"
        log.info(f"  {icon} {r['name']} ({r['elapsed']:.0f}s)")

    log.info("")
    log.info(f"  Passed: {passed}")
    log.info(f"  Failed: {failed}")
    log.info(f"  Expected failures: {xfailed}")
    log.info(f"  Total: {len(RESULTS)}")

    if all_passed:
        log.info("")
        log.info("  ALL TESTS PASSED!")
    else:
        log.info("")
        log.info("  SOME TESTS FAILED - review errors above")

    # Write results JSON
    import json
    with open("test_results_latest.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "all_passed": all_passed,
            "results": RESULTS,
        }, f, indent=2)

    log.info(f"  Results saved to test_results_latest.json")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
