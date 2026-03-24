#!/usr/bin/env python3
"""Wrapper to run quick_lr_test.py with unbuffered output."""
import sys, os, subprocess

root = r"C:\Users\james\.openclaw\workspace\LISA_FTM"
log_out = os.path.join(root, "quick_lr_test3.log")
log_err = os.path.join(root, "quick_lr_test3.err")

proc = subprocess.Popen(
    [sys.executable, "quick_lr_test.py"],
    cwd=root,
    stdout=open(log_out, "w", buffering=1),
    stderr=subprocess.STDOUT,
)
proc.wait()
print(f"Exit: {proc.returncode}")
