#!/usr/bin/env python3
"""
Quick smoke test to verify basic functionality.
Runs each strategy for a minimal duration to catch obvious breakages.
"""

import subprocess
import sys

def test_strategy(strategy, extra_args=None):
    cmd = [
        sys.executable, "main.py",
        "--strategy", strategy,
        "--trials", "2",
        "--generations", "2",
        "--population", "1000",
        "--max_steps", "1000",
        "--env", "6.png",
        "--output_dir", "output/smoke_test",
    ]
    if extra_args:
        cmd.extend(extra_args)
    if strategy == "levy":
        cmd.extend(["--alpha", "1.6", "--min_step", "1.0", "--max_step_len", "200.0"])
    
    print(f"Testing {strategy}...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        print("✓ OK")
        return True
    else:
        print(f"✗ FAILED (code {result.returncode})")
        print(result.stderr.decode()[:200])
        return False

def main():
    strategies = ["random", "levy", "uniform", "random_nn", "spiking"]
    print("SMOKE TEST - Quick functionality check\n")
    passed = 0
    for s in strategies:
        if test_strategy(s):
            passed += 1
    print(f"\nResult: {passed}/{len(strategies)} passed")
    sys.exit(0 if passed == len(strategies) else 1)

if __name__ == "__main__":
    main()