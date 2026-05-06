"""
Test script for observation latency (delay).
Tests different delay lengths independently.
"""

import subprocess
import os
import sys

BASE = {
    "trials": 2,
    "generations": 2,
    "population": 1000,
    "max_steps": 1000,
}

STRATEGIES = ["random_nn", "spiking"]
MAP = "6.png"

# Latency configurations (delay in steps)
LATENCY_TESTS = [
    ("baseline", 0),
    ("delay_1step", 1),
    ("delay_2steps", 2),
    ("delay_3steps", 3),
    ("delay_5steps", 5),
    ("delay_10steps", 10),
]

def run(strategy, test_name, delay):
    out_dir = os.path.join(
        "output", "experiments", "latency", test_name, strategy
    )
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "main.py",
        "--strategy", strategy,
        "--trials", str(BASE["trials"]),
        "--generations", str(BASE["generations"]),
        "--population", str(BASE["population"]),
        "--env", MAP,
        "--max_steps", str(BASE["max_steps"]),
        "--save_top_k", "0",
        "--output_dir", out_dir,
        "--parallel",
        "--obs_delay", str(delay),
    ]

    print(f"\n[RUN] {strategy} | {test_name} (delay={delay} steps)")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {strategy} | {test_name}")
    else:
        print(f"[OK]   {strategy} | {test_name}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("OBSERVATION LATENCY EXPERIMENTS")
    print("="*60)
    print(f"Trials: {BASE['trials']}, Generations: {BASE['generations']}, Population: {BASE['population']}")
    print("="*60)

    for test_name, delay in LATENCY_TESTS:
        print(f"\n{'#'*60}")
        print(f"TEST: {test_name} (delay={delay} steps)")
        print(f"{'#'*60}")

        for strategy in STRATEGIES:
            run(strategy, test_name, delay)
            print("-" * 40)

    print("\n=== DONE ===")
    print("Results stored in output/experiments/latency/")