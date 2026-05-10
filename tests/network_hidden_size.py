#!/usr/bin/env python3
"""
Test script for varying hidden layer size (breadth) in neural network strategies
(random_nn and spiking). Keeps depth fixed at 1 hidden layer.
"""

import subprocess
import os
import sys

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
BASE = {
    "trials": 7,
    "generations": 200,
    "population": 1000,
    "max_steps": 1000,
}

# Hidden layer sizes to test (single hidden layer)
HIDDEN_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]

STRATEGIES = ["random_nn", "spiking"]
MAP = "6.png"


def run(strategy, hidden_size):
    """Run a single strategy with a given hidden layer size (single layer)."""
    # Convert hidden size to string format expected by --hidden_sizes
    hidden_sizes_str = str(hidden_size)

    out_dir = os.path.join(
        "output", "experiments", "hidden_layer_size", str(hidden_size), strategy
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
        "--hidden_sizes", hidden_sizes_str,
    ]

    # For spiking, keep default lif_steps=5 (no change needed)
    print(f"\n[RUN] {strategy} | hidden_size={hidden_size}")
    print(f"  Output: {out_dir}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {strategy} | hidden_size={hidden_size}")
    else:
        print(f"[OK]   {strategy} | hidden_size={hidden_size}")


def main():
    print("\n" + "=" * 70)
    print("HIDDEN LAYER SIZE (BREADTH) EXPERIMENT")
    print("=" * 70)
    print(f"Trials: {BASE['trials']}, Generations: {BASE['generations']}, "
          f"Population: {BASE['population']}, Max steps: {BASE['max_steps']}")
    print(f"Hidden layer sizes: {HIDDEN_SIZES} (single hidden layer)")
    print("=" * 70)

    for size in HIDDEN_SIZES:
        print(f"\n{'#' * 70}")
        print(f"HIDDEN LAYER SIZE: {size}")
        print(f"{'#' * 70}")
        for strategy in STRATEGIES:
            run(strategy, size)
            print("-" * 50)

    print("\n=== DONE ===")
    print("Results stored in output/experiments/hidden_layer_size/")


if __name__ == "__main__":
    main()