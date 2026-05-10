#!/usr/bin/env python3
"""
Test script for varying number of hidden layers in neural network strategies
(random_nn and spiking). Each hidden layer uses the same size (64 neurons).
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

# Define layer configurations: (display_name, hidden_sizes_string)
# Each string is comma-separated values for number of neurons per layer.
LAYER_CONFIGS = [
    ("1layer_64", "64"),
    ("2layers_64", "64,64"),
    ("3layers_64", "64,64,64"),
]

STRATEGIES = ["random_nn", "spiking"]
MAP = "6.png"


def run(strategy, config_name, hidden_sizes_str):
    """Run a single strategy with a given hidden layer configuration."""
    out_dir = os.path.join(
        "output", "experiments", "network_layers", config_name, strategy
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

    # Additional parameters for spiking networks (optional but keep consistency)
    if strategy == "spiking":
        # Default LIF steps = 5; can be kept as is.
        pass

    print(f"\n[RUN] {strategy} | {config_name} (hidden_sizes={hidden_sizes_str})")
    print(f"  Output: {out_dir}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {strategy} | {config_name}")
    else:
        print(f"[OK]   {strategy} | {config_name}")


def main():
    print("\n" + "=" * 70)
    print("NETWORK LAYERS EXPERIMENT")
    print("=" * 70)
    print(f"Trials: {BASE['trials']}, Generations: {BASE['generations']}, "
          f"Population: {BASE['population']}, Max steps: {BASE['max_steps']}")
    print(f"Layer configurations: {[c[0] for c in LAYER_CONFIGS]}")
    print("=" * 70)

    for config_name, hidden_sizes_str in LAYER_CONFIGS:
        print(f"\n{'#' * 70}")
        print(f"CONFIGURATION: {config_name} (hidden_sizes={hidden_sizes_str})")
        print(f"{'#' * 70}")
        for strategy in STRATEGIES:
            run(strategy, config_name, hidden_sizes_str)
            print("-" * 50)

    print("\n=== DONE ===")
    print("Results stored in output/experiments/network_layers/")


if __name__ == "__main__":
    main()