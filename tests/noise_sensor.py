"""
Test script for LiDAR noise types.
Tests Gaussian, Uniform, and Dropout noise independently.
"""

import subprocess
import os
import sys

BASE = {
    "trials": 7,
    "generations": 200,
    "population": 1000,
    "max_steps": 1000,
}

STRATEGIES = ["random_nn", "spiking"]
MAP = "6.png"

# Noise configurations
LIDAR_NOISE_TESTS = [
    # (name, noise_type, noise_params as args list)
    ("baseline", "none", []),
    ("gaussian_std2", "gaussian", ["--lidar_noise_std", "2.0"]),
    ("gaussian_std5", "gaussian", ["--lidar_noise_std", "5.0"]),
    ("gaussian_std10", "gaussian", ["--lidar_noise_std", "10.0"]),
    ("uniform_max5", "uniform", ["--lidar_noise_max", "5.0"]),
    ("uniform_max10", "uniform", ["--lidar_noise_max", "10.0"]),
    ("uniform_max20", "uniform", ["--lidar_noise_max", "20.0"]),
    ("dropout_p0.05", "dropout", ["--lidar_dropout_p", "0.05"]),
    ("dropout_p0.1", "dropout", ["--lidar_dropout_p", "0.1"]),
    ("dropout_p0.2", "dropout", ["--lidar_dropout_p", "0.2"]),
]

def run(strategy, test_name, noise_type, extra_args):
    out_dir = os.path.join(
        "output", "experiments", "noise_sensor", test_name, strategy
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
        "--lidar_noise", noise_type,
    ] + extra_args

    print(f"\n[RUN] {strategy} | {test_name}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {strategy} | {test_name}")
    else:
        print(f"[OK]   {strategy} | {test_name}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LIDAR NOISE EXPERIMENTS")
    print("="*60)
    print(f"Trials: {BASE['trials']}, Generations: {BASE['generations']}, Population: {BASE['population']}")
    print("="*60)

    for test_name, noise_type, extra_args in LIDAR_NOISE_TESTS:
        print(f"\n{'#'*60}")
        print(f"TEST: {test_name} (type={noise_type})")
        print(f"{'#'*60}")

        for strategy in STRATEGIES:
            run(strategy, test_name, noise_type, extra_args)
            print("-" * 40)

    print("\n=== DONE ===")
    print("Results stored in output/experiments/lidar_noise/")