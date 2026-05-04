# tests/robot_size.py
"""
Test script for robot size experiments.
Tests different robot radii to see how they affect navigation success.
"""

import subprocess
import os
import sys

BASE = {
    "trials": 1,
    "generations": 2,
    "population": 100,
    "max_steps": 100,
}

# Robot radii to test (pixels)
ROBOT_SIZES = [1, 3]

STRATEGIES = [
    "random",
    "levy",
    "uniform",
    "random_nn",
    "spiking"
]

MAP = "6.png"


def run(strategy, robot_radius):
    out_dir = os.path.join(
        "output", "experiments", "robot_size", str(robot_radius), strategy
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
        "--parallel"
    ]

    # Add robot radius parameter (needs to be added to env_params in main.py)
    # The env expects robot_radius in env_params, so we need to pass it through
    cmd.extend(["--robot_radius", str(robot_radius)])

    if strategy == "levy":
        cmd.extend(["--alpha", "1.6", "--min_step", "1.0", "--max_step_len", "200.0"])

    print(f"[RUN] {strategy} | robot_radius={robot_radius}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {strategy} | robot_radius={robot_radius}")
    else:
        print(f"[OK]   {strategy} | robot_radius={robot_radius}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ROBOT SIZE EXPERIMENTS")
    print("="*60)
    print(f"Robot radii: {ROBOT_SIZES}")
    print("="*60 + "\n")

    for radius in ROBOT_SIZES:
        print(f"\n{'='*60}")
        print(f"ROBOT RADIUS: {radius} pixels")
        print()

        for strategy in STRATEGIES:
            run(strategy, radius)
            print("-" * 40)

    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)
    print("Results stored in output/experiments/robot_size/")