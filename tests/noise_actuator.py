"""
Test script for motor noise types.
Tests Slip and Deadzone noise independently.
"""

import subprocess
import os
import sys

BASE = {
    "trials": 2,
    "generations": 2,
    "population": 1000,
    "max_steps": 100,
}

STRATEGIES = ["random_nn", "spiking"]
MAP = "6.png"

# Motor noise configurations
MOTOR_NOISE_TESTS = [
    # (name, noise_type, extra_args)
    ("baseline", "none", []),
    
    # Slip tests
    ("slip_prob0.05_mag0.3", "slip", ["--motor_slip_prob", "0.05", "--motor_slip_mag", "0.3"]),
    ("slip_prob0.1_mag0.5", "slip", ["--motor_slip_prob", "0.1", "--motor_slip_mag", "0.5"]),
    ("slip_prob0.2_mag0.7", "slip", ["--motor_slip_prob", "0.2", "--motor_slip_mag", "0.7"]),
    ("slip_prob0.3_mag1.0", "slip", ["--motor_slip_prob", "0.3", "--motor_slip_mag", "1.0"]),
    
    # Deadzone tests
    ("deadzone_0.05", "deadzone", ["--motor_deadzone", "0.05"]),
    ("deadzone_0.1", "deadzone", ["--motor_deadzone", "0.1"]),
    ("deadzone_0.2", "deadzone", ["--motor_deadzone", "0.2"]),
    ("deadzone_0.5", "deadzone", ["--motor_deadzone", "0.5"]),
]

def run(strategy, test_name, noise_type, extra_args):
    out_dir = os.path.join(
        "output", "experiments", "motor_noise", test_name, strategy
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
        "--motor_noise", noise_type,
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
    print("MOTOR NOISE EXPERIMENTS")
    print("="*60)
    print(f"Trials: {BASE['trials']}, Generations: {BASE['generations']}, Population: {BASE['population']}")
    print("="*60)

    for test_name, noise_type, extra_args in MOTOR_NOISE_TESTS:
        print(f"\n{'#'*60}")
        print(f"TEST: {test_name} (type={noise_type})")
        print(f"{'#'*60}")

        for strategy in STRATEGIES:
            run(strategy, test_name, noise_type, extra_args)
            print("-" * 40)

    print("\n=== DONE ===")
    print("Results stored in output/experiments/motor_noise/")