#!/usr/bin/env python3
"""
Curriculum learning ablation test – proper folder hierarchy.

Structure:
    output/experiments/curriculum/<timestamp>/
        baseline/
            off/
                random_nn/
                spiking/
        threshold/
            0.03/
                random_nn/
                spiking/
            0.05/
                ...
            0.1/
                ...
        consecutive/
            2/
                random_nn/
                spiking/
            3/
                ...
            5/
                ...
        increment/
            2.0/
                random_nn/
                spiking/
            5.0/
                ...
            10.0/
                ...
"""

import subprocess
import os
import sys
from datetime import datetime

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
BASE = {
    "trials": 7,               # increase for production (e.g., 7)
    "generations": 200,         # increase for production (e.g., 200)
    "population": 1000,         # increase for production (e.g., 1000)
    "max_steps": 1000,
    "map": "6.png",
}

STRATEGIES = ["random_nn", "spiking"]

# Curriculum parameter definitions
DEFAULT_CURRICULUM = {
    "success_threshold": 0.05,
    "consecutive_gens": 3,
    "distance_increment": 5.0,
}

# Sweep values
THRESHOLD_SWEEP = [0.03, 0.05, 0.1]
CONSECUTIVE_SWEEP = [2, 3, 5]
INCREMENT_SWEEP = [1.0, 2.0, 3.0, 5.0]

# Baseline (curriculum off) – a special dimension
BASELINE_DIM = "baseline"
BASELINE_VAL = "off"


# ----------------------------------------------------------------------
# Core experiment runner
# ----------------------------------------------------------------------
def run_experiment(strategy, dimension, param_value, curriculum_params, base_args):
    """
    Run a single curriculum experiment.
    
    Args:
        strategy: 'random_nn' or 'spiking'
        dimension: 'baseline', 'threshold', 'consecutive', 'increment'
        param_value: value as string (e.g., '0.05', '3', '5.0', 'off')
        curriculum_params: dict with keys:
            - enabled (bool)
            - success_threshold (float, if enabled)
            - consecutive_gens (int, if enabled)
            - distance_increment (float, if enabled)
        base_args: common arguments (output_root, trials, etc.)
    
    Returns:
        bool: True if subprocess succeeded
    """
    # Build output path: output_root / dimension / param_value / strategy
    out_dir = os.path.join(base_args["output_root"], dimension, param_value, strategy)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "main.py",
        "--strategy", strategy,
        "--trials", str(base_args["trials"]),
        "--generations", str(base_args["generations"]),
        "--population", str(base_args["population"]),
        "--env", base_args["map"],
        "--max_steps", str(base_args["max_steps"]),
        "--save_top_k", "0",
        "--output_dir", out_dir,
        "--parallel",
    ]

    if curriculum_params["enabled"]:
        cmd.append("--ga_curriculum")
        cmd.extend([
            "--ga_curriculum_success_threshold", str(curriculum_params["success_threshold"]),
            "--ga_curriculum_consecutive_gens", str(curriculum_params["consecutive_gens"]),
            "--ga_curriculum_distance_increment", str(curriculum_params["distance_increment"]),
        ])

    print(f"\n[RUN] {strategy} | {dimension} = {param_value}")
    print(f"  Output: {out_dir}")
    if curriculum_params["enabled"]:
        print(f"  Curriculum: thr={curriculum_params['success_threshold']}, "
              f"cons={curriculum_params['consecutive_gens']}, inc={curriculum_params['distance_increment']}")
    else:
        print("  Curriculum: OFF")

    result = subprocess.run(cmd)
    return result.returncode == 0


# ----------------------------------------------------------------------
# Main: iterate over dimensions and values
# ----------------------------------------------------------------------
def main():
    output_root = os.path.join("output", "experiments", "curriculum")
    os.makedirs(output_root, exist_ok=True)

    base_args = {
        "output_root": output_root,
        "trials": BASE["trials"],
        "generations": BASE["generations"],
        "population": BASE["population"],
        "max_steps": BASE["max_steps"],
        "map": BASE["map"],
    }

    print("\n" + "="*70)
    print("CURRICULUM LEARNING ABLATION")
    print(f"Output root: {output_root}")
    print(f"Trials: {BASE['trials']}, Generations: {BASE['generations']}, "
          f"Population: {BASE['population']}, Max steps: {BASE['max_steps']}")
    print("="*70)

    success_counts = {}

    # ------------------------------
    # 1. Baseline (curriculum OFF)
    # ------------------------------
    dimension = BASELINE_DIM
    param_value = BASELINE_VAL
    curriculum_params = {"enabled": False}

    for strategy in STRATEGIES:
        ok = run_experiment(strategy, dimension, param_value, curriculum_params, base_args)
        success_counts[f"{dimension}/{param_value}/{strategy}"] = ok

    # ------------------------------
    # 2. Sweep success threshold
    # ------------------------------
    dimension = "threshold"
    for thr in THRESHOLD_SWEEP:
        param_value = str(thr)
        curriculum_params = {
            "enabled": True,
            "success_threshold": thr,
            "consecutive_gens": DEFAULT_CURRICULUM["consecutive_gens"],
            "distance_increment": DEFAULT_CURRICULUM["distance_increment"],
        }
        for strategy in STRATEGIES:
            ok = run_experiment(strategy, dimension, param_value, curriculum_params, base_args)
            success_counts[f"{dimension}/{param_value}/{strategy}"] = ok

    # ------------------------------
    # 3. Sweep consecutive generations
    # ------------------------------
    dimension = "consecutive"
    for cons in CONSECUTIVE_SWEEP:
        param_value = str(cons)
        curriculum_params = {
            "enabled": True,
            "success_threshold": DEFAULT_CURRICULUM["success_threshold"],
            "consecutive_gens": cons,
            "distance_increment": DEFAULT_CURRICULUM["distance_increment"],
        }
        for strategy in STRATEGIES:
            ok = run_experiment(strategy, dimension, param_value, curriculum_params, base_args)
            success_counts[f"{dimension}/{param_value}/{strategy}"] = ok

    # ------------------------------
    # 4. Sweep distance increment
    # ------------------------------
    dimension = "increment"
    for inc in INCREMENT_SWEEP:
        param_value = str(inc)
        curriculum_params = {
            "enabled": True,
            "success_threshold": DEFAULT_CURRICULUM["success_threshold"],
            "consecutive_gens": DEFAULT_CURRICULUM["consecutive_gens"],
            "distance_increment": inc,
        }
        for strategy in STRATEGIES:
            ok = run_experiment(strategy, dimension, param_value, curriculum_params, base_args)
            success_counts[f"{dimension}/{param_value}/{strategy}"] = ok

    # ------------------------------
    # Final summary
    # ------------------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, ok in success_counts.items():
        status = "OK" if ok else "FAIL"
        print(f"{name:50} : {status}")
    print(f"\nAll results saved under: {output_root}")


if __name__ == "__main__":
    main()