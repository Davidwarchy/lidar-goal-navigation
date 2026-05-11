#!/usr/bin/env python3
"""
Test script for environment variation.

Runs all strategies on multiple maps from environments/images/.
Hierarchy: output/experiments/environment/<env_name>/<strategy>/
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path

# ----------------------------------------------------------------------
# Default experiment parameters
# ----------------------------------------------------------------------
DEFAULTS = {
    "trials": 2,
    "generations": 2,
    "population": 100,
    "max_steps": 100,
    "map_dir": "environments/images",
    "output_root": "output/experiments/environment",
}

STRATEGIES = [
    "random",
    "levy",
    "uniform",
    "random_nn",
    "spiking",
]

# List of environment files to test.
# If None, they will be auto‑detected from map_dir.
ENV_FILES = None  # e.g. ["6.png", "maze.png", "corridor.png"]


def get_available_maps(map_dir):
    """Return list of .png map files found in map_dir."""
    img_dir = Path(map_dir)
    if not img_dir.exists():
        print(f"[WARN] Map directory not found: {map_dir}")
        return []
    maps = sorted([f.name for f in img_dir.glob("*.png")])
    if not maps:
        print(f"[WARN] No PNG maps found in {map_dir}")
    return maps


def run_experiment(env_name, strategy, args):
    """Run a single combination of environment and strategy."""
    out_dir = os.path.join(
        args.output_root,
        env_name.replace(".png", ""),   # remove extension for folder name
        strategy
    )
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "main.py",
        "--strategy", strategy,
        "--trials", str(args.trials),
        "--generations", str(args.generations),
        "--population", str(args.population),
        "--env", env_name,
        "--max_steps", str(args.max_steps),
        "--save_top_k", "0",
        "--output_dir", out_dir,
        "--parallel",
    ]

    # Add levy-specific parameters if needed
    if strategy == "levy":
        cmd.extend(["--alpha", "1.6", "--min_step", "1.0", "--max_step_len", "200.0"])

    print(f"\n[RUN] env={env_name} | strategy={strategy}")
    print(f"  Output: {out_dir}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"[OK]   env={env_name} | strategy={strategy}")
    else:
        print(f"[FAIL] env={env_name} | strategy={strategy}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Environment variation test")
    parser.add_argument("--trials", type=int, default=DEFAULTS["trials"])
    parser.add_argument("--generations", type=int, default=DEFAULTS["generations"])
    parser.add_argument("--population", type=int, default=DEFAULTS["population"])
    parser.add_argument("--max_steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--map_dir", type=str, default=DEFAULTS["map_dir"])
    parser.add_argument("--output_root", type=str, default=DEFAULTS["output_root"])
    parser.add_argument("--env", type=str, help="Run only a single environment (file name)")
    parser.add_argument("--strategy", type=str, help="Run only a single strategy")
    args = parser.parse_args()

    # Determine which environments to run
    if args.env:
        envs = [args.env]
    else:
        envs = get_available_maps(args.map_dir)
        if ENV_FILES is not None:
            # Use manually defined list
            envs = [e for e in ENV_FILES if e in envs]

    if not envs:
        print("No environments to test. Exiting.")
        sys.exit(1)

    # Determine which strategies to run
    if args.strategy:
        if args.strategy not in STRATEGIES:
            print(f"Invalid strategy {args.strategy}. Choose from {STRATEGIES}")
            sys.exit(1)
        strategies = [args.strategy]
    else:
        strategies = STRATEGIES

    print("\n" + "=" * 70)
    print("ENVIRONMENT VARIATION EXPERIMENT")
    print("=" * 70)
    print(f"Trials: {args.trials}")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Max steps: {args.max_steps}")
    print(f"Environments: {envs}")
    print(f"Strategies: {strategies}")
    print(f"Output root: {args.output_root}")
    print("=" * 70)

    success_counts = {}
    total = len(envs) * len(strategies)
    completed = 0

    for env in envs:
        for strat in strategies:
            ret = run_experiment(env, strat, args)
            completed += 1
            key = f"{env}/{strat}"
            success_counts[key] = (ret == 0)
            print(f"Progress: {completed}/{total}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, ok in success_counts.items():
        status = "OK" if ok else "FAIL"
        print(f"{key:40} : {status}")

    failed = sum(1 for ok in success_counts.values() if not ok)
    print(f"\nTotal: {len(success_counts)} runs, {failed} failed.")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()