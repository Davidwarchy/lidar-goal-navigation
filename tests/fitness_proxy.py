"""
Test script for fitness proxy selection (health/energy) in NN strategies.

Runs all strategies under each fitness combo:
- fitness_proxy = none, fitness_pool = survivors_only
- fitness_proxy = health, fitness_pool = survivors_only
- fitness_proxy = health, fitness_pool = all_agents
- fitness_proxy = energy, fitness_pool = survivors_only
- fitness_proxy = energy, fitness_pool = all_agents

For basic strategies (random, levy, uniform), fitness args are ignored but still passed.
For NN strategies (random_nn, spiking), they are used for selection.

Uses:
- trials = 2, generations = 2, population = 1000, max_steps = 1000, map = 6.png
- Parallel execution for speed.
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

ALL_STRATEGIES = ["random_nn", "spiking"]

FITNESS_COMBOS = [
    ("none", "survivors_only"),
    ("health", "survivors_only"),
    ("health", "all_agents"),
    ("energy", "survivors_only"),
    ("energy", "all_agents"),
]

MAP = "6.png"


def run(strategy, proxy, pool):
    """Run a single strategy with given fitness_proxy and fitness_pool."""
    out_dir = os.path.join("output", "experiments", "fitness_proxy", proxy, pool, strategy)
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
        "--fitness_proxy", proxy,
        "--fitness_pool", pool,
    ]

    if strategy == "levy":
        cmd.extend(["--alpha", "1.6", "--min_step", "1.0", "--max_step_len", "200.0"])

    desc = f"{strategy} (proxy={proxy}, pool={pool})"
    print(f"\n[RUN] {desc}")
    print(f"  Output: {out_dir}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAIL] {desc}")
    else:
        print(f"[OK]   {desc}")
    return result.returncode


def main():
    print("\n" + "="*60)
    print("FITNESS PROXY TEST")
    print("="*60)
    print(f"Trials: {BASE['trials']}")
    print(f"Generations: {BASE['generations']}")
    print(f"Population: {BASE['population']}")
    print(f"Max steps: {BASE['max_steps']}")
    print(f"Map: {MAP}")
    print(f"Strategies: {ALL_STRATEGIES}")
    print(f"Fitness combos: {FITNESS_COMBOS}")
    print("="*60)

    # Outer loop: fitness combos
    for proxy, pool in FITNESS_COMBOS:
        print(f"\n{'#'*60}")
        print(f"FITNESS: proxy={proxy}, pool={pool}")
        print(f"{'#'*60}")

        # Inner loop: all strategies
        for strategy in ALL_STRATEGIES:
            run(strategy, proxy, pool)
            print("-" * 40)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()