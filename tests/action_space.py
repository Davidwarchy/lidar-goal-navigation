# tests/action_space.py
"""
Test script for comparing discrete vs continuous action spaces.
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

# ALL strategies (including those without neural networks)
STRATEGIES = [
    "random",
    "levy", 
    "uniform",
    "random_nn",
    "spiking"
]

ACTION_SPACES = ["discrete", "continuous"]

MAP = "6.png"


def run(strategy, action_space):
    """Run a single strategy with a specific action space."""
    out_dir = os.path.join(
        "output", "experiments", "action_space", action_space, strategy
    )
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "main.py",
        "--strategy", strategy,
        "--trials", str(BASE["trials"]),
        "--generations", str(BASE["generations"]),
        "--population", str(BASE["population"]),
        "--env", MAP,
        "--max_steps", str(BASE["max_steps"]),
        "--action_space", action_space,
        "--save_top_k", "0",
        "--output_dir", out_dir,
        "--parallel"
    ]

    # Strategy-specific parameters
    if strategy == "levy":
        cmd.extend(["--alpha", "1.6", "--min_step", "1.0", "--max_step_len", "200.0"])

    print(f"[RUN] {strategy} | action_space={action_space}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {strategy} | action_space={action_space}")
    else:
        print(f"[OK]   {strategy} | action_space={action_space}")


if __name__ == "__main__":
    print("\n=== ACTION SPACE COMPARISON ===\n")
    print(f"Trials: {BASE['trials']}")
    print(f"Generations: {BASE['generations']}")
    print(f"Population: {BASE['population']}")
    print(f"Max steps: {BASE['max_steps']}")
    print(f"Map: {MAP}")
    print(f"Action spaces: {ACTION_SPACES}")
    print("\n")

    for action_space in ACTION_SPACES:
        print(f"\n=== ACTION SPACE: {action_space.upper()} ===\n")
        
        for strategy in STRATEGIES:
            run(strategy, action_space)
            print("-" * 40)

    print("\n=== DONE ===")