# tests/action_distribution.py
"""
Test script for action space + action distribution combinations.

Covers:
- discrete + deterministic
- discrete + stochastic
- continuous + deterministic
- continuous + stochastic
"""

import subprocess
import os
import sys

BASE = {
    "trials": 1,
    "generations": 2,
    "population": 1000,
    "max_steps": 1000,
}

STRATEGIES = [
    "random_nn",
    "spiking"
]

ACTION_SPACES = ["discrete", "continuous"]
ACTION_DISTRIBUTIONS = ["deterministic", "stochastic"]

MAP = "6.png"


def run(strategy, action_space, action_distribution):
    out_dir = os.path.join(
        "output",
        "experiments",
        "action_distribution",
        action_space,
        action_distribution,
        strategy
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
        "--action_space", action_space,
        "--action_distribution", action_distribution,
        "--save_top_k", "0",
        "--output_dir", out_dir,
        "--parallel"
    ]

    print(
        f"[RUN] {strategy} | "
        f"action_space={action_space} | "
        f"action_distribution={action_distribution}"
    )

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(
            f"[FAIL] {strategy} | "
            f"{action_space} | {action_distribution}"
        )
    else:
        print(
            f"[OK]   {strategy} | "
            f"{action_space} | {action_distribution}"
        )


if __name__ == "__main__":
    print("\n=== ACTION DISTRIBUTION COMPARISON ===\n")

    for action_space in ACTION_SPACES:
        for action_distribution in ACTION_DISTRIBUTIONS:
            print(
                f"\n=== {action_space.upper()} + "
                f"{action_distribution.upper()} ===\n"
            )

            for strategy in STRATEGIES:
                run(strategy, action_space, action_distribution)
                print("-" * 50)

    print("\n=== DONE ===")