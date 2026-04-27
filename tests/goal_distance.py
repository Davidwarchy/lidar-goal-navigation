import subprocess
import os

BASE = {
    "trials": 7,
    "generations": 200,
    "population": 1000,
    "max_steps": 1000, 
}

GOAL_DISTANCES = [10, 20, 30, 40]

STRATEGIES = [
    "random",
    "levy",
    "uniform",
    "random_nn",
    "spiking"
]

MAP = "6.png"


def run(strategy, goal_distance):
    out_dir = os.path.join(
        "output", "experiments", "goal_distance", str(goal_distance), strategy
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
        "--save_top_k", "0",
        "--output_dir", out_dir,
        "--goal_spawn_dist", str(goal_distance),
        "--parallel"
    ]

    if strategy == "levy":
        cmd += ["--alpha", "1.6", "--min_step", "1.0", "--max_step_len", "200.0"]

    print(f"[RUN] {strategy} | goal_distance={goal_distance}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {strategy} | goal_distance={goal_distance}")
    else:
        print(f"[OK]   {strategy} | goal_distance={goal_distance}")


if __name__ == "__main__":
    print("\n=== GOAL DISTANCE SWEEP ===\n")

    for d in GOAL_DISTANCES:
        print(f"\n=== GOAL DISTANCE: {d} ===\n")

        for strategy in STRATEGIES:
            run(strategy, d)
            print("-"*80)

        print("#"*80)

    print("\n=== DONE ===") 