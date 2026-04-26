import subprocess
import os

BASE = {
    "trials": 7,
    "generations": 200,
    "population": 1000,
}

MAX_STEPS_LIST = [100, 500, 10000]

STRATEGIES = [
    "random",
    "levy", 
    "uniform",
    "random_nn",
    "spiking"
]

MAP = "6.png"

# Default mutation parameters (you can change these)
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_MUTATION_MAG = 0.5


def run(strategy, max_steps):
    out_dir = os.path.join(
        "output", "experiments", "max_steps", str(max_steps), strategy
    )
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "main.py",
        "--strategy", strategy,
        "--trials", str(BASE["trials"]),
        "--generations", str(BASE["generations"]),
        "--population", str(BASE["population"]),
        "--env", MAP,
        "--max_steps", str(max_steps),
        "--save_top_k", "0",
        "--output_dir", out_dir,
        "--parallel"
    ]

    print(f"[RUN] {strategy} | max_steps={max_steps}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {strategy} | max_steps={max_steps}")
    else:
        print(f"[OK]   {strategy} | max_steps={max_steps}")


if __name__ == "__main__":
    print("\n=== MAX_STEPS SWEEP ===\n")

    for steps in MAX_STEPS_LIST:
        print(f"\n=== MAX_STEPS: {steps} ===\n")

        for strategy in STRATEGIES:
            run(strategy, steps)

    print("\n=== DONE ===")