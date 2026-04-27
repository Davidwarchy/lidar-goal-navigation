import subprocess
import os

BASE = {
    "trials": 7,
    "generations": 200,
    "population": 1000,
    "max_steps": 1000,
}

STRATEGIES = [
    "random",
    "levy", 
    "uniform",
    # "random_nn",
    # "spiking"
]

MAP = "6.png"


def run(strategy):
    out_dir = os.path.join(
        "output", "experiments", "standard", strategy
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
        "--parallel"
    ]

    print(f"[RUN] {strategy} (standard)")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {strategy}")
    else:
        print(f"[OK]   {strategy}")


if __name__ == "__main__":
    print("\n=== STANDARD RUN (max_steps=1000) ===\n")

    for strategy in STRATEGIES:
        run(strategy)

    print("\n=== DONE ===")