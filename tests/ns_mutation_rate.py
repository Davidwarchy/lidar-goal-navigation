import subprocess
import os

BASE = {
    "trials": 7,
    "generations": 200,
    "population": 1000,
    "max_steps": 1000, 
}

MUTATION_RATES = [0.01, 0.05, 0.1, 0.3, 0.5, 0.75]

STRATEGIES = [
    "random_nn",
    "spiking"
]

MAP = "6.png"


def run(strategy, mutation_rate):
    out_dir = os.path.join(
        "output", "experiments", "mutation_rate", str(mutation_rate), strategy
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
        "--mutation_rate", str(mutation_rate), 
        "--parallel"
    ]

    if strategy == "levy":
        cmd += ["--alpha", "1.6", "--min_step", "1.0", "--max_step_len", "200.0"]

    print(f"[RUN] {strategy} | mutation_rate={mutation_rate}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {strategy} | mutation_rate={mutation_rate}")
    else:
        print(f"[OK]   {strategy} | mutation_rate={mutation_rate}")


if __name__ == "__main__":
    print("\n=== MUTATION RATE SWEEP ===\n")

    for m in MUTATION_RATES:
        print(f"\n=== MUTATION RATE: {m} ===\n")

        for strategy in STRATEGIES:
            run(strategy, m)

    print("\n=== DONE ===")