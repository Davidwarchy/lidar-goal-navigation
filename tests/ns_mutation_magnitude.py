import subprocess
import os

BASE = {
    "trials": 7,
    "generations": 200,
    "population": 1000,
    "max_steps": 1000,
}

MUTATION_MAGS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

STRATEGIES = [
    "random_nn",
    "spiking"
]

MAP = "6.png"


def run(strategy, mutation_mag):
    out_dir = os.path.join(
        "output", "experiments", "mutation_magnitude", str(mutation_mag), strategy
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
        "--mutation_rate", "0.1",  # fixed rate
        "--mutation_mag", str(mutation_mag),
        "--parallel"
    ]

    if strategy == "levy":
        cmd += ["--alpha", "1.6", "--min_step", "1.0", "--max_step_len", "200.0"]

    print(f"[RUN] {strategy} | mutation_magnitude={mutation_mag}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] {strategy} | mutation_magnitude={mutation_mag}")
    else:
        print(f"[OK]   {strategy} | mutation_magnitude={mutation_mag}")


if __name__ == "__main__":
    print("\n=== MUTATION MAGNITUDE SWEEP ===\n")

    for strategy in STRATEGIES:
        print(f"\n=== STRATEGY: {strategy.upper()} ===\n")

        for mag in MUTATION_MAGS:
            run(strategy, mag)

    print("\n=== DONE ===")