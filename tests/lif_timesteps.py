import subprocess
import os
import sys

BASE = {
    "trials": 7,
    "generations": 200,
    "population": 1000,
    "max_steps": 1000,
}
STRATEGY = "spiking"   # only spiking uses lif_steps
MAP = "6.png"
LIF_STEPS_VALUES = [1, 2, 5, 10]

def run(n_steps):
    out_dir = os.path.join("output", "experiments", "lif_steps", str(n_steps), STRATEGY)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "main.py",
        "--strategy", STRATEGY,
        "--trials", str(BASE["trials"]),
        "--generations", str(BASE["generations"]),
        "--population", str(BASE["population"]),
        "--env", MAP,
        "--max_steps", str(BASE["max_steps"]),
        "--save_top_k", "0",
        "--lif_steps", str(n_steps),
        "--output_dir", out_dir,
        "--parallel"
    ]

    print(f"\n[RUN] LIF steps = {n_steps}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[OK]   LIF steps = {n_steps}")
    else:
        print(f"[FAIL] LIF steps = {n_steps}")

if __name__ == "__main__":
    print("=== VARYING LIF TIMESTEPS EXPERIMENT ===")
    for steps in LIF_STEPS_VALUES:
        run(steps)
    print("\n=== DONE ===")