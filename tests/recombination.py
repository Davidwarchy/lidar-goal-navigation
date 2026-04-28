import subprocess
import os
import sys

BASE = {
    "trials": 7,
    "generations": 200,
    "population": 1000,
    "max_steps": 1000,
}
STRATEGIES = ["random_nn", "spiking"]
MAP = "6.png"
RECOMBINATION_VALUES = [False, True]  # False = no recombination, True = with recombination

def run(strategy, recombination_enabled):
    subdir = "with_recombination" if recombination_enabled else "no_recombination"
    out_dir = os.path.join("output", "experiments", "recombination", subdir, strategy)
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
        "--parallel"
    ]
    if recombination_enabled:
        cmd.append("--recombination")

    print(f"\n[RUN] recombination={recombination_enabled} | strategy={strategy}")
    print(f"  Output: {out_dir}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAIL] rec={recombination_enabled}, strategy={strategy}")
    else:
        print(f"[OK]   rec={recombination_enabled}, strategy={strategy}")

if __name__ == "__main__":
    print("\n=== RECOMBINATION COMPARISON EXPERIMENT ===")
    print(f"Trials: {BASE['trials']}, Generations: {BASE['generations']}, Population: {BASE['population']}")
    
    # Outer loop: recombination configuration
    for rec_enabled in RECOMBINATION_VALUES:
        print(f"\n{'='*60}")
        print(f"CONFIGURATION: recombination={'ON' if rec_enabled else 'OFF'}")
        print(f"{'='*60}")
        # Inner loop: all strategies
        for strategy in STRATEGIES:
            run(strategy, rec_enabled)
    
    print("\n=== DONE ===")
    print("Results stored in output/experiments/recombination/")