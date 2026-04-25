import subprocess
import os
import sys
from datetime import datetime

# --- Standards ---
TRIALS = 7
GENERATIONS = 200
MAX_STEPS = 1_000
STRATEGIES = ["random", "levy", "uniform", "random_nn", "spiking"]
POPULATIONS = [100, 500, 10_000] # Standard 1000 is skipped coz it's in standard test configuration 

def run_population_experiments():
    # Create a unique root for this variable test
    base_root = os.path.join("output", "experiments", "population")
    
    print(f"Starting population size experiments. Root: {base_root}")
    
    for pop in POPULATIONS:
        print(f"\n--- Testing Population Size: {pop} ---")
        
        for strategy in STRATEGIES:
            # Map strategy names to CLI flags
            cmd_strategy = "random_nn" if strategy == "random_nn" else strategy
            if strategy == "spiking": cmd_strategy = "spiking"
            
            # Define output path: experiments/population/<val>/<strategy>
            output_dir = os.path.join(base_root, str(pop), strategy)
            os.makedirs(output_dir, exist_ok=True)
            
            cmd = [
                sys.executable, "main.py",
                "--strategy", cmd_strategy,
                "--trials", str(TRIALS),
                "--generations", str(GENERATIONS),
                "--population", str(pop),
                "--max_steps", str(MAX_STEPS),
                "--output_dir", output_dir,
                "--save_top_k", "0",
                "--parallel" 
            ]
            
            print(f"Running {strategy}...")
            try:
                subprocess.run(cmd, check=True)
                print(f"✓ Completed: {strategy} (Pop: {pop})")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed: {strategy} (Pop: {pop}) - Error: {e}")

if __name__ == "__main__":
    run_population_experiments()