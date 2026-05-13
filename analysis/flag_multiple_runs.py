import os

def find_strategies_with_multiple_runs(base_experiments_folder, threshold=1):
    """
    Identifies strategy folders that contain more than 'threshold' runs.
    Each run is defined as a directory containing 'trial_' folders.
    """
    # Key: path to strategy folder, Value: set of run (timestamp) folder names
    strategy_runs = {}

    for root, dirs, files in os.walk(base_experiments_folder):
        # If this directory contains 'trial_' subdirectories, it is a "Run" folder
        if any(d.startswith('trial_') for d in dirs):
            run_folder = os.path.basename(root)
            strategy_path = os.path.dirname(root)
            
            if strategy_path not in strategy_runs:
                strategy_runs[strategy_path] = set()
            
            strategy_runs[strategy_path].add(run_folder)

    print(f"{'Strategy Path':<70} | {'Run Count':<10}")
    print("-" * 85)
    
    found_any = False
    for path, runs in strategy_runs.items():
        if len(runs) > threshold:
            # Get the relative path for cleaner printing
            rel_path = os.path.relpath(path, base_experiments_folder)
            print(f"{rel_path:<70} | {len(runs):<10}")
            # Optional: print the specific run names
            # for run in runs: print(f"  - {run}")
            found_any = True
            
    if not found_any:
        print(f"No strategy folders found with more than {threshold} runs.")

# Usage
experiments_dir = r"C:\Users\HP\Desktop\Projects\navigation\10-lidar-goal-navigation\output\experiments"
# Set threshold=1 to find strategies with 2 or more runs
# Set threshold=2 to find strategies with more than 2 runs
find_strategies_with_multiple_runs(experiments_dir, threshold=1)