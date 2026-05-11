"""
It scans the main experiments directory, identifies all top-level experiment parameters (such as action, max_steps, or other hyperparameters), and recursively traverses their subfolders to locate individual trial result files ending in summary.csv. From each discovered trial, it extracts useful metadata directly from the folder hierarchy, including:
"""
import os
import pandas as pd
import glob

def consolidate_experiment_data(base_experiments_folder):
    """
    Traverses the experiment folder, bundles sub-parameters, and creates
    a <parameter>.csv for each main category.
    """
    # Get all top-level parameter folders (e.g., 'action', 'max_steps')
    parameters = [d for d in os.listdir(base_experiments_folder) 
                  if os.path.isdir(os.path.join(base_experiments_folder, d))]

    for param in parameters:
        param_path = os.path.join(base_experiments_folder, param)
        all_data = []
        
        # We look for trial folders deep in the tree
        # Structure: param / [sub_params...] / strategy / timestamp_folder / trial_X
        for root, dirs, files in os.walk(param_path):
            if 'trial_' in os.path.basename(root):
                trial_folder = root
                trial_name = os.path.basename(trial_folder).replace('trial_', '')
                
                # Step back to find Strategy and Parameter Value
                # root: .../action/continuous/deterministic/random_nn/timestamp/trial_1
                parts = os.path.relpath(trial_folder, param_path).split(os.sep)
                
                # parts looks like: ['sub1', 'sub2', 'strategy', 'timestamp', 'trial_1']
                if len(parts) >= 3:
                    trial_part = parts[-1]
                    timestamp_part = parts[-2]
                    strategy = parts[-3]
                    sub_params = parts[:-3]
                    parameter_value = "_".join(sub_params) if sub_params else strategy
                else:
                    # Fallback for simpler structures
                    strategy = "unknown"
                    parameter_value = "unknown"

                # Look for summary.csv (handling the user's note about potential naming)
                # The user mentioned summary.csv might be repeated or slightly different
                summary_files = [f for f in files if f.endswith('summary.csv')]
                
                if not summary_files:
                    continue
                
                summary_path = os.path.join(trial_folder, summary_files[0])
                try:
                    df = pd.read_csv(summary_path)
                    
                    # Add our metadata columns to the start
                    df.insert(0, 'parameter', param)
                    df.insert(1, 'parameter_value', parameter_value)
                    df.insert(2, 'strategy', strategy)
                    df.insert(3, 'trial', trial_name)
                    
                    all_data.append(df)
                except Exception as e:
                    print(f"Error reading {summary_path}: {e}")

        # Combine and save if data was found
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            output_file = os.path.join(base_experiments_folder, f"{param}.csv")
            combined_df.to_csv(output_file, index=False)
            print(f"Saved consolidated data to: {output_file}")

# Usage
dir = r"C:\Users\HP\Desktop\Projects\navigation\10-lidar-goal-navigation\output\experiments"
consolidate_experiment_data(dir)