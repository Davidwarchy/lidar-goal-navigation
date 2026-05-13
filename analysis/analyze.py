import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Base directory where CSV files are located
EXPERIMENTS_DIR = "output/experiments"
OUTPUT_DIR = "output/experiments/plots-sd"
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to clean success_rate_percent
def clean_success_rate(val):
    if isinstance(val, str):
        return float(val.replace('%', ''))
    return val

# Find all CSV files in the experiments directory
csv_files = [f for f in os.listdir(EXPERIMENTS_DIR) if f.endswith('.csv')]

if not csv_files:
    print(f"No CSV files found in {EXPERIMENTS_DIR}")
    exit()

print(f"Found CSV files: {csv_files}")

# Load each CSV file
dataframes = {}
for csv_file in csv_files:
    file_path = os.path.join(EXPERIMENTS_DIR, csv_file)
    df = pd.read_csv(file_path)
    df['success_rate_percent'] = df['success_rate_percent'].apply(clean_success_rate)
    dataframes[csv_file] = df
    
    # Print info for first file to verify
    if csv_file == csv_files[0]:
        print(f"\nFirst few rows of {csv_file}:")
        print(df.head())
        print(df.info())

# Create plots - one figure per CSV file
for csv_file, df in dataframes.items():
    plt.figure(figsize=(10, 6))
    
    # Extract parameter name from filename (remove .csv)
    param_name = csv_file.replace('.csv', '')
    
    # Plot success_rate_percent over generations
    sns.lineplot(data=df, x='generation', y='success_rate_percent', hue='parameter_value', errorbar='sd')
    
    # Set titles and labels
    plt.title(f'Performance Comparison: {param_name}', fontsize=14)
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Generation')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title=param_name, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save figure in the experiments directory
    output_path = os.path.join(OUTPUT_DIR, f'{param_name}_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

print(f"\nDone! Plots saved to {OUTPUT_DIR}")