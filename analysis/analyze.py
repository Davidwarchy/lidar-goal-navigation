import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

files = [
    'action.csv', 'fitness_proxy.csv', 'goal_distance.csv', 'lif_steps.csv',
    'max_steps.csv', 'mutation_magnitude.csv', 'mutation_rate.csv',
    'population.csv', 'recombination.csv', 'robot_size.csv'
]

# Function to clean success_rate_percent
def clean_success_rate(val):
    if isinstance(val, str):
        return float(val.replace('%', ''))
    return val

dataframes = {}
for f in files:
    df = pd.read_csv(f)
    df['success_rate_percent'] = df['success_rate_percent'].apply(clean_success_rate)
    dataframes[f] = df

# Checking the first few rows of one to ensure cleaning worked
print(dataframes['action.csv'].head())
print(dataframes['action.csv'].info())

# Create two sets of plots (5 each) to ensure they are readable
fig1, axes1 = plt.subplots(5, 1, figsize=(10, 25))
fig2, axes2 = plt.subplots(5, 1, figsize=(10, 25))

all_axes = list(axes1) + list(axes2)

for i, f in enumerate(files):
    df = dataframes[f]
    ax = all_axes[i]
    
    # We'll plot success_rate_percent over generations
    # Use lineplot which automatically aggregates trials with error bands
    sns.lineplot(data=df, x='generation', y='success_rate_percent', hue='parameter_value', ax=ax)
    
    # Set titles and labels
    param_name = df['parameter'].iloc[0]
    ax.set_title(f'Performance Comparison: {param_name}', fontsize=14)
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Generation')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title=param_name, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
fig1.savefig('performance_comparison_1.png')
fig2.savefig('performance_comparison_2.png')