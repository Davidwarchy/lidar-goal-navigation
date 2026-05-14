import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('output/combined_results.csv')

# Merge spiking_nn into spiking
df['strategy'] = df['strategy'].replace('spiking_nn', 'spiking')

# Define trial identifier and get max generation
# identifier: source_file, parameter, parameter_value, strategy, trial
trial_id_cols = ['source_file', 'parameter', 'parameter_value', 'strategy', 'trial']
trial_lengths = df.groupby(trial_id_cols)['generation'].max().reset_index()
trial_lengths.columns = trial_id_cols + ['length']

# Stats calculation
total_trials = len(trial_lengths)
target_gen = 200

stats = trial_lengths.groupby('strategy')['length'].agg(['count', 'mean', 'median']).reset_index()
reached_200 = trial_lengths[trial_lengths['length'] >= target_gen].groupby('strategy').size().reset_index(name='reached_200')
stats = stats.merge(reached_200, on='strategy', how='left').fillna(0)
stats['% reached 200'] = (stats['reached_200'] / stats['count'] * 100).round(2)
stats.columns = ['Strategy', 'Total Trials', 'Mean Length', 'Median Length', 'Trials reached 200', '% reached 200']

# Conditional Probability Calculation
# P(S_g+1 | S_g)
gens = range(1, target_gen + 1)
survival_data = []
for g in gens:
    at_risk = trial_lengths[trial_lengths['length'] >= g]
    count_at_risk = len(at_risk)
    if count_at_risk > 0:
        survived = trial_lengths[trial_lengths['length'] >= g + 1]
        prob = len(survived) / count_at_risk if g < target_gen else 0
        survival_data.append({'generation': g, 'prob': prob, 'n': count_at_risk})

survival_df = pd.DataFrame(survival_data)

# Plots
# 1. Histogram of trial length by strategy
plt.figure(figsize=(15, 10))
g = sns.FacetGrid(trial_lengths, col="strategy", col_wrap=3, sharey=False, height=4)
g.map(sns.histplot, "length", bins=20, color='teal', edgecolor='black')
g.set_axis_labels("Generations Reached", "Frequency (Trials)")
g.set_titles("{col_name}")
plt.tight_layout()
plt.savefig('output/strategy_histograms.png')

# 2. Conditional Probability of Survival
plt.figure(figsize=(10, 6))
plt.plot(survival_df['generation'][:-1], survival_df['prob'][:-1], marker='.', color='crimson', linewidth=1)
plt.axhline(1.0, color='black', linestyle='--', alpha=0.3)
plt.title('Conditional Probability of Survival: $P(Gen \geq G+1 \mid Gen \geq G)$')
plt.xlabel('Generation (G)')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/strategy_conditional_survival.png')

print(f"Total Unique Trials: {total_trials}")
print("\nTrial Length Statistics (Strategy Merged):")
print(stats.to_string(index=False))