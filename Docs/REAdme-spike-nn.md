# Spike-NN: Neuroevolution Strategy

This folder contains the Spike-NN (Neuroevolution) strategy implementation for the lidar-goal-navigation project.

## Overview

The Spike-NN strategy implements a genetic algorithm (neuroevolution) that evolves neural network populations to navigate to a goal. Multiple agents compete each generation, with the best performers selected for reproduction through natural selection.

## How It Works

1. **Population**: Multiple neural network agents compete each generation
2. **Energy System**: Each agent starts with predefined energy; each step costs 1 energy point
3. **Goal**: Success is reaching within 2 units of the reward/goal
4. **Evolution**: Best performing agents are selected for reproduction through natural selection

## Architecture

- **Input**: 4 neurons (simplified lidar: front, left, right, back)
- **Hidden**: 8 neurons with tanh activation
- **Output**: 2 neurons (left/right turn decisions)

## Genetic Algorithm Features

- **Selection**: Tournament selection
- **Crossover**: Single-point crossover of weight vectors
- **Mutation**: Random weight changes with configurable rate
- **Elitism**: Best performers preserved across generations

## Fitness Metrics (Tracked per Generation)

| Metric | Description |
|--------|-------------|
| Generation | Current iteration number |
| % of Generation Successful | Percentage of agents that reached the goal |
| Average Success Path Length | Average steps taken by successful agents |
| Average Success Energy Remaining | Average energy left for successful agents |
| Average Success Health | Same as energy remaining |
| Average Distance to Reward | Average distance to goal at end of episode |
| Weights Directory | Location where neural network weights are saved |

## Running Spike-NN

From the project root:

```bash
python main.py --strategy spike_nn --max_steps 5000 --env 6.png
```

## Output

Weights are saved to `spike_weights/gen_XXX.json` for each generation, containing:
- Generation number
- Fitness score
- Neural network weights
- Success status
- Steps taken
- Energy remaining

A training summary is saved to `spike_weights/training_summary.json` after completion.

## Files

- `spike-nn.py`: Main implementation containing:
  - `SimpleNeuralNetwork`: Feedforward neural network class
  - `Agent`: Individual agent with energy and fitness tracking
  - `GeneticAlgorithm`: Evolution operators (selection, crossover, mutation)
  - `SpikeNNGeneticStrategy`: Complete training loop

## Configuration

The strategy can be configured with:
- `population_size`: Number of agents per generation (default: 20)
- `generations`: Number of evolution iterations (default: 5)
- `energy_per_agent`: Starting energy per agent (default: 100)
- `elite_count`: Number of top performers to preserve (default: 3)
- `mutation_rate`: Probability of weight mutation (default: 0.1)
- `weights_dir`: Directory to save weights (default: "spike_weights")
