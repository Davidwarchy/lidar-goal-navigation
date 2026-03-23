# Spike-NN: Neuroevolution Strategy

This folder contains the Spike-NN (Neuroevolution) strategy implementation for the lidar-goal-navigation project.

## Overview

The Spike-NN strategy implements a genetic algorithm (neuroevolution) that evolves neural network populations to explore the environment. Multiple agents compete each generation, with the best performers selected for reproduction through natural selection. Unlike goal-based strategies, Spike-NN optimizes for **exploration coverage** - agents are rewarded for covering as much of the map as possible. 


## How It Works

1. **Population**: Multiple neural network agents compete each generation
2. **Energy System**: Each agent starts with predefined energy; each step costs 1 energy point
3. **Goal**: Success is measured by exploration coverage (>2% of map explored)
4. **Evolution**: Best performing agents are selected for reproduction through natural selection
5. **Fitness**: Based on coverage * 100 + distance_traveled * 0.5 - steps_taken * 0.05

## Architecture

- **Input**: 8 neurons (100 lidar rays downsampled to 8 inputs for better coverage)
- **Hidden**: 16 neurons with ReLU activation
- **Output**: 4 neurons (4 actions: forward, left, right, backward)

### Activation Functions
- Hidden layer: ReLU activation (`max(0, x)`)
- Output layer: Linear activation

## Genetic Algorithm Features

- **Selection**: Tournament selection (k=3) for parent selection
- **Crossover**: Uniform crossover of weight vectors
- **Mutation**: Gaussian noise with configurable rate (default 0.2) and magnitude (0.5)
- **Elitism**: Best performers preserved directly across generations (no modification)
- **He Initialization**: Weights initialized with He initialization for better training

## Fitness Metrics (Tracked per Generation)

| Metric                           | Description                                     |
| -------------------------------- | ----------------------------------------------- |
| Generation                       | Current iteration number                        |
| % of Generation Successful       | Percentage of agents with >2% coverage          |
| Average Success Path Length      | Average steps taken by successful agents        |
| Average Success Energy Remaining | Average energy left for successful agents       |
| Average Success Health           | Same as energy remaining                        |
| Average Distance to Reward       | Average total distance traveled by agents      |
| Best Fitness                     | Highest fitness score in the generation         |
| Best Coverage                    | Best exploration coverage achieved              |

## Running Spike-NN

From the project root:

```bash
python main.py --strategy spike_nn --max_steps 5000 --env 6.png
```

### Default Parameters

| Parameter         | Default | Description                           |
| ----------------- | ------- | ------------------------------------- |
| population_size   | 50      | Number of agents per generation      |
| generations      | 20      | Number of evolution iterations       |
| energy_per_agent  | 100     | Starting energy per agent            |
| elite_count       | 5       | Number of top performers to preserve |
| mutation_rate     | 0.2     | Probability of weight mutation        |
| num_rays          | 100     | Number of lidar rays (input)         |

### Custom Configuration

```bash
python main.py --strategy spike_nn --max_steps 5000 --env 6.png \
    --population-size 100 \
    --generations 50 \
    --energy-per-agent 150 \
    --elite-count 10 \
    --mutation-rate 0.3
```

## Output

Weights and metrics are saved to `spike_weights/`:

### Neural Network Weights
Each generation saves `gen_XXX.json` containing:

- Generation number
- Fitness score
- Exploration coverage percentage
- Neural network weights (flat vector)
- Success status
- Steps taken
- Total distance traveled

### Training Summary
After completion, `training_summary.json` is saved with:

- Total generations
- Final metrics
- Total steps across all generations
- Best coverage achieved

### CSV Metrics
`training_metrics.csv` contains all generation metrics in tabular format for analysis.

## Files

- `strategies/Spike_NN.py`: Main implementation containing:
  - `SimpleNeuralNetwork`: Feedforward neural network class with He initialization
  - `Agent`: Individual agent with energy, fitness, and exploration tracking
  - `GeneticAlgorithm`: Evolution operators (tournament selection, uniform crossover, mutation)
  - `SpikeNNGeneticStrategy`: Complete training loop with metrics tracking
  - `create_spike_nn_strategy`: Factory function for strategy creation

## Key Implementation Details

### Lidar Preprocessing
- 100 lidar rays are downsampled to 8 inputs for the neural network
- Normalized to [0, 1] range with small offset (+0.01) to prevent dead neurons

### Fitness Calculation
```
fitness = coverage * 100 + distance_traveled * 0.5 - steps_taken * 0.05
```

- **Coverage** (heavily weighted): Percentage of map explored
- **Distance**: Bonus for exploring far from starting position
- **Efficiency**: Penalty for using too many steps

### Success Criteria
An agent is considered "successful" if it explores more than 2% of the map.

## Configuration

The strategy can be configured with:

- `population_size`: Number of agents per generation (default: 20)
- `generations`: Number of evolution iterations (default: 5)
- `energy_per_agent`: Starting energy per agent (default: 100)
- `elite_count`: Number of top performers to preserve (default: 3)
- `mutation_rate`: Probability of weight mutation (default: 0.1)
- `weights_dir`: Directory to save weights (default: "spike_weights")
