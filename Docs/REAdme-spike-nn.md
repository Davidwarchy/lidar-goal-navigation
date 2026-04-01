# Spike-NN: Spiking Neural Network with Genetic Algorithm (Neuroevolution)

This folder contains the Spike-NN strategy implementation for the lidar-goal-navigation project. It uses a **Leaky Integrate-and-Fire (LIF) Spiking Neural Network** with neuroevolution to train agents to reach goals.

## Overview

The Spike-NN strategy implements a genetic algorithm (neuroevolution) that evolves spiking neural network populations to navigate toward goals. Multiple agents compete each generation, with the best performers (those that reach the goal) selected for reproduction. Agents have limited energy and must reach the goal before running out.

## How It Works

1. **Population**: Multiple spiking neural network agents compete each generation
2. **Energy System**: Each agent has limited energy; reaching the goal = survival
3. **Goal**: Success is measured by reaching the goal location before energy depletion
4. **Evolution**: Best performing agents (survivors) are selected for reproduction
5. **Extinction Logic**: If no agents survive a generation, evolution terminates early

## Architecture

### Leaky Integrate-and-Fire (LIF) Spiking Neurons

The network uses biologically-inspired spiking neurons:

- **Membrane Potential (V)**: Each neuron maintains a membrane potential that leaks over time
- **Firing**: A neuron fires (output=1) when potential exceeds threshold, then resets
- **Dynamics**: `V[t] = leak * V[t-1] + W @ x + b`

### Network Structure

- **Input**: `num_rays` neurons (lidar readings as rate-coded input)
- **Hidden**: 64 LIF neurons with leak=0.9, threshold=1.0
- **Output**: 4 action neurons (binary spikes → argmax for action)

### Input Encoding

- Lidar values normalized to [0, 1] range
- Poisson/rate encoding: spike probability = input value
- Network runs for 5 internal time-steps per observation to accumulate spike activity

## Genetic Algorithm Features

- **Selection**: Survivors (agents that reached goal) form the gene pool
- **Reproduction**: Offspring inherit weights from survivors with mutation
- **Mutation**: Gaussian noise with configurable rate (default 0.2) and magnitude (default 0.5)
- **Elitism**: Top 10% of survivors preserved directly across generations
- **Extinction Detection**: Training stops if no survivors in a generation
- **Multi-Trial Support**: Run multiple independent trials (default 3)
- **Action Selection**: Choose between deterministic (argmax) or stochastic (categorical/softmax) policies

## Fitness Metrics (Tracked per Generation)

| Metric                   | Description                                       |
| ------------------------ | ------------------------------------------------- |
| Run                     | Trial number                                      |
| Generation              | Current iteration number                          |
| % Success               | Percentage of agents that reached the goal        |
| Avg Path Length         | Average steps taken by successful agents         |
| Avg Energy Remaining    | Average energy left for successful agents        |
| Avg Health Remaining    | Same as energy remaining                          |
| Avg Distance to Goal    | Final distance to goal for all agents            |
| Gen Total Steps         | Total steps executed in this generation          |
| Gen Time (s)            | Time taken for this generation                   |
| ms/Step                 | Milliseconds per step (performance metric)        |

## Running Spike-NN

From the project root:

```bash
python main.py --strategy spike_nn --max_steps 5000 --env 6.png
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--strategy` | str | required | Strategy name (use "spike_nn") |
| `--max_steps` | int | 1000 | Max steps per agent episode |
| `--env` | str | "6.png" | Environment map image |
| `--action_method` | str | "argmax" | Action selection method (argmax/softmax/categorical) |
| `--action_distribution` | str | "deterministic" | Policy type: deterministic or stochastic |
| `--trials` | int | 20 | Number of independent trials (for spike_nn: num_trials) |
| `--generations` | int | 50 | Generations per trial (for spike_nn: generations) |
| `--population` | int | 1000 | Population size (for spike_nn: population_size) |
| `--mutation_rate` | float | 0.1 | Mutation rate for genetic algorithm |
| `--render` | flag | false | Enable rendering visualization |

### Action Selection via Arguments

The strategy supports three action selection methods controlled by `--action_method`:

```bash
# Deterministic (greedy) - default
python main.py --strategy spike_nn --action_method argmax --action_distribution deterministic

# Softmax sampling (stochastic)
python main.py --strategy spike_nn --action_method softmax --action_distribution stochastic

# Categorical sampling (stochastic)
python main.py --strategy spike_nn --action_method categorical --action_distribution stochastic
```

### Default Parameters

| Parameter              | Default | Description                              |
| ---------------------- | ------- | ---------------------------------------- |
| population_size       | 50      | Number of agents per generation         |
| generations           | 20      | Number of evolution iterations          |
| num_trials            | 3       | Number of independent trials to run     |
| mutation_rate         | 0.2     | Probability of weight mutation          |
| mutation_mag          | 0.5     | Magnitude of mutation noise             |
| action_distribution   | deterministic | Policy type: deterministic or stochastic |
| weights_dir           | spike_weights | Directory to save weights        |

### Custom Configuration

```bash
python main.py --strategy spike_nn --max_steps 5000 --env 6.png \
    --population 100 \
    --generations 50 \
    --trials 5 \
    --mutation-rate 0.3 \
    --action_method softmax \
    --action_distribution stochastic
```

## Output

Weights and metrics are saved to `spike_weights/`:

### Neural Network Weights

Each generation saves `trial_X_gen_Y/best_survivor.json` containing:
- Flat weight vector for the neural network
- Can be loaded to reproduce best-performing agents

### CSV Metrics

`trial_metrics.csv` contains all generation metrics in tabular format:
- Success rate per generation
- Path length statistics
- Timing performance (ms/step)
- Extinction events

## Files

- [`strategies/spiking.py`](../strategies/spiking.py): Main implementation containing:
  - `LIFNeuronLayer`: Leaky Integrate-and-Fire neuron layer implementation
  - `SpikingNeuralNetwork`: Two-layer LIF spiking network
  - `SpikeNNGeneticStrategy`: Complete training loop with metrics tracking
  - `_reproduce()`: Reproduction with elitism and mutation
  - `_select_action()`: Action selection (deterministic or stochastic)

## Key Implementation Details

### LIF Neuron Dynamics

```
# Each timestep:
V = leak * V + W @ input + b    # Integrate with leak
spikes = (V >= threshold)       # Fire if threshold exceeded
V = np.where(spikes, reset_val, V)  # Reset fired neurons
```

### Forward Pass

1. Normalize lidar input to [0, 1] range
2. For 5 time-steps:
   - Poisson-encode input as random spikes
   - Forward through hidden LIF layer
   - Forward through output LIF layer
   - Accumulate spike counts
3. Return spike counts as action preferences

### Action Selection

- **Deterministic**: Select action with highest spike count (argmax)
- **Stochastic**: Sample from softmax distribution over spike counts

### Action Selection

The strategy supports two action selection modes controlled by `--action_distribution`:

| Mode | Implementation | Description |
|------|----------------|-------------|
| `deterministic` | **Argmax** | Select action with highest spike count (greedy) |
| `stochastic` | **Categorical/Softmax** | Sample from softmax distribution over spike counts |

**Deterministic (via `--action_distribution deterministic`)**:
```python
action = np.argmax(output)  # Greedy selection
```

**Stochastic (via `--action_distribution stochastic`)**:
```python
exp_output = np.exp(output - np.max(output))  # numerical stability
probs = exp_output / np.sum(exp_output)       # softmax
action = np.random.choice(len(output), p=probs)  # sample
```

The `--action_method` argument is also available but currently defaults to "argmax".

### Fitness Criteria

An agent is "successful" (survivor) if it reaches the goal location before energy depletion.

## Configuration

The strategy can be configured with:

- `population_size`: Number of agents per generation (default: 50)
- `generations`: Number of evolution iterations (default: 20)
- `num_trials`: Number of independent trials (default: 3)
- `mutation_rate`: Probability of weight mutation (default: 0.2)
- `mutation_mag`: Magnitude of mutation noise (default: 0.5)
- `action_distribution`: "deterministic" (argmax) or "stochastic" (categorical) (default: "deterministic")
- `weights_dir`: Directory to save weights (default: "spike_weights")
