"""
Feedforward Neural Network (FFNN) with Genetic Algorithm (Neuroevolution)

This strategy implements a population-based neuroevolution system where
multiple neural network agents compete in each generation. The best performing
agents are selected for reproduction through natural selection.

Key Features:
- Population of feedforward neural networks (not just one)
- Genetic algorithm: selection, crossover, mutation
- Energy system: each step costs 1 energy point
- Exploration-based fitness: reward based on coverage and distance traveled

Neural Network:
- Standard Feedforward Neural Network with ReLU hidden activations
- Softmax output layer for action probability distribution
- Deterministic argmax action selection at inference time
- No temporal state — each observation is processed independently

Fitness Metrics Tracked:
- Generation: Current iteration number
- % of Generation Successful: Percentage of agents that survived (reached goal)
- Average Success Path Length: Average steps taken by successful agents
- Average Success Energy Remaining: Average energy left for successful agents
- Average Success Health: Same as energy remaining
- Average Distance to Reward: Average distance traveled
- Weights Directory: Location where weights are saved
"""

import numpy as np
import os
import json
import csv
import random
from datetime import datetime
from .base_strategy import BaseStrategy
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Feedforward Neural Network
# ---------------------------------------------------------------------------

class DenseLayer:
    """
    A fully-connected (dense) layer with a configurable activation function.

    Forward pass:
        z = x @ W + b
        output = activation(z)

    Parameters
    ----------
    n_in       : number of input features
    n_out      : number of neurons (output features)
    activation : one of 'relu', 'tanh', 'sigmoid', 'softmax', or 'linear'
    """

    ACTIVATIONS = ('relu', 'tanh', 'sigmoid', 'softmax', 'linear')

    def __init__(self, n_in: int, n_out: int, activation: str = 'relu'):
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"activation must be one of {self.ACTIVATIONS}")

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        # He initialisation for ReLU; Xavier for others
        if activation == 'relu':
            scale = np.sqrt(2.0 / n_in)
        else:
            scale = np.sqrt(1.0 / n_in)

        self.W = np.random.randn(n_in, n_out) * scale
        self.b = np.zeros(n_out)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the layer output for a single observation vector.

        Parameters
        ----------
        x : (n_in,) float array

        Returns
        -------
        out : (n_out,) float array after activation
        """
        z = x @ self.W + self.b

        if self.activation == 'relu':
            return np.maximum(0.0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        elif self.activation == 'softmax':
            z_stable = z - z.max()          # numerical stability
            exp_z = np.exp(z_stable)
            return exp_z / exp_z.sum()
        else:  # linear
            return z

    # ------------------------------------------------------------------
    # Genome helpers (flat weight vector for the GA)
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.W.flatten(), self.b])

    def set_weights(self, weights: np.ndarray):
        size_W = self.n_in * self.n_out
        self.W = weights[:size_W].reshape(self.n_in, self.n_out)
        self.b = weights[size_W:]


class FeedforwardNeuralNetwork:
    """
    Two-hidden-layer feedforward neural network for discrete action selection.

    Architecture
    ------------
    Input  (input_size,)
      ↓  Dense + ReLU
    Hidden-1 (hidden_size,)
      ↓  Dense + ReLU
    Hidden-2 (hidden_size // 2,)
      ↓  Dense + Softmax
    Output (output_size,)  →  argmax → discrete action

    Parameters
    ----------
    input_size  : dimensionality of the observation vector
    hidden_size : width of the first hidden layer
                  (second hidden layer uses hidden_size // 2 neurons)
    output_size : number of discrete actions
    """

    def __init__(self, input_size: int = 100,
                 hidden_size: int = 64,
                 output_size: int = 4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        h2 = max(hidden_size // 2, output_size)

        self.layer1 = DenseLayer(input_size, hidden_size, activation='relu')
        self.layer2 = DenseLayer(hidden_size, h2,         activation='relu')
        self.layer3 = DenseLayer(h2,          output_size, activation='softmax')

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run a single forward pass and return action probabilities.

        Parameters
        ----------
        inputs : (input_size,) float array, values in [0, 1]

        Returns
        -------
        probs : (output_size,) float array summing to 1.0
                Larger value → more preferred action.
        """
        x = self.layer1.forward(inputs)
        x = self.layer2.forward(x)
        return self.layer3.forward(x)

    def reset_state(self):
        """No-op — included for API compatibility with the SNN version."""
        pass

    # ------------------------------------------------------------------
    # Genome helpers
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            self.layer1.get_weights(),
            self.layer2.get_weights(),
            self.layer3.get_weights(),
        ])

    def set_weights(self, weights: np.ndarray):
        n1 = self.layer1.n_in * self.layer1.n_out + self.layer1.n_out
        n2 = self.layer2.n_in * self.layer2.n_out + self.layer2.n_out

        self.layer1.set_weights(weights[:n1])
        self.layer2.set_weights(weights[n1:n1 + n2])
        self.layer3.set_weights(weights[n1 + n2:])


# ---------------------------------------------------------------------------
# Neuroevolution Strategy with Extinction Logic
# ---------------------------------------------------------------------------

class RNNGeneticStrategy(BaseStrategy):
    """
    RNN - Random Neural Network 

    Genetic-algorithm neuroevolution strategy using a standard feedforward
    neural network (FFNN) in place of the spiking LIF network.

    Parameters
    ----------
    population_size : number of agents per generation
    generations     : maximum number of generations per trial
    num_trials      : number of independent trials to run
    mutation_rate   : probability of mutating each weight
    mutation_mag    : standard deviation of Gaussian mutation noise
    weights_dir     : directory where per-generation weights are saved
    hidden_size     : width of the first hidden layer in each network
    """

    def __init__(self, population_size: int = 50,
                 generations: int = 20,
                 num_trials: int = 3,
                 mutation_rate: float = 0.2,
                 mutation_mag: float = 0.5,
                 load_weights: bool = False,
                 weights_dir: str = "ffnn_weights",
                 hidden_size: int = 64):
        params = {
            "population_size": population_size,
            "generations": generations,
            "num_trials": num_trials,
            "mutation_rate": mutation_rate,
            "mutation_mag": mutation_mag,
            "weights_dir": weights_dir,
            "hidden_size": hidden_size,
        }
        super().__init__("ffnn_genetic", params)

        self.pop_size = population_size
        self.max_gens = generations
        self.num_trials = num_trials
        self.mutation_rate = mutation_rate
        self.mutation_mag = mutation_mag
        self.weights_dir = weights_dir
        self.hidden_size = hidden_size

        os.makedirs(self.weights_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, env):
        import time

        results_file = os.path.join(env.output_dir, "trial_metrics.csv")
        headers = [
            "Run", "Generation", "% Success",
            "Avg Path Length", "Avg Energy Rem", "Avg Health Rem",
            "Avg Dist to Reward", "Gen Total Steps",
            "Gen Time (s)", "ms/Step", "Weights Directory",
        ]

        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for trial in range(1, self.num_trials + 1):

                gen_pbar = tqdm(
                    range(1, self.max_gens + 1),
                    desc=f"Trial {trial}/{self.num_trials}",
                    unit="gen",
                    position=0,
                )

                # Initialise population
                population = [
                    FeedforwardNeuralNetwork(
                        input_size=env.num_rays,
                        hidden_size=self.hidden_size,
                    )
                    for _ in range(self.pop_size)
                ]

                for gen in gen_pbar:
                    survivors = []
                    gen_stats = {
                        "path_lengths": [],
                        "energies": [],
                        "healths": [],
                        "dist_to_reward": [],
                    }

                    gen_start_time = time.time()
                    gen_total_steps = 0

                    ind_pbar = tqdm(
                        enumerate(population),
                        total=self.pop_size,
                        desc=f"  Gen {gen} Progress",
                        leave=False,
                        unit="ind",
                        position=1,
                    )

                    for idx, net in ind_pbar:
                        obs = env.reset()
                        net.reset_state()   # no-op for FFNN; kept for API parity
                        done = False

                        while not done:
                            normalized_obs = obs / env.ray_length

                            # Forward pass → action probabilities → greedy action
                            action_probs = net.forward(normalized_obs)
                            action = int(np.argmax(action_probs))

                            obs, reward, done, info = env.step(action)
                            gen_total_steps += 1

                            if env.render_flag:
                                env.render()

                        if info.get("goal_reached", False):
                            survivors.append(net)
                            gen_stats["path_lengths"].append(env.current_step)
                            gen_stats["energies"].append(info["energy"])
                            gen_stats["healths"].append(info["health"])

                        dx = env.robot_x - env.goal_x
                        dy = env.robot_y - env.goal_y
                        dist = np.sqrt(dx * dx + dy * dy)
                        gen_stats["dist_to_reward"].append(dist)

                        ind_pbar.set_postfix({
                            "Found": len(survivors),
                            "Dist": f"{dist:.1f}",
                        })

                    # ── timing ──────────────────────────────────────────────
                    gen_duration = time.time() - gen_start_time
                    ms_per_step = (
                        (gen_duration * 1000) / gen_total_steps
                        if gen_total_steps > 0 else 0
                    )

                    success_rate = len(survivors) / self.pop_size
                    avg_path = (
                        np.mean(gen_stats["path_lengths"]) if survivors else 0
                    )
                    avg_dist = np.mean(gen_stats["dist_to_reward"])

                    gen_pbar.set_postfix({
                        "SR": f"{success_rate * 100:.1f}%",
                        "ms/st": f"{ms_per_step:.2f}",
                        "Steps": gen_total_steps,
                    })

                    # ── save weights ────────────────────────────────────────
                    # gen_dir = os.path.join(
                    #     self.weights_dir, f"trial_{trial}_gen_{gen}"
                    # )
                    # os.makedirs(gen_dir, exist_ok=True)
                    # if survivors:
                    #     with open(
                    #         os.path.join(gen_dir, "best_survivor.json"), 'w'
                    #     ) as wf:
                    #         json.dump(
                    #             {"weights": survivors[0].get_weights().tolist()},
                    #             wf,
                    #         )

                    # ── log row ─────────────────────────────────────────────
                    row = [
                        trial, gen,
                        f"{success_rate * 100:.2f}%",
                        f"{avg_path:.2f}",
                        f"{np.mean(gen_stats['energies']) if survivors else 0:.2f}",
                        f"{np.mean(gen_stats['healths'])  if survivors else 0:.2f}",
                        f"{avg_dist:.2f}",
                        gen_total_steps,
                        f"{gen_duration:.2f}",
                        f"{ms_per_step:.2f}",
                        None,
                    ]
                    writer.writerow(row)
                    f.flush()

                    # ── extinction check ─────────────────────────────────────
                    if not survivors:
                        gen_pbar.write(
                            f"[EXTINCT] Trial {trial} Gen {gen} — No survivors."
                        )
                        break

                    population = self._reproduce(survivors)

        return 0, 0

    # ------------------------------------------------------------------
    # Reproduction: elitism + mutation
    # ------------------------------------------------------------------

    def _reproduce(self, survivors: list) -> list:
        """
        Build the next generation from survivors.

        Strategy
        --------
        1. Elitism  — top 10 % of survivors carry over unchanged.
        2. Mutation — remaining slots filled by randomly chosen survivors
                      whose weights are perturbed with Gaussian noise.

        Parameters
        ----------
        survivors : list of FeedforwardNeuralNetwork instances that reached
                    the goal this generation.

        Returns
        -------
        new_population : list of FeedforwardNeuralNetwork (length = pop_size)
        """
        new_population: list = []

        # Elites
        n_elite = max(1, self.pop_size // 10)
        new_population.extend(survivors[:n_elite])

        # Fill remainder via mutation
        while len(new_population) < self.pop_size:
            parent = random.choice(survivors)
            child_weights = parent.get_weights().copy()

            # Gaussian perturbation on a random subset of weights
            mask = np.random.random(len(child_weights)) < self.mutation_rate
            child_weights[mask] += (
                np.random.randn(mask.sum()) * self.mutation_mag
            )

            child = FeedforwardNeuralNetwork(
                input_size=parent.input_size,
                hidden_size=parent.hidden_size,
                output_size=parent.output_size,
            )
            child.set_weights(child_weights)
            new_population.append(child)

        return new_population