"""
Spiking Neural Network (SNN) with Genetic Algorithm (Neuroevolution)

This strategy implements a population-based neuroevolution system where
multiple neural network agents compete in each generation. The best performing
agents are selected for reproduction through natural selection.

Key Features:
- Population of neural networks (not just one)
- Genetic algorithm: selection, crossover, mutation
- Energy system: each step costs 1 energy point

Neural Network:
- Leaky Integrate-and-Fire (LIF) Spiking Neural Network
- Neurons accumulate membrane potential over time
- Fire binary spikes when potential crosses threshold
- Membrane potential leaks toward resting state each timestep
- Spike traces used for weight representation in GA genome

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
# Leaky Integrate-and-Fire Spiking Neural Network
# ---------------------------------------------------------------------------

class LIFNeuronLayer:
    """
    A layer of Leaky Integrate-and-Fire (LIF) neurons.

    Each neuron maintains a membrane potential V.  On every call to
    `forward()` the potential is updated according to:

        V[t] = leak * V[t-1] + W @ x[t] + b

    A neuron fires (output = 1) when V >= threshold, after which its
    potential is reset to `reset_potential`.

    Parameters
    ----------
    n_in  : number of pre-synaptic inputs
    n_out : number of neurons in this layer
    leak  : membrane leak factor  (0 < leak < 1); default 0.9
    threshold : firing threshold; default 1.0
    reset_potential : post-spike reset value; default 0.0
    """

    def __init__(self, n_in: int, n_out: int,
                 leak: float = 0.9,
                 threshold: float = 1.0,
                 reset_potential: float = 0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.leak = leak
        self.threshold = threshold
        self.reset_potential = reset_potential

        # Synaptic weight matrix  (n_in x n_out) and bias (n_out,)
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)

        # Runtime state (reset between episodes)
        self.V = np.zeros(n_out)   # membrane potential

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Process one time-step of input spikes / rates.

        Parameters
        ----------
        x : (n_in,) array – input activations (spikes or rates)

        Returns
        -------
        spikes : (n_out,) binary array  (1 = fired, 0 = silent)
        """
        # Leak + integrate
        self.V = self.leak * self.V + x @ self.W + self.b

        # Fire
        spikes = (self.V >= self.threshold).astype(np.float32)

        # Reset fired neurons
        self.V = np.where(spikes == 1.0, self.reset_potential, self.V)

        return spikes

    def reset_state(self):
        """Reset membrane potentials to zero (call between episodes)."""
        self.V[:] = 0.0

    # ------------------------------------------------------------------
    # Genome helpers (flat weight vector for the GA)
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.W.flatten(), self.b])

    def set_weights(self, weights: np.ndarray):
        size_W = self.n_in * self.n_out
        self.W = weights[:size_W].reshape(self.n_in, self.n_out)
        self.b = weights[size_W:]

class SpikingNeuralNetwork:
    """
    Two-layer Leaky Integrate-and-Fire Spiking Neural Network.

    Architecture
    ------------
    Input  (n_in  neurons, rate-coded)
      ↓  LIF hidden layer
    Hidden (hidden_size neurons, binary spikes)
      ↓  LIF output layer
    Output (n_out neurons, binary spikes → argmax action)

    The network is run for `n_steps` internal time-steps per call so that
    spiking dynamics can accumulate meaningful activity from a single
    observation vector.

    Parameters
    ----------
    input_size  : dimensionality of the observation vector
    hidden_size : number of hidden LIF neurons
    output_size : number of action neurons
    n_steps     : internal simulation steps per observation (default 5)
    leak        : membrane leak for all layers (default 0.9)
    threshold   : firing threshold for all layers (default 1.0)
    """

    def __init__(self, input_size=100, hidden_size=64, output_size=4, n_steps=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_steps = n_steps
        self.hidden_layer = LIFNeuronLayer(input_size, hidden_size)
        self.output_layer = LIFNeuronLayer(hidden_size, output_size)
    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the SNN for `n_steps` time-steps and return accumulated output
        spike counts (one value per action neuron).

        Parameters
        ----------
        inputs : (input_size,) float array, values in [0, 1]
                 Treated as a constant Poisson rate stimulus.

        Returns
        -------
        spike_counts : (output_size,) float array
                       Larger value → neuron fired more → preferred action.
        """
        # Encode continuous input as Bernoulli spikes each time-step
        spike_counts = np.zeros(self.output_size)

        for _ in range(self.n_steps):
            # Poisson / rate encoding: spike with probability = input value
            encoded = (np.random.rand(self.input_size) < inputs).astype(np.float32)

            h_spikes = self.hidden_layer.forward(encoded)
            o_spikes = self.output_layer.forward(h_spikes)
            spike_counts += o_spikes

        return spike_counts

    def reset_state(self):
        """Reset all neuron membrane potentials (call between episodes)."""
        self.hidden_layer.reset_state()
        self.output_layer.reset_state()

    # ------------------------------------------------------------------
    # Genome helpers
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            self.hidden_layer.get_weights(),
            self.output_layer.get_weights()
        ])

    def set_weights(self, weights: np.ndarray):
        n_h = (self.input_size * self.hidden_size) + self.hidden_size
        self.hidden_layer.set_weights(weights[:n_h])
        self.output_layer.set_weights(weights[n_h:])

# ---------------------------------------------------------------------------
# Neuroevolution Strategy with Extinction Logic
# ---------------------------------------------------------------------------

class SpikeNNGeneticStrategy(BaseStrategy):
    def __init__(self, population_size=50, generations=20, num_trials=3, 
                 mutation_rate=0.2, mutation_mag=0.5, weights_dir="spike_weights"):
        params = {
            "population_size": population_size,
            "generations": generations,
            "num_trials": num_trials,
            "mutation_rate": mutation_rate,
            "weights_dir": weights_dir
        }
        super().__init__("spike_nn", params)
        self.pop_size = population_size
        self.max_gens = generations
        self.num_trials = num_trials
        self.mutation_rate = mutation_rate
        self.mutation_mag = mutation_mag
        self.weights_dir = weights_dir
        os.makedirs(self.weights_dir, exist_ok=True)

    def run(self, env):
        import time
        from tqdm import tqdm
        
        results_file = os.path.join(env.output_dir, "trial_metrics.csv")
        headers = ["Run", "Generation", "% Success", "Avg Path Length", 
                   "Avg Energy Rem", "Avg Health Rem", "Avg Dist to Reward", 
                   "Gen Total Steps", "Gen Time (s)", "ms/Step", "Weights Directory"]
        
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for trial in range(1, self.num_trials + 1):
                # Outer progress bar for Generations
                gen_pbar = tqdm(range(1, self.max_gens + 1), 
                                desc=f"Trial {trial}/{self.num_trials}", 
                                unit="gen", position=0)
                
                population = [SpikingNeuralNetwork(input_size=env.num_rays) for _ in range(self.pop_size)]
                
                for gen in gen_pbar:
                    survivors = []
                    gen_stats = {"path_lengths": [], "energies": [], "healths": [], "dist_to_reward": []}
                    
                    # Performance Tracking for this generation
                    gen_start_time = time.time()
                    gen_total_steps = 0

                    # Inner progress bar for Individuals
                    ind_pbar = tqdm(enumerate(population), total=self.pop_size, 
                                    desc=f"  Gen {gen} Progress", leave=False, 
                                    unit="ind", position=1)

                    for idx, net in ind_pbar:
                        obs = env.reset()
                        net.reset_state()
                        done = False
                        
                        while not done:
                            normalized_obs = obs / env.ray_length
                            output = net.forward(normalized_obs)
                            action = np.argmax(output)
                            obs, reward, done, info = env.step(action)
                            
                            # Increment total steps for the whole generation
                            gen_total_steps += 1
                            
                            if env.render_flag: env.render()

                        if info.get("goal_reached", False):
                            survivors.append(net)
                            gen_stats["path_lengths"].append(env.current_step)
                            gen_stats["energies"].append(info["energy"])
                            gen_stats["healths"].append(info["health"])
                            
                        dx, dy = env.robot_x - env.goal_x, env.robot_y - env.goal_y
                        dist = np.sqrt(dx*dx + dy*dy)
                        gen_stats["dist_to_reward"].append(dist)
                        
                        ind_pbar.set_postfix({"Found": len(survivors), "Dist": f"{dist:.1f}"})

                    # Calculate timing stats
                    gen_duration = time.time() - gen_start_time
                    ms_per_step = (gen_duration * 1000) / gen_total_steps if gen_total_steps > 0 else 0

                    success_rate = len(survivors) / self.pop_size
                    avg_path = np.mean(gen_stats["path_lengths"]) if survivors else 0
                    avg_dist = np.mean(gen_stats["dist_to_reward"])

                    # Update outer bar with Success Rate and Performance
                    gen_pbar.set_postfix({
                        "SR": f"{success_rate*100:.1f}%",
                        "ms/st": f"{ms_per_step:.2f}",
                        "Steps": gen_total_steps
                    })

                    # Save weights and Log Data
                    gen_dir = os.path.join(self.weights_dir, f"trial_{trial}_gen_{gen}")
                    os.makedirs(gen_dir, exist_ok=True)
                    if survivors:
                        with open(os.path.join(gen_dir, "best_survivor.json"), 'w') as wf:
                            json.dump({"weights": survivors[0].get_weights().tolist()}, wf)

                    row = [trial, gen, f"{success_rate*100:.2f}%", f"{avg_path:.2f}", 
                           f"{np.mean(gen_stats['energies']) if survivors else 0:.2f}", 
                           f"{np.mean(gen_stats['healths']) if survivors else 0:.2f}", 
                           f"{avg_dist:.2f}", gen_total_steps, f"{gen_duration:.2f}", 
                           f"{ms_per_step:.2f}", gen_dir]
                    writer.writerow(row)
                    f.flush()

                    if not survivors:
                        gen_pbar.write(f"[EXTINCT] Trial {trial} Gen {gen} - No survivors.")
                        break
                    
                    population = self._reproduce(survivors)

        return 0, 0
    
    def _reproduce(self, survivors):
        new_population = []
        # Keep survivors (Elitism) 
        new_population.extend(survivors[:max(1, self.pop_size // 10)])
        
        while len(new_population) < self.pop_size:
            parent = random.choice(survivors)
            child_weights = parent.get_weights().copy()
            
            # Mutation: Gaussian noise 
            mask = np.random.random(len(child_weights)) < self.mutation_rate
            child_weights[mask] += np.random.randn(np.sum(mask)) * self.mutation_mag
            
            child = SpikingNeuralNetwork(input_size=parent.input_size)
            child.set_weights(child_weights)
            new_population.append(child)
            
        return new_population