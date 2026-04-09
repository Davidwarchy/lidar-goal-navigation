"""
Vectorized Neural Networks with Genetic Algorithm (Neuroevolution)

This strategy implements a population-based neuroevolution system where
multiple neural network agents compete in each generation. The best performing
agents are selected for reproduction through natural selection.

Key Features:
- Population of neural networks (not just one)
- Genetic algorithm: selection, crossover, mutation
- Energy system: each step costs 1 energy point
- Pluggable neural network architecture (SNN or Feedforward)

Supported Network Types:
- "spiking": Leaky Integrate-and-Fire Spiking Neural Network
- "feedforward": Standard multi-layer perceptron

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
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Abstract Neural Network Interface
# ---------------------------------------------------------------------------

class NeuralNetwork(ABC):
    """Abstract base class for neural networks used in neuroevolution."""
    
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Process inputs and return output activations."""
        pass
    
    @abstractmethod
    def reset_state(self):
        """Reset any internal state (e.g., for RNNs/SNNs)."""
        pass
    
    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """Get flat array of all trainable weights."""
        pass
    
    @abstractmethod
    def set_weights(self, weights: np.ndarray):
        """Set weights from flat array."""
        pass
    
    @property
    @abstractmethod
    def input_size(self) -> int:
        """Return input dimension."""
        pass
    
    @property
    @abstractmethod
    def output_size(self) -> int:
        """Return output dimension."""
        pass


# ---------------------------------------------------------------------------
# Vectorized Feedforward Neural Network
# ---------------------------------------------------------------------------

class VectorFeedforwardNetwork(NeuralNetwork):
    """
    Vectorized standard multi-layer perceptron with ReLU activations.
    
    Architecture:
    Input -> Hidden Layer(s) (ReLU) -> Output Layer (Linear)
    
    All agents are evaluated in parallel using vectorized operations.
    """
    
    def __init__(self, num_envs: int, input_size: int, hidden_sizes: list, output_size: int):
        """
        Parameters:
        -----------
        num_envs : int
            Number of parallel agents
        input_size : int
            Dimensionality of input observations
        hidden_sizes : list of int
            Number of neurons in each hidden layer (e.g., [64, 32])
        output_size : int
            Number of output neurons (actions)
        """
        self._num_envs = num_envs
        self._input_size = input_size
        self._output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        # Build layer sizes
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights and biases for all agents
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU
            # Shape: (num_envs, layer_sizes[i], layer_sizes[i+1])
            w = np.random.randn(num_envs, layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((num_envs, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    @property
    def input_size(self) -> int:
        return self._input_size
    
    @property
    def output_size(self) -> int:
        return self._output_size
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass for all agents.
        
        Parameters:
        -----------
        inputs : (num_envs, input_size) array
        
        Returns:
        --------
        output : (num_envs, output_size) array
        """
        x = inputs  # Shape: (num_envs, input_size)
        
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            # For each agent: output = ReLU(input @ weight + bias)
            # Using einsum: (num_envs, input_dim) @ (num_envs, input_dim, output_dim) -> (num_envs, output_dim)
            x = np.maximum(0, np.einsum('bi,bij->bj', x, self.weights[i]) + self.biases[i])
        
        # Output layer (linear) - no ReLU
        x = np.einsum('bi,bij->bj', x, self.weights[-1]) + self.biases[-1]
        
        return x
    
    def reset_state(self):
        """Feedforward networks have no state to reset."""
        pass
    
    def get_weights(self) -> np.ndarray:
        """Get flat array of all trainable weights for a single agent (for compatibility)."""
        all_weights = []
        for w, b in zip(self.weights, self.biases):
            all_weights.append(w[0].flatten())
            all_weights.append(b[0].flatten())
        return np.concatenate(all_weights)
    
    def set_weights(self, weights: np.ndarray):
        """Set weights from flat array for a single agent (for compatibility)."""
        idx = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            w_size = w.shape[1] * w.shape[2]
            b_size = b.shape[1]
            
            # Set all agents to the same weights
            w_flat = weights[idx:idx + w_size]
            self.weights[i] = w_flat.reshape(1, w.shape[1], w.shape[2]).repeat(self.num_envs, axis=0)
            idx += w_size
            
            b_flat = weights[idx:idx + b_size]
            self.biases[i] = b_flat.reshape(1, b.shape[1]).repeat(self.num_envs, axis=0)
            idx += b_size
    
    def get_genomes(self) -> list:
        """Get list of flat weight arrays for all agents."""
        genomes = []
        for agent_idx in range(self.num_envs):
            all_weights = []
            for w, b in zip(self.weights, self.biases):
                all_weights.append(w[agent_idx].flatten())
                all_weights.append(b[agent_idx].flatten())
            genomes.append(np.concatenate(all_weights))
        return genomes
    
    def set_genomes(self, genomes: list):
        """Set weights from list of flat arrays."""
        for agent_idx, genome in enumerate(genomes):
            idx = 0
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                w_size = w.shape[1] * w.shape[2]
                b_size = b.shape[1]
                
                self.weights[i][agent_idx] = genome[idx:idx + w_size].reshape(w.shape[1], w.shape[2])
                idx += w_size
                
                self.biases[i][agent_idx] = genome[idx:idx + b_size]
                idx += b_size


# ---------------------------------------------------------------------------
# Vectorized LIF Layer
# ---------------------------------------------------------------------------

class VectorLIFLayer:
    def __init__(self, num_envs, n_in, n_out, leak=0.9, threshold=1.0):
        self.num_envs = num_envs
        self.n_in = n_in
        self.n_out = n_out
        self.leak = leak
        self.threshold = threshold
        
        # Unique weights for every robot: (num_envs, n_in, n_out)
        limit = np.sqrt(2.0 / n_in)
        self.W = np.random.randn(num_envs, n_in, n_out) * limit
        self.b = np.zeros((num_envs, n_out))
        self.V = np.zeros((num_envs, n_out))

    def reset_state(self):
        self.V.fill(0.0)

    def forward(self, x):
        # x shape: (num_envs, n_in)
        # Vectorized batch matrix multiplication for unique weights per env
        z = np.einsum('bi,bij->bj', x, self.W) + self.b
        self.V = self.leak * self.V + z
        
        spikes = (self.V >= self.threshold).astype(np.float32)
        self.V[spikes == 1.0] = 0.0  # Reset fired neurons
        return spikes

    def get_weights(self):
        # Returns a list of flat arrays for the GA
        return [np.concatenate([self.W[i].flatten(), self.b[i]]) for i in range(self.num_envs)]

    def set_weights(self, flat_weights_list):
        for i, weights in enumerate(flat_weights_list):
            size_W = self.n_in * self.n_out
            self.W[i] = weights[:size_W].reshape(self.n_in, self.n_out)
            self.b[i] = weights[size_W:]


# ---------------------------------------------------------------------------
# Vectorized Spiking Neural Network
# ---------------------------------------------------------------------------

class VectorSpikingNetwork(NeuralNetwork):
    """
    Two-layer vectorized Leaky Integrate-and-Fire Spiking Neural Network.
    
    Architecture:
    Input  (n_in  neurons, rate-coded)
      ↓  LIF hidden layer
    Hidden (hidden_size neurons, binary spikes)
      ↓  LIF output layer
    Output (n_out neurons, binary spikes → argmax action)

    The network is run for `n_steps` internal time-steps per call so that
    spiking dynamics can accumulate meaningful activity from a single
    observation vector.
    """
    
    def __init__(self, num_envs, input_size=100, hidden_size=64, output_size=4, n_steps=5):
        self._num_envs = num_envs
        self._input_size = input_size
        self.hidden_size = hidden_size
        self._output_size = output_size
        self.n_steps = n_steps
        self.hidden_layer = VectorLIFLayer(num_envs, input_size, hidden_size)
        self.output_layer = VectorLIFLayer(num_envs, hidden_size, output_size)
    
    @property
    def input_size(self) -> int:
        return self._input_size
    
    @property
    def output_size(self) -> int:
        return self._output_size
    
    @property
    def num_envs(self) -> int:
        return self._num_envs

    def forward(self, obs):
        """
        Run the SNN for `n_steps` time-steps and return accumulated output
        spike counts (one value per action neuron).

        Parameters
        ----------
        obs : (num_envs, input_size) float array, values in [0, 1]
              Treated as a constant Poisson rate stimulus.

        Returns
        -------
        spike_counts : (num_envs, output_size) float array
                       Larger value → neuron fired more → preferred action.
        """
        spike_counts = np.zeros((self.num_envs, self.output_layer.n_out))
        for _ in range(self.n_steps):
            # Poisson / rate encoding: spike with probability = input value
            encoded = (np.random.rand(*obs.shape) < obs).astype(np.float32)
            h_spikes = self.hidden_layer.forward(encoded)
            o_spikes = self.output_layer.forward(h_spikes)
            spike_counts += o_spikes
        return spike_counts

    def reset_state(self):
        """Reset all neuron membrane potentials (call between episodes)."""
        self.hidden_layer.reset_state()
        self.output_layer.reset_state()

    def get_weights(self) -> np.ndarray:
        """Get flat array of all trainable weights for a single agent (for compatibility)."""
        # This returns weights for agent 0 - needed for NeuralNetwork interface
        h_w = self.hidden_layer.get_weights()[0]
        o_w = self.output_layer.get_weights()[0]
        return np.concatenate([h_w, o_w])
    
    def set_weights(self, weights: np.ndarray):
        """Set weights from flat array for a single agent (for compatibility)."""
        # This sets all agents to the same weights
        h_size = (self.hidden_layer.n_in * self.hidden_layer.n_out) + self.hidden_layer.n_out
        h_weights = weights[:h_size]
        o_weights = weights[h_size:]
        
        # Set all agents to the same weights
        self.hidden_layer.set_weights([h_weights] * self.num_envs)
        self.output_layer.set_weights([o_weights] * self.num_envs)
    
    def get_genomes(self):
        h_w = self.hidden_layer.get_weights()
        o_w = self.output_layer.get_weights()
        return [np.concatenate([h, o]) for h, o in zip(h_w, o_w)]

    def set_genomes(self, genomes):
        h_size = (self.hidden_layer.n_in * self.hidden_layer.n_out) + self.hidden_layer.n_out
        h_weights = [g[:h_size] for g in genomes]
        o_weights = [g[h_size:] for g in genomes]
        self.hidden_layer.set_weights(h_weights)
        self.output_layer.set_weights(o_weights)


# ---------------------------------------------------------------------------
# Network Factory
# ---------------------------------------------------------------------------

def create_vector_neural_network(network_type: str, num_envs: int, input_size: int, 
                                  output_size: int, **kwargs) -> NeuralNetwork:
    """
    Factory function to create different types of vectorized neural networks.
    
    Parameters:
    -----------
    network_type : str
        Type of network to create. Options: "spiking", "feedforward"
    num_envs : int
        Number of parallel agents
    input_size : int
        Input dimension
    output_size : int
        Output dimension
    **kwargs : additional parameters
        For spiking: hidden_size (default 64), n_steps (default 5)
        For feedforward: hidden_sizes (default [64])
    
    Returns:
    --------
    NeuralNetwork instance
    """
    if network_type == "spiking":
        hidden_size = kwargs.get("hidden_size", 64)
        n_steps = kwargs.get("n_steps", 5)
        return VectorSpikingNetwork(num_envs, input_size, hidden_size, output_size, n_steps)
    
    elif network_type == "feedforward":
        hidden_sizes = kwargs.get("hidden_sizes", [64])
        return VectorFeedforwardNetwork(num_envs, input_size, hidden_sizes, output_size)
    
    else:
        raise ValueError(f"Unknown network type: {network_type}. Choose from 'spiking' or 'feedforward'")


# ---------------------------------------------------------------------------
# Neuroevolution Strategy with Extinction Logic (Vectorized)
# ---------------------------------------------------------------------------

class NNStrategy(BaseStrategy):
    def __init__(self, population_size=50, generations=20, num_trials=3, 
                 mutation_rate=0.2, mutation_mag=0.5, max_samples_per_gen=10,
                 network_type="spiking", **network_params):
        """
        Parameters:
        -----------
        population_size : int
            Number of agents per generation (also used as num_envs)
        generations : int
            Maximum number of generations
        num_trials : int
            Number of independent trials to run
        mutation_rate : float
            Probability of mutating each weight
        mutation_mag : float
            Standard deviation of Gaussian mutation noise
        max_samples_per_gen : int
            Maximum number of successful individuals to save per generation
        network_type : str
            Type of neural network to use ("spiking" or "feedforward")
        **network_params : additional parameters passed to network factory
            For spiking: hidden_size, n_steps
            For feedforward: hidden_sizes
        """
        params = {
            "pop_size": population_size,
            "generations": generations,
            "num_trials": num_trials,
            "mutation_rate": mutation_rate,
            "mutation_mag": mutation_mag,
            "max_samples_per_gen": max_samples_per_gen,
            "network_type": network_type,
            "network_params": network_params
        }
        super().__init__("vec_spike_nn", params)
        self.pop_size = population_size
        self.max_gens = generations
        self.num_trials = num_trials
        self.mutation_rate = mutation_rate
        self.mutation_mag = mutation_mag
        self.max_samples_per_gen = max_samples_per_gen
        self.network_type = network_type
        self.network_params = network_params

    def run(self, env):
        """Encapsulated trial and generation loop for vectorized SNN with progress tracking"""
        from tqdm import tqdm
        import time
        
        # Ensure environment matches population size
        if env.num_envs != self.pop_size:
            raise ValueError(f"Environment has {env.num_envs} agents but strategy expects {self.pop_size}. "
                           f"Run with --population {env.num_envs} to match.")
        
        for trial in range(1, self.num_trials + 1):
            # Create trial directory
            trial_dir = os.path.join(env.output_dir, f"trial_{trial}")
            os.makedirs(trial_dir, exist_ok=True)
            
            # Create population using factory
            pop_brain = create_vector_neural_network(
                self.network_type,
                self.pop_size,  # This now matches env.num_envs
                env.num_rays,
                4,  # 4 actions
                **self.network_params
            )
            
            # 1. Trial-level progress bar
            gen_pbar = tqdm(range(1, self.max_gens + 1), 
                            desc=f"Trial {trial}/{self.num_trials}", 
                            unit="gen", position=0)
            
            for gen in gen_pbar:
                # Create generation directory
                gen_dir = os.path.join(trial_dir, f"gen_{gen}")
                os.makedirs(gen_dir, exist_ok=True)
                
                # Create samples directory
                samples_dir = os.path.join(gen_dir, "samples")
                os.makedirs(samples_dir, exist_ok=True)
                
                obs = env.reset()
                
                # Store initial positions, reward positions, and initial energy/health for each individual
                initial_poses = [(env.robot_x[i], env.robot_y[i]) for i in range(self.pop_size)]
                reward_positions = [(env.goal_x[i], env.goal_y[i]) for i in range(self.pop_size)]
                
                # Calculate initial distances to reward for each individual
                initial_distances = []
                for i in range(self.pop_size):
                    dx = initial_poses[i][0] - reward_positions[i][0]
                    dy = initial_poses[i][1] - reward_positions[i][1]
                    initial_distances.append(np.sqrt(dx*dx + dy*dy))
                
                pop_brain.reset_state()
                
                gen_start_time = time.time()
                steps = 0
                
                # Track progress over steps for this generation
                goals_reached_by_step = set()
                
                # Track which individuals have reached the goal and their success metrics
                reached_goal = np.zeros(self.pop_size, dtype=bool)
                success_steps = np.full(self.pop_size, -1, dtype=int)
                success_energy = np.full(self.pop_size, -1, dtype=float)
                success_health = np.full(self.pop_size, -1, dtype=float)
                
                # CSV log for this generation (step-level data)
                step_csv_path = os.path.join(gen_dir, "log.csv")
                with open(step_csv_path, 'w', newline='') as step_f:
                    step_writer = csv.writer(step_f)
                    step_writer.writerow(["step", "percent_done", "new_successes"])
                    
                    while True:
                        # Parallel inference and step
                        output = pop_brain.forward(obs / env.ray_length)
                        actions = np.argmax(output, axis=1)
                        obs, rewards, dones, info = env.step(actions)
                        
                        steps += 1
                        
                        # Track newly reached goals
                        new_successes = info["goal_reached"] & ~reached_goal
                        new_success_indices = np.where(new_successes)[0]
                        
                        for idx in new_success_indices:
                            goals_reached_by_step.add(idx)
                            reached_goal[idx] = True
                            success_steps[idx] = steps
                            success_energy[idx] = env.energy[idx]
                            success_health[idx] = env.health[idx]
                            
                            # Save sample weights for successful individuals (limited per gen)
                            if len(os.listdir(samples_dir)) < self.max_samples_per_gen:
                                sample_path = os.path.join(samples_dir, f"ind_{idx}.json")
                                genome = pop_brain.get_genomes()[idx]
                                sample_data = {
                                    "individual_id": int(idx),
                                    "generation": gen,
                                    "trial": trial,
                                    "weights": genome.tolist(),
                                    "initial_position": {"x": float(initial_poses[idx][0]), "y": float(initial_poses[idx][1])},
                                    "initial_distance_to_reward": float(initial_distances[idx]),
                                    "reward_position": {"x": float(reward_positions[idx][0]), "y": float(reward_positions[idx][1])},
                                    "steps_to_success": int(success_steps[idx]),
                                    "energy_remaining": float(success_energy[idx]),
                                    "health_remaining": float(success_health[idx])
                                }
                                with open(sample_path, 'w') as wf:
                                    json.dump(sample_data, wf, indent=2)
                        
                        percent_done = (len(goals_reached_by_step) / self.pop_size) * 100
                        step_writer.writerow([steps, f"{percent_done:.2f}%", len(new_success_indices)])
                        step_f.flush()
                        
                        # Update progress every 10 steps to reduce overhead
                        if steps % 10 == 0:
                            gen_pbar.set_postfix({
                                "Step": steps,
                                "%Done": f"{percent_done:.1f}%"
                            })
                        
                        if env.render_flag:
                            env.render()
                        
                        if np.all(dones):
                            break
                
                # 2. End-of-Generation reporting
                gen_duration = time.time() - gen_start_time
                survivor_indices = np.where(reached_goal)[0]
                success_rate = (len(survivor_indices) / self.pop_size) * 100
                
                # Calculate generation statistics for successful individuals
                path_lengths = [success_steps[idx] for idx in survivor_indices if success_steps[idx] > 0]
                energies = [success_energy[idx] for idx in survivor_indices if success_energy[idx] > 0]
                healths = [success_health[idx] for idx in survivor_indices if success_health[idx] > 0]
                initial_dists = [initial_distances[idx] for idx in survivor_indices]
                
                # Generate log.json for this generation
                gen_stats = {
                    "generation": gen,
                    "trial": trial,
                    "timestamp": datetime.now().isoformat(),
                    "success_rate_percent": success_rate,
                    "num_successful": len(survivor_indices),
                    "population_size": self.pop_size,
                    "avg_path_length": float(np.mean(path_lengths)) if path_lengths else 0,
                    "avg_energy_remaining": float(np.mean(energies)) if energies else 0,
                    "avg_health_remaining": float(np.mean(healths)) if healths else 0,
                    "avg_initial_distance_to_reward": float(np.mean(initial_dists)) if initial_dists else 0,
                    "generation_duration_seconds": gen_duration,
                    "total_steps_in_gen": steps,
                    "max_samples_per_gen": self.max_samples_per_gen,
                    "samples_saved": len(os.listdir(samples_dir))
                }
                
                gen_stats_path = os.path.join(gen_dir, "log.json")
                with open(gen_stats_path, 'w') as wf:
                    json.dump(gen_stats, wf, indent=2)
                
                gen_pbar.set_postfix({
                    "Success": f"{success_rate:.1f}%",
                    "Total_Steps": steps,
                    "Sec/Gen": f"{gen_duration:.1f}s"
                })
                
                # Also maintain a trial-level summary CSV
                trial_summary_path = os.path.join(trial_dir, "summary.csv")
                file_exists = os.path.exists(trial_summary_path)
                with open(trial_summary_path, 'a', newline='') as sf:
                    writer = csv.writer(sf)
                    if not file_exists:
                        writer.writerow(["generation", "success_rate_percent", "num_successful", 
                                        "avg_path_length", "avg_energy_remaining", 
                                        "avg_health_remaining", "avg_initial_distance_to_reward",
                                        "gen_duration_seconds", "total_steps_in_gen"])
                    writer.writerow([
                        gen, f"{success_rate:.2f}%", len(survivor_indices),
                        f"{np.mean(path_lengths) if path_lengths else 0:.2f}",
                        f"{np.mean(energies) if energies else 0:.2f}",
                        f"{np.mean(healths) if healths else 0:.2f}",
                        f"{np.mean(initial_dists) if initial_dists else 0:.2f}",
                        f"{gen_duration:.2f}", steps
                    ])
                
                if len(survivor_indices) == 0:
                    gen_pbar.write(f"[EXTINCT] Trial {trial} Gen {gen} - No survivors.")
                    break
                
                new_genomes = self._reproduce(pop_brain.get_genomes(), survivor_indices)
                pop_brain.set_genomes(new_genomes)
    
    def _reproduce(self, all_genomes, survivor_indices):
        new_genomes = []
        # Elitism: Keep best survivors
        n_elites = max(1, self.pop_size // 10)
        for i in range(min(n_elites, len(survivor_indices))):
            new_genomes.append(all_genomes[survivor_indices[i]])
            
        # Mutation: Fill remaining population
        while len(new_genomes) < self.pop_size:
            parent_idx = np.random.choice(survivor_indices)
            child_genome = all_genomes[parent_idx].copy()
            
            mask = np.random.random(len(child_genome)) < self.mutation_rate
            child_genome[mask] += np.random.randn(np.sum(mask)) * self.mutation_mag
            new_genomes.append(child_genome)
            
        return new_genomes