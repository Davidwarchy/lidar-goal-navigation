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
import multiprocessing as mp
from tqdm import tqdm
import time


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

        self._total_params = sum(
            layer_in * layer_out + layer_out
            for layer_in, layer_out in zip([input_size] + hidden_sizes, hidden_sizes + [output_size])
        )
    
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
        x = inputs
        for i in range(len(self.weights) - 1):
            # x: (num_envs, in_i) -> (num_envs, 1, in_i) @ (num_envs, in_i, out_i) -> (num_envs, 1, out_i) -> (num_envs, out_i)
            x = np.maximum(0, (x[:, None, :] @ self.weights[i]).squeeze(1) + self.biases[i])
        # Output layer (linear)
        x = (x[:, None, :] @ self.weights[-1]).squeeze(1) + self.biases[-1]
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
    
    def get_genomes(self) -> np.ndarray:
        """Return (num_envs, total_params) array of all agent weights."""
        parts = []
        for w, b in zip(self.weights, self.biases):
            # w: (num_envs, in, out) -> flatten last two dims
            parts.append(w.reshape(self.num_envs, -1))
            parts.append(b)                     # (num_envs, out)
        return np.concatenate(parts, axis=1)

    def set_genomes(self, genomes: np.ndarray):
        """Assign from (num_envs, total_params) array."""
        start = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            w_size = w.shape[1] * w.shape[2]
            b_size = b.shape[1]

            # Weight matrix
            w_flat = genomes[:, start:start + w_size]
            self.weights[i] = w_flat.reshape(self.num_envs, w.shape[1], w.shape[2])
            start += w_size

            # Bias vector
            self.biases[i] = genomes[:, start:start + b_size]
            start += b_size

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

        self._total_params = n_in * n_out + n_out   # ← store for later slicing

    def reset_state(self):
        self.V.fill(0.0)

    def forward(self, x):
        # x shape: (num_envs, n_in)
        # Vectorized batch matrix multiplication for unique weights per env
        # x: (num_envs, n_in)
        z = (x[:, None, :] @ self.W).squeeze(1) + self.b
        self.V = self.leak * self.V + z
        spikes = (self.V >= self.threshold).astype(np.float32)
        self.V[spikes == 1.0] = 0.0
        return spikes

    def get_weights(self):
        """Return (num_envs, total_params) array."""
        w_flat = self.W.reshape(self.num_envs, -1)          # (num_envs, n_in*n_out)
        return np.concatenate([w_flat, self.b], axis=1)     # (num_envs, total_params)

    def set_weights(self, weights):
        """Set from (num_envs, total_params) array."""
        w_size = self.n_in * self.n_out
        self.W = weights[:, :w_size].reshape(self.num_envs, self.n_in, self.n_out)
        self.b = weights[:, w_size:]


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

    def get_genomes(self):
        """Return (num_envs, total_params) array."""
        h_weights = self.hidden_layer.get_weights()   # (num_envs, h_params)
        o_weights = self.output_layer.get_weights()   # (num_envs, o_params)
        return np.concatenate([h_weights, o_weights], axis=1)

    def set_genomes(self, genomes):
        """Set from (num_envs, total_params) array."""
        h_size = self.hidden_layer._total_params
        h_weights = genomes[:, :h_size]
        o_weights = genomes[:, h_size:]
        self.hidden_layer.set_weights(h_weights)
        self.output_layer.set_weights(o_weights)
    
    # Keep old get_weights/set_weights for compatibility if needed
    def get_weights(self) -> np.ndarray:
        """Return flat weights for agent 0 (backward compatible)."""
        return self.get_genomes()[0]

    def set_weights(self, weights: np.ndarray):
        """Set all agents to the same weights (backward compatible)."""
        genomes = np.tile(weights, (self.num_envs, 1))
        self.set_genomes(genomes)


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
# Helper function for parallel trial execution (used when parallel_trials=True)
# ---------------------------------------------------------------------------

def _run_single_trial(trial_idx, strategy_params, env_params, base_output_dir):
    trial_dir = os.path.join(base_output_dir, f"trial_{trial_idx}")
    env_params_with_out = env_params.copy()
    env_params_with_out["output_dir"] = trial_dir
    from env import VectorRobotExplorationEnv   # local import to avoid circular issues
    env = VectorRobotExplorationEnv(**env_params_with_out)
    
    # Create population using factory
    pop_brain = create_vector_neural_network(
        strategy_params['network_type'],
        strategy_params['population_size'],
        env.num_rays,
        4,  # 4 actions
        **strategy_params['network_params']
    )
    
    curriculum_streak = 0
    gen_pbar = tqdm(range(1, strategy_params['max_gens'] + 1),
                    desc=f"Trial {trial_idx}/{strategy_params['num_trials']}",
                    unit="gen", position=0)
    
    for gen in gen_pbar:
        gen_dir = os.path.join(trial_dir, f"gen_{gen}")
        os.makedirs(gen_dir, exist_ok=True)
        
        if strategy_params['save_top_k'] > 0:
            top_k_dir = os.path.join(gen_dir, "top_k")
            os.makedirs(top_k_dir, exist_ok=True)
        
        obs = env.reset()
        
        initial_poses = [(env.robot_x[i], env.robot_y[i]) for i in range(strategy_params['population_size'])]
        reward_positions = [(env.goal_x[i], env.goal_y[i]) for i in range(strategy_params['population_size'])]
        
        initial_distances = []
        for i in range(strategy_params['population_size']):
            dx = initial_poses[i][0] - reward_positions[i][0]
            dy = initial_poses[i][1] - reward_positions[i][1]
            initial_distances.append(np.sqrt(dx*dx + dy*dy))
        
        pop_brain.reset_state()
        
        gen_start_time = time.time()
        steps = 0
        
        reached_goal = np.zeros(strategy_params['population_size'], dtype=bool)
        success_steps = np.full(strategy_params['population_size'], -1, dtype=int)
        success_energy = np.full(strategy_params['population_size'], -1, dtype=float)
        success_health = np.full(strategy_params['population_size'], -1, dtype=float)
        
        step_csv_path = os.path.join(gen_dir, "log.csv")
        with open(step_csv_path, 'w', newline='') as step_f:
            step_writer = csv.writer(step_f)
            step_writer.writerow(["step", "percent_done", "new_successes"])
            
            while True:
                output = pop_brain.forward(obs / env.ray_length)
                
                if strategy_params['action_space'] == "discrete":
                    if strategy_params['action_distribution'] == "deterministic":
                        actions = np.argmax(output, axis=1)
                    else:
                        exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
                        probs = exp_output / np.sum(exp_output, axis=1, keepdims=True)
                        actions = np.array([np.random.choice(4, p=probs[i]) for i in range(len(probs))])
                else:
                    linear_vel = output[:, 0]
                    angular_vel = output[:, 1]
                    linear_vel = np.clip(linear_vel, -1, 1) * env.linear_speed
                    angular_vel = np.clip(angular_vel, -1, 1) * 90
                    actions = np.stack([linear_vel, angular_vel], axis=1)
                    if strategy_params['action_distribution'] == "stochastic":
                        noise_scale = 0.1
                        actions += np.random.randn(*actions.shape) * noise_scale
                        actions[:, 0] = np.clip(actions[:, 0], -env.linear_speed, env.linear_speed)
                        actions[:, 1] = np.clip(actions[:, 1], -90, 90)
                
                obs, rewards, dones, info = env.step(actions, action_space=strategy_params['action_space'])
                steps += 1
                
                new_successes = info["goal_reached"] & ~reached_goal
                new_success_indices = np.where(new_successes)[0]
                for idx in new_success_indices:
                    reached_goal[idx] = True
                    success_steps[idx] = steps
                    success_energy[idx] = env.energy[idx]
                    success_health[idx] = env.health[idx]
                
                percent_done = (np.sum(reached_goal) / strategy_params['population_size']) * 100
                step_writer.writerow([steps, f"{percent_done:.2f}%", len(new_success_indices)])
                step_f.flush()
                
                if steps % 10 == 0:
                    gen_pbar.set_postfix({"Step": steps, "%Done": f"{percent_done:.1f}%"})
                
                if env.render_flag:
                    env.render()
                
                if np.all(dones):
                    break
        
        gen_duration = time.time() - gen_start_time
        survivor_indices = np.where(reached_goal)[0]
        success_rate = (len(survivor_indices) / strategy_params['population_size']) * 100
        
        curriculum_promoted = False
        curriculum_active = (strategy_params['network_type'] == "feedforward" and strategy_params['curriculum_enabled'])
        if curriculum_active:
            if success_rate >= (strategy_params['curriculum_success_threshold'] * 100.0):
                curriculum_streak += 1
            else:
                curriculum_streak = 0
            if curriculum_streak >= strategy_params['curriculum_consecutive_gens']:
                env.goal_spawn_dist += strategy_params['curriculum_distance_increment']
                curriculum_streak = 0
                curriculum_promoted = True
                gen_pbar.write(f"[CURRICULUM] Trial {trial_idx} Gen {gen}: goal_spawn_dist -> {env.goal_spawn_dist:.2f}")
        
        if strategy_params['save_top_k'] > 0 and len(survivor_indices) > 0:
            survivors_list = list(survivor_indices)
            random.shuffle(survivors_list)
            top_k = min(strategy_params['save_top_k'], len(survivors_list))
            genomes = pop_brain.get_genomes()
            for idx in survivors_list[:top_k]:
                weight_path = os.path.join(top_k_dir, f"ind_{idx}.json")
                weight_data = {
                    "individual_id": int(idx),
                    "generation": gen,
                    "trial": trial_idx,
                    "steps_to_success": int(success_steps[idx]),
                    "energy_remaining": float(success_energy[idx]),
                    "health_remaining": float(success_health[idx]),
                    "initial_distance_to_reward": float(initial_distances[idx]),
                    "initial_position": {"x": float(initial_poses[idx][0]), "y": float(initial_poses[idx][1])},
                    "reward_position": {"x": float(reward_positions[idx][0]), "y": float(reward_positions[idx][1])},
                    "weights": genomes[idx].tolist()
                }
                with open(weight_path, 'w') as wf:
                    json.dump(weight_data, wf, indent=2)
        
        path_lengths = [success_steps[idx] for idx in survivor_indices if success_steps[idx] > 0]
        energies = [success_energy[idx] for idx in survivor_indices if success_energy[idx] > 0]
        healths = [success_health[idx] for idx in survivor_indices if success_health[idx] > 0]
        initial_dists = [initial_distances[idx] for idx in survivor_indices]
        
        gen_stats = {
            "generation": gen,
            "trial": trial_idx,
            "timestamp": datetime.now().isoformat(),
            "success_rate_percent": success_rate,
            "num_successful": len(survivor_indices),
            "population_size": strategy_params['population_size'],
            "avg_path_length": float(np.mean(path_lengths)) if path_lengths else 0,
            "avg_energy_remaining": float(np.mean(energies)) if energies else 0,
            "avg_health_remaining": float(np.mean(healths)) if healths else 0,
            "avg_initial_distance_to_reward": float(np.mean(initial_dists)) if initial_dists else 0,
            "generation_duration_seconds": gen_duration,
            "total_steps_in_gen": steps,
            "save_top_k": strategy_params['save_top_k'],
            "top_k_saved": min(strategy_params['save_top_k'], len(survivor_indices)) if strategy_params['save_top_k'] > 0 else 0,
            "goal_spawn_distance": float(env.goal_spawn_dist),
            "curriculum_streak": int(curriculum_streak if curriculum_active else 0),
            "curriculum_promoted": bool(curriculum_promoted)
        }
        gen_stats_path = os.path.join(gen_dir, "log.json")
        with open(gen_stats_path, 'w') as wf:
            json.dump(gen_stats, wf, indent=2)
        
        gen_pbar.set_postfix({"Success": f"{success_rate:.1f}%", "Total_Steps": steps, "Sec/Gen": f"{gen_duration:.1f}s"})
        
        trial_summary_path = os.path.join(trial_dir, "summary.csv")
        file_exists = os.path.exists(trial_summary_path)
        with open(trial_summary_path, 'a', newline='') as sf:
            writer = csv.writer(sf)
            if not file_exists:
                writer.writerow(["generation", "success_rate_percent", "num_successful", 
                                "avg_path_length", "avg_energy_remaining", 
                                "avg_health_remaining", "avg_initial_distance_to_reward",
                                "gen_duration_seconds", "total_steps_in_gen",
                                "goal_spawn_distance", "curriculum_streak", "curriculum_promoted"])
            writer.writerow([
                gen, f"{success_rate:.2f}%", len(survivor_indices),
                f"{np.mean(path_lengths) if path_lengths else 0:.2f}",
                f"{np.mean(energies) if energies else 0:.2f}",
                f"{np.mean(healths) if healths else 0:.2f}",
                f"{np.mean(initial_dists) if initial_dists else 0:.2f}",
                f"{gen_duration:.2f}", steps,
                f"{env.goal_spawn_dist:.2f}",
                curriculum_streak if curriculum_active else 0,
                "yes" if curriculum_promoted else "no"
            ])
        
        if len(survivor_indices) == 0:
            gen_pbar.write(f"[EXTINCT] Trial {trial_idx} Gen {gen} - No survivors.")
            break
        
        # Reproduction: random selection
        if len(survivor_indices) > 0:
            # Get genomes of survivors as a 2D array
            survivor_genomes = pop_brain.get_genomes()[survivor_indices]  # (n_survivors, total_params)

            # Select parents with replacement
            parent_idx = np.random.choice(len(survivor_indices), size=strategy_params['population_size'], replace=True)
            child_genomes = survivor_genomes[parent_idx].copy()

            # Vectorised mutation
            mutation_mask = np.random.random(child_genomes.shape) < strategy_params['mutation_rate']
            noise = np.random.randn(*child_genomes.shape) * strategy_params['mutation_mag']
            child_genomes[mutation_mask] += noise[mutation_mask]

            # Assign back to the population
            pop_brain.set_genomes(child_genomes)
    
    env.close()

# ---------------------------------------------------------------------------
# Neuroevolution Strategy with Extinction Logic (Vectorized)
# ---------------------------------------------------------------------------

class NNStrategy(BaseStrategy):
    def __init__(self, population_size=50, generations=20, num_trials=3, 
                mutation_rate=0.2, mutation_mag=0.5, max_samples_per_gen=10,
                network_type="spiking",
                save_top_k=0,
                curriculum_enabled=False,
                curriculum_success_threshold=0.05,
                curriculum_consecutive_gens=3,
                curriculum_distance_increment=5.0,
                action_space="discrete",
                action_distribution="deterministic",
                strategy_name="nn",   # this will be overridden in load_strategy
                parallel_trials=False,
                **network_params):
        params = {
            "pop_size": population_size,
            "generations": generations,
            "num_trials": num_trials,
            "mutation_rate": mutation_rate,
            "mutation_mag": mutation_mag,
            "max_samples_per_gen": max_samples_per_gen,
            "network_type": network_type,
            "save_top_k": save_top_k,
            "network_params": network_params,
            "curriculum_enabled": curriculum_enabled,
            "curriculum_success_threshold": curriculum_success_threshold,
            "curriculum_consecutive_gens": curriculum_consecutive_gens,
            "curriculum_distance_increment": curriculum_distance_increment,
            "action_space": action_space,
            "action_distribution": action_distribution,
        }
        super().__init__(strategy_name, params, parallel_trials=parallel_trials)
        self.pop_size = population_size
        self.max_gens = generations
        self.num_trials = num_trials
        self.mutation_rate = mutation_rate
        self.mutation_mag = mutation_mag
        self.max_samples_per_gen = max_samples_per_gen
        self.network_type = network_type
        self.network_params = network_params
        self.save_top_k = save_top_k
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_success_threshold = curriculum_success_threshold
        self.curriculum_consecutive_gens = max(1, curriculum_consecutive_gens)
        self.curriculum_distance_increment = curriculum_distance_increment
        self.action_space = action_space
        self.action_distribution = action_distribution

    def run(self, env_params, base_output_dir):
        # Gather strategy parameters for helper
        strategy_params = {
            'population_size': self.pop_size,
            'max_gens': self.max_gens,
            'num_trials': self.num_trials,
            'mutation_rate': self.mutation_rate,
            'mutation_mag': self.mutation_mag,
            'max_samples_per_gen': self.max_samples_per_gen,
            'network_type': self.network_type,
            'network_params': self.network_params,
            'save_top_k': self.save_top_k,
            'curriculum_enabled': self.curriculum_enabled,
            'curriculum_success_threshold': self.curriculum_success_threshold,
            'curriculum_consecutive_gens': self.curriculum_consecutive_gens,
            'curriculum_distance_increment': self.curriculum_distance_increment,
            'action_space': self.action_space,
            'action_distribution': self.action_distribution
        }
        
        if self.parallel_trials:
            with mp.Pool(processes=self.num_trials) as pool:
                args_list = [(trial_idx+1, strategy_params, env_params, base_output_dir)
                            for trial_idx in range(self.num_trials)]
                pool.starmap(_run_single_trial, args_list)
        else:
            for trial_idx in range(1, self.num_trials + 1):
                _run_single_trial(trial_idx, strategy_params, env_params, base_output_dir)