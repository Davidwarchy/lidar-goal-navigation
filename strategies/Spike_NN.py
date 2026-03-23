"""
Spiking Neural Network (SNN) with Genetic Algorithm (Neuroevolution)

This strategy implements a population-based neuroevolution system where
multiple neural network agents compete in each generation. The best performing
agents are selected for reproduction through natural selection.

Key Features:
- Population of neural networks (not just one)
- Genetic algorithm: selection, crossover, mutation
- Energy system: each step costs 1 energy point
- Goal-based fitness: reward when within 2 units of goal

Fitness Metrics Tracked:
- Generation: Current iteration number
- % of Generation Successful: Percentage of agents that reached the goal
- Average Success Path Length: Average steps taken by successful agents
- Average Success Energy Remaining: Average energy left for successful agents
- Average Success Health: Same as energy remaining
- Average Distance to Reward: Average distance to goal at end
- Weights Directory: Location where weights are saved
"""

import numpy as np
import os
import json
import csv
import shutil
from datetime import datetime


class SimpleNeuralNetwork:
    """
    Simple Feedforward Neural Network (for genetic algorithm)
    
    Architecture:
    - Input: 4 neurons (4 lidar rays for simplified model)
    - Hidden: 8 neurons
    - Output: 2 neurons (2 actions: turn left/right, or speed)
    
    Uses simple weights (no spiking) for faster evolution
    """
    
    def __init__(self, input_size=8, hidden_size=16, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with random values
        # Input -> Hidden
        
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.random.randn(hidden_size) * 0.1
        
        # Hidden -> Output
        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.random.randn(output_size) * 0.1
        
    def forward(self, inputs):
        """
        Forward pass through the network
        
        Args:
            inputs: Input vector (normalized lidar readings)
            
        Returns:
            output: Output vector (action probabilities)
        """
        # Hidden layer with tanh activation
        hidden = np.tanh(np.dot(inputs, self.w1) + self.b1)
        
        # Output layer with softmax
        output = np.dot(hidden, self.w2) + self.b2
        
        # Return as action selection
        return output
    
    def get_weights(self):
        """Get all weights as a flat vector (for genetic algorithms)"""
        return np.concatenate([
            self.w1.flatten(), self.b1,
            self.w2.flatten(), self.b2
        ])
    
    def set_weights(self, weights):
        """Set weights from a flat vector"""
        idx = 0
        
        # w1
        size = self.input_size * self.hidden_size
        self.w1 = weights[idx:idx+size].reshape(self.input_size, self.hidden_size)
        idx += size
        
        # b1
        self.b1 = weights[idx:idx+self.hidden_size]
        idx += self.hidden_size
        
        # w2
        size = self.hidden_size * self.output_size
        self.w2 = weights[idx:idx+size].reshape(self.hidden_size, self.output_size)
        idx += size
        
        # b2
        self.b2 = weights[idx:idx+self.output_size]
        
    def copy(self):
        """Create a copy of this network"""
        new_net = SimpleNeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_net.set_weights(self.get_weights().copy())
        return new_net
    
    def mutate(self, rate=0.1, magnitude=0.5):
        """Mutate weights randomly"""
        weights = self.get_weights()
        
        # Random mutations
        mask = np.random.random(len(weights)) < rate
        mutations = np.random.randn(len(weights)) * magnitude
        
        weights = np.where(mask, weights + mutations, weights)
        
        # Clip to prevent extreme weights
        weights = np.clip(weights, -3.0, 3.0)
        
        self.set_weights(weights)


class Agent:
    """
    Individual agent in the population
    
    Each agent has:
    - Neural network (brain)
    - Energy level (starts with predefined amount)
    - Fitness score
    - Success status
    """
    
    def __init__(self, network=None, energy=100):
        if network is None:
            self.network = SimpleNeuralNetwork(input_size=8, hidden_size=16, output_size=4)
        else:
            self.network = network
            
        self.energy = energy
        self.start_energy = energy
        self.fitness = 0.0
        self.successful = False
        self.steps_taken = 0
        self.final_distance = float('inf')
        self.energy_remaining = 0
        
    def reset(self, energy=100):
        """Reset agent state for new episode"""
        self.energy = energy
        self.start_energy = energy
        self.fitness = 0.0
        self.successful = False
        self.steps_taken = 0
        self.final_distance = float('inf')
        self.energy_remaining = 0
        
    def take_action(self, lidar_readings):
        """
        Use neural network to decide action
        
        Args:
            lidar_readings: Normalized lidar data
            
        Returns:
            action: 0=forward, 1=left, 2=right, 3=backward
        """
        # Downsample lidar to 8 inputs for better coverage
        if len(lidar_readings) > 8:
            indices = np.linspace(0, len(lidar_readings)-1, 8).astype(int)
            inputs = lidar_readings[indices]
        else:
            inputs = np.pad(lidar_readings, (0, 8 - len(lidar_readings)))
            
        # Get network output
        output = self.network.forward(inputs)
        
        # Convert to action - use argmax to select best action
        # Output has 4 values: [forward, left, right, backward]
        # Map to action indices: 0=forward, 1=left, 2=right, 3=backward
        return int(np.argmax(output))
            
    def update(self, action, reward=0, done=False, goal_distance=None):
        """
        Update agent state after each step
        
        Args:
            action: Action taken
            reward: Reward received
            done: Whether episode is done
            goal_distance: Distance to goal (if available)
        """
        # Each step costs 1 energy
        self.energy -= 1
        self.steps_taken += 1
        
        # Track final distance if done
        if done and goal_distance is not None:
            self.final_distance = goal_distance
            
        # Success if reached goal (reward > 0 or very close)
        if reward > 0 or (goal_distance is not None and goal_distance < 2):
            self.successful = True
            
        # Energy remaining at end
        self.energy_remaining = max(0, self.energy)
        
    def calculate_fitness(self):
        """
        Calculate fitness score based on performance
        
        Higher fitness = better agent
        """
        if self.successful:
            # Big reward for success + bonus for remaining energy
            self.fitness = 1000.0 + self.energy_remaining * 10.0 - self.steps_taken * 0.5
        else:
            # Partial reward based on getting closer to goal
            # Inverse of distance (closer = higher fitness)
            if self.final_distance < float('inf'):
                self.fitness = max(0, 100.0 - self.final_distance)
            else:
                self.fitness = 0.0
                
        return self.fitness
    
    def copy(self):
        """
        Create a copy of this agent
        """
        new_agent = Agent(network=self.network.copy(), energy=self.start_energy)
        return new_agent


class GeneticAlgorithm:
    """
    Genetic Algorithm for evolving neural network populations
    
    Implements:
    - Tournament selection
    - Crossover (crossover of weight vectors)
    - Mutation (random weight changes)
    """
    
    def __init__(self, population_size=50, elite_count=5, mutation_rate=0.1):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        
    def selection(self, agents):
        """Tournament selection - select best from random subset"""
        # Sort by fitness (descending)
        sorted_agents = sorted(agents, key=lambda a: a.fitness, reverse=True)
        return sorted_agents[:self.elite_count]
        
    def crossover(self, parent1, parent2):
        """
        Crossover two parent networks to create child
        
        Args:
            parent1: First parent Agent
            parent2: Second parent Agent
            
        Returns:
            child: New Agent with combined genes
        """
        # Get parent weights
        w1 = parent1.network.get_weights()
        w2 = parent2.network.get_weights()
        
        # Single-point crossover
        crossover_point = np.random.randint(len(w1))
        
        # Create child weights
        child_weights = np.concatenate([
            w1[:crossover_point],
            w2[crossover_point:]
        ])
        
        # Create child network
        child_network = SimpleNeuralNetwork()
        child_network.set_weights(child_weights)
        
        return Agent(network=child_network)
        
    def mutate(self, agent):
        """Apply mutation to agent's network"""
        agent.network.mutate(rate=self.mutation_rate, magnitude=0.5)
        return agent
    
    def evolve(self, agents):
        """
        Create next generation from current population
        
        Args:
            agents: Current population of agents
            
        Returns:
            new_agents: New generation of agents
        """
        # Calculate fitness for all agents
        for agent in agents:
            agent.calculate_fitness()
            
        # Sort by fitness
        sorted_agents = sorted(agents, key=lambda a: a.fitness, reverse=True)
        
        # Elitism: keep top performers
        new_agents = [agent.copy() for agent in sorted_agents[:self.elite_count]]
        
        # Fill rest of population with offspring
        while len(new_agents) < self.population_size:
            # Tournament selection
            tournament = np.random.choice(len(sorted_agents), 3, replace=False)
            tournament_agents = [sorted_agents[i] for i in tournament]
            parent1 = max(tournament_agents, key=lambda a: a.fitness)
            
            tournament = np.random.choice(len(sorted_agents), 3, replace=False)
            tournament_agents = [sorted_agents[i] for i in tournament]
            parent2 = max(tournament_agents, key=lambda a: a.fitness)
            
            # Crossover
            child = self.crossover(parent1, parent2)
            
            # Mutation
            child = self.mutate(child)
            
            new_agents.append(child)
            
        return new_agents


class SpikeNNGeneticStrategy:
    """
    Spike-NN Strategy with Genetic Algorithm (Neuroevolution)
    
    Manages the complete evolutionary process:
    - Initialize population
    - Run generations
    - Track fitness metrics
    - Save best weights
    """
    
    def __init__(self, 
                 population_size=50,
                 generations=20,
                 energy_per_agent=100,
                 elite_count=5,
                 mutation_rate=0.2,
                 num_rays=100,
                 weights_dir="spike_weights"):
        
        self.name = "spike_nn_genetic"
        self.parameters = {
            "population_size": population_size,
            "generations": generations,
            "energy_per_agent": energy_per_agent,
            "elite_count": elite_count,
            "mutation_rate": mutation_rate,
            "num_rays": num_rays,
            "weights_dir": weights_dir
        }
        
        self.population_size = population_size
        self.generations = generations
        self.energy_per_agent = energy_per_agent
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.num_rays = num_rays
        self.weights_dir = weights_dir
        
        # Create genetic algorithm
        self.ga = GeneticAlgorithm(population_size, elite_count, mutation_rate)
        
        # Initialize population
        self.population = [Agent(energy=energy_per_agent) for _ in range(population_size)]
        
        # Tracking metrics
        self.metrics = {
            "generation": 0,
            "percent_successful": 0.0,
            "avg_success_path_length": 0.0,
            "avg_success_energy_remaining": 0.0,
            "avg_success_health": 0.0,
            "avg_distance_to_reward": float('inf'),
            "best_fitness": 0.0,
            "weights_directory": weights_dir
        }
        
        # Store all generation metrics for CSV export
        self.all_generation_metrics = []
        
        # Create weights directory
        os.makedirs(weights_dir, exist_ok=True)
        
    def _preprocess_lidar(self, lidar_readings):
        """Normalize and downsample lidar readings to 8 inputs"""
        if not isinstance(lidar_readings, np.ndarray):
            lidar_readings = np.array(lidar_readings)
            
        # Normalize to [0, 1]
        if np.max(lidar_readings) > 0:
            lidar_readings = lidar_readings / np.max(lidar_readings)
        
        # Downsample to 8 inputs for better coverage
        if len(lidar_readings) > 8:
            indices = np.linspace(0, len(lidar_readings)-1, 8).astype(int)
            lidar_readings = lidar_readings[indices]
        elif len(lidar_readings) < 8:
            lidar_readings = np.pad(lidar_readings, (0, 8 - len(lidar_readings)))
        
        return np.clip(lidar_readings + 0.01, 0.0, 1.0)
        
    def _run_agent(self, agent, env):
        """
        Run a single agent in the environment
        
        Args:
            agent: Agent to run
            env: Environment instance
            
        Returns:
            done: Whether episode completed
        """
        # Reset agent and environment
        agent.reset(self.energy_per_agent)
        obs = env.reset()
        
        done = False
        
        while not done and agent.energy > 0:
            # Get lidar observation
            if isinstance(obs, tuple):
                lidar = obs[0]
            else:
                lidar = obs
            
            # Preprocess lidar - downsample to 8 inputs
            inputs = self._preprocess_lidar(lidar)
            
            # Ensure input size is exactly 8
            if len(inputs) != 8:
                inputs = np.pad(inputs, (0, max(0, 8 - len(inputs))))
            
            # Get action from neural network
            action = agent.take_action(inputs)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            
            # Calculate goal distance if available
            goal_dist = None
            if hasattr(env, 'robot_x') and hasattr(env, 'goal_x'):
                goal_dist = np.sqrt((env.robot_x - env.goal_x)**2 + 
                                   (env.robot_y - env.goal_y)**2)
            
            # Update agent
            agent.update(action, reward, done, goal_dist)
            
        return done
        
    def _evaluate_population(self, env):
        """
        Evaluate entire population in the environment
        
        Returns:
            metrics: Dictionary of fitness metrics
        """
        successful_agents = []
        
        for i, agent in enumerate(self.population):
            self._run_agent(agent, env)
            
            if agent.successful:
                successful_agents.append(agent)
                
        # Calculate metrics
        num_successful = len(successful_agents)
        self.metrics["percent_successful"] = (num_successful / self.population_size) * 100.0
        
        if num_successful > 0:
            self.metrics["avg_success_path_length"] = np.mean([a.steps_taken for a in successful_agents])
            self.metrics["avg_success_energy_remaining"] = np.mean([a.energy_remaining for a in successful_agents])
            self.metrics["avg_success_health"] = self.metrics["avg_success_energy_remaining"]  # Same metric
        else:
            self.metrics["avg_success_path_length"] = 0.0
            self.metrics["avg_success_energy_remaining"] = 0.0
            self.metrics["avg_success_health"] = 0.0
            
        # Average distance to reward (for unsuccessful agents too)
        all_distances = [a.final_distance for a in self.population if a.final_distance < float('inf')]
        if all_distances:
            self.metrics["avg_distance_to_reward"] = np.mean(all_distances)
        else:
            self.metrics["avg_distance_to_reward"] = float('inf')
            
        # Best fitness this generation
        best_agent = max(self.population, key=lambda a: a.calculate_fitness())
        self.metrics["best_fitness"] = best_agent.fitness
        
        return self.metrics
        
    def _save_weights(self, generation):
        """Save best agent weights to file"""
        # Get best agent
        best_agent = max(self.population, key=lambda a: a.calculate_fitness())
        
        # Save weights
        filename = os.path.join(self.weights_dir, f"gen_{generation:03d}.json")
        
        weights_data = {
            "generation": generation,
            "fitness": best_agent.fitness,
            "weights": best_agent.network.get_weights().tolist(),
            "successful": best_agent.successful,
            "steps": best_agent.steps_taken,
            "energy_remaining": best_agent.energy_remaining
        }
        
        with open(filename, 'w') as f:
            json.dump(weights_data, f)
            
        return filename
        
    def _print_metrics(self, metrics):
        """Print generation metrics in table format"""
        print("\n" + "="*70)
        print(f"Generation: {metrics['generation']}")
        print("="*70)
        print(f"{'Metric':<40} {'Value':>25}")
        print("-"*70)
        print(f"{'% of Generation Successful':<40} {metrics['percent_successful']:>24.1f}%")
        print(f"{'Average Success Path Length':<40} {metrics['avg_success_path_length']:>25.1f}")
        print(f"{'Average Success Energy Remaining':<40} {metrics['avg_success_energy_remaining']:>25.1f}")
        print(f"{'Average Success Health':<40} {metrics['avg_success_energy_remaining']:>25.1f}")
        if metrics['avg_distance_to_reward'] < float('inf'):
            print(f"{'Average Distance to Reward':<40} {metrics['avg_distance_to_reward']:>25.2f}")
        else:
            print(f"{'Average Distance to Reward':<40} {'N/A':>25}")
        print(f"{'Best Fitness':<40} {metrics['best_fitness']:>25.2f}")
        print(f"{'Weights Directory':<40} {metrics['weights_directory']:>25}")
        print("="*70)
        
    def run(self, env):
        """
        Run the neuroevolution training process
        
        Args:
            env: RobotExplorationEnv instance
            
        Returns:
            steps: Total steps across all generations
            best_coverage: Best coverage achieved
        """
        total_steps = 0
        best_coverage = 0.0
        
        print("\n" + "="*70)
        print("Starting Neuroevolution Training")
        print("="*70)
        print(f"Population Size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Energy per Agent: {self.energy_per_agent}")
        print(f"Mutation Rate: {self.mutation_rate}")
        print(f"Weights Directory: {self.weights_dir}")
        print("="*70)
        
        for gen in range(self.generations):
            self.metrics["generation"] = gen + 1
            
            # Evaluate current population
            metrics = self._evaluate_population(env)
            
            # Store metrics for CSV export
            self.all_generation_metrics.append(metrics.copy())
            
            # Print metrics
            self._print_metrics(metrics)
            
            # Track best coverage
            current_best = max(self.population, key=lambda a: a.calculate_fitness())
            if current_best.successful:
                best_coverage = max(best_coverage, 100.0)  # Consider success as 100%
                
            total_steps += sum(a.steps_taken for a in self.population)
            
            # Save best weights
            weight_file = self._save_weights(gen)
            print(f"Saved weights to: {weight_file}")
            
            # Evolve for next generation (except last generation)
            if gen < self.generations - 1:
                self.population = self.ga.evolve(self.population)
                print(f"\nEvolving to generation {gen + 2}...")
                
        print("\n" + "="*70)
        print("Neuroevolution Training Complete!")
        print("="*70)
        
        # Save final summary
        summary_file = os.path.join(self.weights_dir, "training_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                "total_generations": self.generations,
                "final_metrics": self.metrics,
                "total_steps": total_steps
            }, f, indent=2)
            
        # Save CSV export of all generation metrics
        csv_file = os.path.join(self.weights_dir, "training_metrics.csv")
        with open(csv_file, 'w', newline='') as f:
            fieldnames = [
                'Generation',
                '% of Generation Successful',
                'Average Success Path Length',
                'Average Success Energy Remaining',
                'Average Success Health',
                'Average Distance to Reward',
                'Weights Directory'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for gen_metrics in self.all_generation_metrics:
                row = {
                    'Generation': gen_metrics.get('generation', 0),
                    '% of Generation Successful': f"{gen_metrics.get('percent_successful', 0.0):.1f}%",
                    'Average Success Path Length': f"{gen_metrics.get('avg_success_path_length', 0.0):.1f}",
                    'Average Success Energy Remaining': f"{gen_metrics.get('avg_success_energy_remaining', 0.0):.1f}",
                    'Average Success Health': f"{gen_metrics.get('avg_success_health', 0.0):.1f}",
                    'Average Distance to Reward': f"{gen_metrics.get('avg_distance_to_reward', 'N/A') if gen_metrics.get('avg_distance_to_reward', 0) != float('inf') else 'N/A'}",
                    'Weights Directory': gen_metrics.get('weights_directory', '')
                }
                writer.writerow(row)
        
        print(f"Training summary saved to: {summary_file}")
        print(f"CSV metrics saved to: {csv_file}")
        
        return total_steps, best_coverage


def create_spike_nn_strategy(**kwargs):
    """Factory function to create Spike-NN strategy"""
    return SpikeNNGeneticStrategy(**kwargs)