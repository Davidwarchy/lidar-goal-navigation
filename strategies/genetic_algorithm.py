import os
import json
import numpy as np
from .base_strategy import BaseStrategy

class NeuralNetwork:
    """
    A simple Multi-Layer Perceptron (MLP) neural network using pure NumPy operations.
    It links the perception (LIDAR rays) to the actuators (motors).
    """
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        """
        Initializes the neural network weights and biases using He initialization.
        
        Args:
            input_size (int): Number of input features (LIDAR rays).
            hidden_sizes (list): List containing the number of neurons in each hidden layer.
            output_size (int): Number of output neurons (motor actions).
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # He initialization: We initialize weights from a normal distribution and scale
        # by sqrt(2/fan_in). This helps maintain the variance of activations across layers,
        # preventing gradients from vanishing or exploding when using ReLU activation.
        self.W1 = np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2. / input_size)
        self.b1 = np.zeros(hidden_sizes[0])
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2. / hidden_sizes[0])
        self.b2 = np.zeros(hidden_sizes[1])
        self.W3 = np.random.randn(hidden_sizes[1], output_size) * np.sqrt(2. / hidden_sizes[1])
        self.b3 = np.zeros(output_size)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the neural network using highly optimized vectorized operations.
        
        Args:
            x (np.ndarray): Input array of shape (input_size,) containing normalized LIDAR distances.
            
        Returns:
            np.ndarray: Output array of shape (output_size,) containing raw action scores.
        """
        # --- First Hidden Layer ---
        # Matrix multiplication: np.dot(x, W1) computes the weighted sum of inputs for each neuron 
        # in the hidden layer. Adding biases shifts the activation space.
        z1 = np.dot(x, self.W1) + self.b1
        # ReLU activation: f(x) = max(0, x). We use np.maximum for vectorized element-wise max.
        # This introduces non-linearity, allowing the network to learn complex non-linear mappings 
        # from LIDAR inputs to required actions.
        a1 = np.maximum(0, z1)
        
        # --- Second Hidden Layer ---
        # Matrix multiplication to project from the first hidden layer space to the second hidden layer.
        z2 = np.dot(a1, self.W2) + self.b2
        # Apply ReLU activation again for further non-linear transformation.
        a2 = np.maximum(0, z2)
        
        # --- Output Layer ---
        # Project to the output space (action scores). No activation function here 
        # because we only need the relative magnitudes to select the max (argmax) later.
        z3 = np.dot(a2, self.W3) + self.b3
        return z3

    def get_action(self, x: np.ndarray) -> int:
        """
        Determines the best action to take given the current observation.
        
        Args:
            x (np.ndarray): Input array of LIDAR distances.
            
        Returns:
            int: The index of the selected action.
        """
        out = self.forward(x)
        # Select the action with the highest raw score using np.argmax. This is equivalent 
        # to applying a softmax and taking the most probable action, but computationally cheaper.
        return int(np.argmax(out))

    def get_genes(self) -> np.ndarray:
        """
        Serializes the network's weights and biases into a single 1D array.
        This represents the 'genetic code' of the neural network.
        
        Returns:
            np.ndarray: A 1D array containing all weights and biases.
        """
        # Flatten all weight matrices and bias vectors into 1D arrays, then concatenate them 
        # into a single contiguous array. This is required for fast vectorized genetic operations.
        return np.concatenate([
            self.W1.flatten(), self.b1.flatten(),
            self.W2.flatten(), self.b2.flatten(),
            self.W3.flatten(), self.b3.flatten()
        ])
        
    def set_genes(self, genes: np.ndarray):
        """
        Deserializes a 1D array of genes back into the network's weights and biases.
        
        Args:
            genes (np.ndarray): The 1D array containing the genetic code.
        """
        idx = 0
        
        # Reconstruct W1 by extracting the right number of elements and reshaping back to 2D.
        s = self.input_size * self.hidden_sizes[0]
        self.W1 = genes[idx:idx+s].reshape((self.input_size, self.hidden_sizes[0]))
        idx += s
        
        # Reconstruct b1. No reshaping needed as biases are 1D.
        s = self.hidden_sizes[0]
        self.b1 = genes[idx:idx+s]
        idx += s
        
        # Reconstruct W2.
        s = self.hidden_sizes[0] * self.hidden_sizes[1]
        self.W2 = genes[idx:idx+s].reshape((self.hidden_sizes[0], self.hidden_sizes[1]))
        idx += s
        
        # Reconstruct b2.
        s = self.hidden_sizes[1]
        self.b2 = genes[idx:idx+s]
        idx += s
        
        # Reconstruct W3.
        s = self.hidden_sizes[1] * self.output_size
        self.W3 = genes[idx:idx+s].reshape((self.hidden_sizes[1], self.output_size))
        idx += s
        
        # Reconstruct b3.
        s = self.output_size
        self.b3 = genes[idx:idx+s]

class GeneticAlgorithmStrategy(BaseStrategy):
    """
    A strategy that uses a Genetic Algorithm to evolve Random Neural Networks
    for navigating the robot to the goal.
    """
    def __init__(self, population_size=100, num_generations=30, mutation_rate=0.1, mutation_scale=0.1, survivor_ratio=0.2, load_weights=None):
        """
        Initializes the Genetic Algorithm strategy.
        
        Args:
            population_size (int): The number of neural networks in each generation.
            num_generations (int): The number of generations to simulate.
            mutation_rate (float): The probability of mutating a single gene.
            mutation_scale (float): The standard deviation of the Gaussian mutation noise.
            survivor_ratio (float): The fraction of the population that survives to the next generation.
            load_weights (str): Path to a directory with JSON weight files from a previous run.
        """
        super().__init__("ga", {
            "population_size": population_size,
            "num_generations": num_generations,
            "mutation_rate": mutation_rate,
            "mutation_scale": mutation_scale,
            "survivor_ratio": survivor_ratio,
            "load_weights": load_weights
        })
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.survivor_ratio = survivor_ratio
        self.load_weights = load_weights

    def _crossover(self, parent1_genes: np.ndarray, parent2_genes: np.ndarray) -> np.ndarray:
        """
        Performs uniform crossover between two parents to produce a child.
        
        Args:
            parent1_genes (np.ndarray): The 1D genetic array of the first parent.
            parent2_genes (np.ndarray): The 1D genetic array of the second parent.
            
        Returns:
            np.ndarray: The resulting child genetic array.
        """
        # Create a boolean mask of the same length as the genes using np.random.rand.
        # True means take gene from parent1, False means take from parent2.
        mask = np.random.rand(len(parent1_genes)) > 0.5
        # Vectorized conditional selection using np.where. This recombines genes extremely fast 
        # without standard Python loops.
        child_genes = np.where(mask, parent1_genes, parent2_genes)
        return child_genes

    def _mutate(self, genes: np.ndarray) -> np.ndarray:
        """
        Mutates a subset of genes by adding random Gaussian noise.
        
        Args:
            genes (np.ndarray): The 1D genetic array to mutate.
            
        Returns:
            np.ndarray: The mutated genetic array.
        """
        # Create a boolean mask where True indicates the gene will be mutated based on mutation_rate.
        mask = np.random.rand(len(genes)) < self.mutation_rate
        # Generate random Gaussian noise for every gene, scaled by mutation_scale.
        mutations = np.random.randn(len(genes)) * self.mutation_scale
        # Apply the mutations only to the genes selected by the boolean mask in a vectorized manner.
        genes[mask] += mutations[mask]
        return genes

    def run(self, env):
        """
        Executes the genetic algorithm experiment loop.
        
        Args:
            env (RobotExplorationEnv): The simulation environment.
            
        Returns:
            tuple: (total steps taken, final environment coverage)
        """
        # 1. Initialization
        population = []
        loaded_elites = []
        
        if self.load_weights and os.path.isdir(self.load_weights):
            print(f"[INFO] Loading previous generation weights from: {self.load_weights}")
            for filename in os.listdir(self.load_weights):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.load_weights, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    nn = NeuralNetwork(input_size=env.num_rays, hidden_sizes=[16, 16], output_size=4)
                    nn.W1 = np.array(data["W1"])
                    nn.b1 = np.array(data["b1"])
                    nn.W2 = np.array(data["W2"])
                    nn.b2 = np.array(data["b2"])
                    nn.W3 = np.array(data["W3"])
                    nn.b3 = np.array(data["b3"])
                    loaded_elites.append(nn)
                    population.append(nn)
                    
            if not loaded_elites:
                print(f"[WARNING] No .json files found in {self.load_weights}. Starting with completely random weights.")
            else:
                print(f"[INFO] Successfully loaded {len(loaded_elites)} robots.")

        # Fill the rest of the starting population
        # If we loaded elites, we recombine and mutate them to fill the population.
        # Otherwise, we fill with completely random neural networks.
        while len(population) < self.population_size:
            if loaded_elites:
                p1_idx, p2_idx = np.random.choice(len(loaded_elites), size=2, replace=True) if len(loaded_elites) > 1 else (0, 0)
                parent1 = loaded_elites[p1_idx]
                parent2 = loaded_elites[p2_idx]
                
                child_genes = self._crossover(parent1.get_genes(), parent2.get_genes())
                child_genes = self._mutate(child_genes)
                
                child_nn = NeuralNetwork(input_size=env.num_rays, hidden_sizes=[16, 16], output_size=4)
                child_nn.set_genes(child_genes)
                population.append(child_nn)
            else:
                population.append(NeuralNetwork(input_size=env.num_rays, hidden_sizes=[16, 16], output_size=4))
        
        # Truncate to exact population size just in case we loaded more JSONs than population size
        population = population[:self.population_size]
        
        total_steps = 0
        
        # Prepare CSV file for automatic logging
        csv_filepath = os.path.join(env.output_dir, "generation_results.csv")
        with open(csv_filepath, 'w') as f:
            f.write("Generation,% of Generation Successful,Average Success Path Length,Average Success Energy Remaining,Average Success Health,Average Distance to Reward,Weights Directory\n")
        
        # Output table headers matching exactly the requirements
        print(f"\n{'-'*130}")
        print(f"{'Generation':<10} | {'% Success':<10} | {'Avg Path Length':<15} | {'Avg Energy Rem':<15} | {'Avg Health Rem':<15} | {'Avg Dist to Reward':<18} | {'Weights Directory'}")
        print(f"{'-'*130}")
        
        for generation in range(1, self.num_generations + 1):
            # Pre-allocate numpy arrays to track metrics for the whole population
            fitness_scores = np.zeros(self.population_size)
            success_flags = np.zeros(self.population_size, dtype=bool)
            path_lengths = np.zeros(self.population_size)
            energy_remains = np.zeros(self.population_size)
            health_remains = np.zeros(self.population_size)
            dists_to_reward = np.zeros(self.population_size)
            
            # Evaluate each candidate in the population
            for i, nn in enumerate(population):
                obs = env.reset()
                done = False
                
                min_dist_to_reward = float('inf')
                
                while not done:
                    # Normalize observation (LIDAR distances) to [0, 1] range to prevent large 
                    # input values from causing exploding gradients/activations in the neural network.
                    normalized_obs = obs / env.ray_length
                    
                    # Forward pass through the neural network to get the action
                    action = nn.get_action(normalized_obs)
                    
                    # Step the environment
                    obs, reward, done, info = env.step(action)
                    total_steps += 1
                    
                    # Calculate Euclidean distance to reward using Pythagoras.
                    # We track the minimum distance achieved during the episode for fitness evaluation.
                    dx = env.robot_x - env.goal_x
                    dy = env.robot_y - env.goal_y
                    current_dist = np.sqrt(dx*dx + dy*dy)
                    if current_dist < min_dist_to_reward:
                        min_dist_to_reward = current_dist
                        
                    if env.render_flag:
                        env.render()
                
                # Evaluation Metrics Collection
                success = info.get('goal_reached', False)
                
                # Fitness Function Logic:
                # We want to minimize the minimum distance to the reward. Adding 1e-6 prevents division by zero.
                # If the goal is reached, give a massive fitness boost (1000.0) so they are practically 
                # guaranteed to be selected as elites for replication.
                fitness = 1.0 / (min_dist_to_reward + 1e-6)
                if success:
                    fitness += 1000.0
                    
                fitness_scores[i] = fitness
                success_flags[i] = success
                path_lengths[i] = env.current_step
                energy_remains[i] = info.get('energy', 0)
                health_remains[i] = info.get('health', 0)
                dists_to_reward[i] = min_dist_to_reward
                
            # Compute Generation Statistics for the Table
            percent_success = (np.sum(success_flags) / self.population_size) * 100.0
            
            if np.any(success_flags):
                avg_path = np.mean(path_lengths[success_flags])
                avg_energy = np.mean(energy_remains[success_flags])
                avg_health = np.mean(health_remains[success_flags])
            else:
                avg_path = 0.0
                avg_energy = 0.0
                avg_health = 0.0
                
            avg_dist = np.mean(dists_to_reward)
            
            # Create a specific directory to save this generation's successful weights
            gen_weights_dir = os.path.join(env.output_dir, f"gen_{generation}_weights")
            os.makedirs(gen_weights_dir, exist_ok=True)
            
            # Selection: successful creatures survive while those that couldn't cope die out
            # Sort population descending by fitness. np.argsort returns ascending, so we slice [::-1]
            sorted_indices = np.argsort(fitness_scores)[::-1]
            num_survivors = max(1, int(self.population_size * self.survivor_ratio))
            
            # If there are actual successes, we could optionally only keep those, but to keep 
            # the GA moving when there are no successes, we always keep the top `num_survivors`.
            elite_indices = sorted_indices[:num_survivors]
            elites = [population[idx] for idx in elite_indices]
            
            # Replication: Save weights of successful/elite robots
            for rank, idx in enumerate(elite_indices):
                elite_nn = population[idx]
                weights = {
                    "W1": elite_nn.W1.tolist(),
                    "b1": elite_nn.b1.tolist(),
                    "W2": elite_nn.W2.tolist(),
                    "b2": elite_nn.b2.tolist(),
                    "W3": elite_nn.W3.tolist(),
                    "b3": elite_nn.b3.tolist()
                }
                # Save the weights to the Weights Directory
                weight_file = os.path.join(gen_weights_dir, f"rank_{rank}_fitness_{fitness_scores[idx]:.4f}.json")
                with open(weight_file, 'w') as f:
                    json.dump(weights, f)
            
            # Print row for the current generation table
            print(f"{generation:<10} | {percent_success:<9.1f}% | {avg_path:<15.1f} | {avg_energy:<15.1f} | {avg_health:<15.1f} | {avg_dist:<18.1f} | {gen_weights_dir}")
            
            # Write row to CSV
            with open(csv_filepath, 'a') as f:
                f.write(f"{generation},{percent_success:.1f}%,{avg_path:.1f},{avg_energy:.1f},{avg_health:.1f},{avg_dist:.1f},{gen_weights_dir}\n")
            
            # If this is the last generation, we stop
            if generation == self.num_generations:
                break
                
            # Create next generation
            new_population = []
            
            # Replication: duplication of successful creatures (elites carry over directly)
            for elite in elites:
                new_nn = NeuralNetwork(input_size=env.num_rays, hidden_sizes=[16, 16], output_size=4)
                new_nn.set_genes(elite.get_genes().copy())
                new_population.append(new_nn)
                
            # Recombination & Mutation to fill the rest of the generation
            while len(new_population) < self.population_size:
                # Randomly select two distinct parents from the successful creatures (elites)
                if len(elites) > 1:
                    p1_idx, p2_idx = np.random.choice(len(elites), size=2, replace=False)
                else:
                    p1_idx, p2_idx = 0, 0
                    
                parent1 = elites[p1_idx]
                parent2 = elites[p2_idx]
                
                # Recombination: mixing up of the genes from successful creatures to form a new creature
                child_genes = self._crossover(parent1.get_genes(), parent2.get_genes())
                
                # Mutation: slight changes in the genetic code of new creatures
                child_genes = self._mutate(child_genes)
                
                # Instantiate new creature with the combined and mutated genetic code
                child_nn = NeuralNetwork(input_size=env.num_rays, hidden_sizes=[16, 16], output_size=4)
                child_nn.set_genes(child_genes)
                new_population.append(child_nn)
                
            population = new_population

        print(f"{'-'*130}")
        # Return total steps taken and final coverage (from the very last episode run)
        return total_steps, env._get_coverage()
