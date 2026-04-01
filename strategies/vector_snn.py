from .base_strategy import BaseStrategy
import numpy as np

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

class VectorSNN:
    def __init__(self, num_envs, input_size=100, hidden_size=64, output_size=4, n_steps=5):
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.hidden_layer = VectorLIFLayer(num_envs, input_size, hidden_size)
        self.output_layer = VectorLIFLayer(num_envs, hidden_size, output_size)

    def forward(self, obs):
        spike_counts = np.zeros((self.num_envs, self.output_layer.n_out))
        for _ in range(self.n_steps):
            encoded = (np.random.rand(*obs.shape) < obs).astype(np.float32)
            h_spikes = self.hidden_layer.forward(encoded)
            o_spikes = self.output_layer.forward(h_spikes)
            spike_counts += o_spikes
        return spike_counts

    def reset_state(self):
        self.hidden_layer.reset_state()
        self.output_layer.reset_state()

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

class VectorSpikeNNStrategy(BaseStrategy):
    def __init__(self, population_size=50, generations=20, num_trials=3, mutation_rate=0.2, mutation_mag=0.5):
        super().__init__("vec_spike_nn", {"pop_size": population_size})
        self.pop_size = population_size
        self.max_gens = generations
        self.num_trials = num_trials
        self.mutation_rate = mutation_rate
        self.mutation_mag = mutation_mag

    def run(self, env):
        """Encapsulated trial and generation loop for vectorized SNN with progress tracking"""
        from tqdm import tqdm
        import time
        
        for trial in range(1, self.num_trials + 1):
            pop_brain = VectorSNN(num_envs=self.pop_size, input_size=env.num_rays) 
            
            # 1. Trial-level progress bar
            gen_pbar = tqdm(range(1, self.max_gens + 1), 
                            desc=f"Trial {trial}/{self.num_trials}", 
                            unit="gen", position=0) 
            
            for gen in gen_pbar:
                obs = env.reset() 
                pop_brain.reset_state() 
                
                gen_start_time = time.time()
                steps = 0
                
                while True:
                    # Parallel inference and step
                    output = pop_brain.forward(obs / env.ray_length) 
                    actions = np.argmax(output, axis=1) 
                    obs, rewards, dones, info = env.step(actions) 
                    
                    steps += 1
                    
                    # Update progress every 10 steps to reduce overhead
                    if steps % 10 == 0:
                        gen_pbar.set_postfix({
                            "Step": steps,
                            "Live_Found": np.sum(info["goal_reached"])
                        }) 

                    if env.render_flag: 
                        env.render() 
                    
                    if np.all(dones): 
                        break 
                
                # 2. End-of-Generation reporting
                gen_duration = time.time() - gen_start_time
                survivor_indices = np.where(info["goal_reached"])[0] 
                success_rate = (len(survivor_indices) / self.pop_size) * 100
                
                gen_pbar.set_postfix({
                    "Success": f"{success_rate:.1f}%",
                    "Total_Steps": steps,
                    "Sec/Gen": f"{gen_duration:.1f}s"
                }) 
                
                if len(survivor_indices) == 0:
                    gen_pbar.write(f"[EXTINCT] Trial {trial} Gen {gen} - No survivors.") 
                    break 
                
                new_genomes = self._reproduce(pop_brain.get_genomes(), survivor_indices) 
                pop_brain.set_genomes(new_genomes) 
                
    def _reproduce(self, all_genomes, survivor_indices):
        new_genomes = []
        # Elitism: Keep best survivors : 326]
        n_elites = max(1, self.pop_size // 10)
        for i in range(min(n_elites, len(survivor_indices))):
            new_genomes.append(all_genomes[survivor_indices[i]])
            
        # Mutation: Fill remaining population : 327]
        while len(new_genomes) < self.pop_size:
            parent_idx = np.random.choice(survivor_indices)
            child_genome = all_genomes[parent_idx].copy()
            
            mask = np.random.random(len(child_genome)) < self.mutation_rate
            child_genome[mask] += np.random.randn(np.sum(mask)) * self.mutation_mag
            new_genomes.append(child_genome)
            
        return new_genomes