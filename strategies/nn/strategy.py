import torch
import os
import json
import csv
import time
import random
import numpy as np
from tqdm import tqdm
from strategies.base_strategy import BaseStrategy
from strategies.log import StrategyLoggingMixin
from .helper import create_vector_neural_network, _average_recombination


# ---------------------------------------------------------------------------
# Helper function for parallel trial execution (used when parallel_trials=True)
# ---------------------------------------------------------------------------

def _run_single_trial(strategy_instance, trial_idx, strategy_params, env_params, base_output_dir):
    """
    Run one trial of the neuroevolution strategy.
    Uses the strategy_instance's logging mixin methods.
    """
    trial_dir = os.path.join(base_output_dir, f"trial_{trial_idx}")
    env_params_with_out = env_params.copy()
    env_params_with_out["output_dir"] = trial_dir
    from env import VectorRobotExplorationEnv   # local import to avoid circular issues
    env = VectorRobotExplorationEnv(**env_params_with_out)
    device = env.device
    
    # Set up trial logging using the strategy instance (which has the mixin)
    strategy_instance.setup_trial_logging(env, trial_idx, strategy_params['num_trials'], strategy_params['max_gens'])
    
    # Set curriculum parameters if enabled
    if strategy_params['curriculum_enabled']:
        strategy_instance.set_curriculum_params(
            enabled=True,
            success_threshold=strategy_params['curriculum_success_threshold'],
            consecutive_gens=strategy_params['curriculum_consecutive_gens'],
            distance_increment=strategy_params['curriculum_distance_increment']
        )
    
    # Create population using factory
    pop_brain = create_vector_neural_network(
        strategy_params['network_type'],
        strategy_params['population_size'],
        env.num_rays,
        4,  # 4 actions
        n_steps=strategy_params['n_steps'],
        device=device,
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
        
        # Setup generation logging (mixin)
        strategy_instance.setup_generation_logging(env, gen)
        
        # Reset environment – returns GPU tensor directly
        obs = env.reset()   # GPU tensor, shape (pop_size, num_rays)
        
        # Store initial positions (still need CPU for logging)
        initial_poses = [(env.robot_x[i].item(), env.robot_y[i].item()) for i in range(strategy_params['population_size'])]
        reward_positions = [(env.goal_x[i].item(), env.goal_y[i].item()) for i in range(strategy_params['population_size'])]
        
        initial_distances = []
        for i in range(strategy_params['population_size']):
            dx = initial_poses[i][0] - reward_positions[i][0]
            dy = initial_poses[i][1] - reward_positions[i][1]
            initial_distances.append(np.sqrt(dx*dx + dy*dy))
        strategy_instance._log_initial_agent_data(env)   # sets gen_initial_distances and writes initial CSV rows
        
        pop_brain.reset_state()
        
        gen_start_time = time.time()
        steps = 0
        
        # GPU tensors for tracking successes
        reached_goal = torch.zeros(strategy_params['population_size'], dtype=torch.bool, device=device)
        success_steps = torch.full((strategy_params['population_size'],), -1, dtype=torch.long, device=device)
        success_energy = torch.full((strategy_params['population_size'],), -1.0, dtype=torch.float32, device=device)
        success_health = torch.full((strategy_params['population_size'],), -1.0, dtype=torch.float32, device=device)
        
        # Main evaluation loop – wrapped in no_grad to prevent graph buildup
        with torch.no_grad():
            while True:
                # Normalized observations
                normalized_obs = obs / env.ray_length
                output = pop_brain.forward(normalized_obs)
                
                if strategy_params['action_space'] == "discrete":
                    if strategy_params['action_distribution'] == "deterministic":
                        actions = torch.argmax(output, dim=1)
                    else:
                        exp_output = torch.exp(output - torch.max(output, dim=1, keepdim=True)[0])
                        probs = exp_output / torch.sum(exp_output, dim=1, keepdim=True)
                        actions = torch.multinomial(probs, 1).squeeze(1)
                else:
                    linear_vel = output[:, 0]
                    angular_vel = output[:, 1]
                    linear_vel = torch.clamp(linear_vel, -1, 1) * env.linear_speed
                    angular_vel = torch.clamp(angular_vel, -1, 1) * 90
                    actions = torch.stack([linear_vel, angular_vel], dim=1)
                    if strategy_params['action_distribution'] == "stochastic":
                        noise_scale = 0.1
                        actions += torch.randn_like(actions) * noise_scale
                        actions[:, 0] = torch.clamp(actions[:, 0], -env.linear_speed, env.linear_speed)
                        actions[:, 1] = torch.clamp(actions[:, 1], -90, 90)
                
                # Step environment – actions are already GPU tensors
                obs, rewards, dones, info = env.step(actions, action_space=strategy_params['action_space'])
                steps += 1
                
                # GPU‑only goal tracking
                goal_reached = info["goal_reached"]
                new_successes = goal_reached & ~reached_goal
                if torch.any(new_successes):
                    new_indices = torch.where(new_successes)[0]
                    reached_goal[new_indices] = True
                    success_steps[new_indices] = steps
                    success_energy[new_indices] = env.energy[new_indices].float()
                    success_health[new_indices] = env.health[new_indices].float()
                
                percent_done = (torch.sum(reached_goal).item() / strategy_params['population_size']) * 100
                if steps % 10 == 0:
                    gen_pbar.set_postfix({"Step": steps, "%Done": f"{percent_done:.1f}%"})
                
                if env.render_flag:
                    env.render()
                
                if torch.all(dones):
                    break
        
        gen_duration = time.time() - gen_start_time
        
        # Curriculum logic
        curriculum_promoted = False
        curriculum_active = (strategy_params['network_type'] == "feedforward" and strategy_params['curriculum_enabled'])
        success_rate = (torch.sum(reached_goal).item() / strategy_params['population_size']) * 100
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
        
        # Save top‑k individuals (unchanged)
        if strategy_params['save_top_k'] > 0 and torch.any(reached_goal):
            survivor_indices_cpu = torch.where(reached_goal)[0].cpu().tolist()
            random.shuffle(survivor_indices_cpu)
            top_k = min(strategy_params['save_top_k'], len(survivor_indices_cpu))
            genomes = pop_brain.get_weights()
            for idx in survivor_indices_cpu[:top_k]:
                weight_path = os.path.join(top_k_dir, f"ind_{idx}.json")
                weight_data = {
                    "individual_id": int(idx),
                    "generation": gen,
                    "trial": trial_idx,
                    "steps_to_success": int(success_steps[idx].item()),
                    "energy_remaining": float(success_energy[idx].item()),
                    "health_remaining": float(success_health[idx].item()),
                    "initial_distance_to_reward": float(initial_distances[idx]),
                    "initial_position": {"x": float(initial_poses[idx][0]), "y": float(initial_poses[idx][1])},
                    "reward_position": {"x": float(reward_positions[idx][0]), "y": float(reward_positions[idx][1])},
                    "weights": genomes[idx].cpu().tolist()
                }
                with open(weight_path, 'w') as wf:
                    json.dump(weight_data, wf, indent=2)
        
        # Finalize generation – pass curriculum info
        has_survivors = strategy_instance.finalize_generation(
            env, reached_goal, success_steps, success_energy, success_health,
            curriculum_streak=curriculum_streak if curriculum_active else None,
            curriculum_promoted=curriculum_promoted
        )
        
        gen_pbar.set_postfix({"Success": f"{success_rate:.1f}%", "Total_Steps": steps, "Sec/Gen": f"{gen_duration:.1f}s"})
        
        if not has_survivors:
            gen_pbar.write(f"[EXTINCT] Trial {trial_idx} Gen {gen} - No survivors.")
            break
        
        # Reproduction: random selection (all operations inside no_grad)
        with torch.no_grad():
            survivor_indices = torch.where(reached_goal)[0]
            if len(survivor_indices) > 0:
                # Get genomes of survivors as a 2D tensor
                survivor_genomes = pop_brain.get_weights()[survivor_indices]

                # Get `population_size` random indices from survivors
                parent_idx = torch.randint(0, len(survivor_indices), (strategy_params['population_size'],), device=device)

                # Clone the selected genomes to create `population_size` children
                child_genomes = survivor_genomes[parent_idx].clone()

                # --- Add recombination (averaging) here ---
                if strategy_params['recombination_enabled']:
                    child_genomes = _average_recombination(child_genomes)

                # Vectorised mutation
                mutation_mask = torch.rand_like(child_genomes) < strategy_params['mutation_rate']
                noise = torch.randn_like(child_genomes) * strategy_params['mutation_mag']
                child_genomes[mutation_mask] += noise[mutation_mask]

                pop_brain.set_weights(child_genomes)
    
    # Optional: clear GPU cache after trial to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    env.close()
    # Finalize trial logging (mixin)
    strategy_instance._log_trial_complete(env)


# ---------------------------------------------------------------------------
# Neuroevolution Strategy with Extinction Logic (Vectorized)
# ---------------------------------------------------------------------------

class NNStrategy(BaseStrategy, StrategyLoggingMixin):
    def __init__(self, 
                num_trials=3, 
                generations=20, 
                population_size=50, 
                mutation_rate=0.2, 
                mutation_mag=0.5, 
                recombination_enabled=False,
                n_steps = 5,
                max_samples_per_gen=10,
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
                device='cuda',
                **network_params):
        params = {
            "pop_size": population_size,
            "generations": generations,
            "num_trials": num_trials,
            "mutation_rate": mutation_rate,
            "mutation_mag": mutation_mag,
            "recombination_enabled": recombination_enabled,
            "n_steps": n_steps,
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
            "device": device
        }
        super().__init__(strategy_name, params, parallel_trials=parallel_trials)
        self.pop_size = population_size
        self.max_gens = generations
        self.num_trials = num_trials
        self.mutation_rate = mutation_rate
        self.mutation_mag = mutation_mag
        self.recombination_enabled = recombination_enabled
        self.n_steps = n_steps
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
        self.device = device

    def run(self, env_params, base_output_dir):
        # Gather strategy parameters for helper
        strategy_params = {
            'population_size': self.pop_size,
            'max_gens': self.max_gens,
            'num_trials': self.num_trials,
            'mutation_rate': self.mutation_rate,
            'mutation_mag': self.mutation_mag,
            "recombination_enabled": self.recombination_enabled,
            "n_steps": self.n_steps,
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
                args_list = [(self, trial_idx+1, strategy_params, env_params, base_output_dir)
                            for trial_idx in range(self.num_trials)]
                pool.starmap(_run_single_trial, args_list)
        else:
            for trial_idx in range(1, self.num_trials + 1):
                _run_single_trial(self, trial_idx, strategy_params, env_params, base_output_dir)