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
import multiprocessing as mp


# ---------------------------------------------------------------------------
# Helper Functions for Selection & Reproduction
# ---------------------------------------------------------------------------

def _select_parents_by_survival(pop_brain, reached_goal, population_size, device):
    """
    Original selection: random choice from survivors (goal-reached agents).
    
    Returns:
        child_genomes: torch.Tensor of shape (population_size, total_params)
        has_parents: bool
    """
    survivor_indices = torch.where(reached_goal)[0]
    
    if len(survivor_indices) == 0:
        return None, False
    
    survivor_genomes = pop_brain.get_weights()[survivor_indices]
    parent_idx = torch.randint(0, len(survivor_indices), (population_size,), device=device)
    child_genomes = survivor_genomes[parent_idx].clone()
    
    return child_genomes, True


def _select_parents_by_fitness_proxy(pop_brain, env, reached_goal, 
                                      fitness_proxy, fitness_pool, 
                                      population_size, device):
    """
    Proxy-based selection: select top 10% by health or energy.
    
    Args:
        fitness_proxy: 'health' or 'energy'
        fitness_pool: 'all_agents' or 'survivors_only'
    
    Returns:
        child_genomes: torch.Tensor of shape (population_size, total_params)
        has_parents: bool
    """
    # Determine candidate pool
    if fitness_pool == 'survivors_only':
        candidate_mask = reached_goal
    else:  # 'all_agents'
        candidate_mask = torch.ones(population_size, dtype=torch.bool, device=device)
    
    candidate_indices = torch.where(candidate_mask)[0]
    
    if len(candidate_indices) == 0:
        return None, False
    
    # Get fitness values for candidates
    if fitness_proxy == 'health':
        fitness_values = env.health[candidate_indices].float()
    elif fitness_proxy == 'energy':
        fitness_values = env.energy[candidate_indices].float()
    else:
        raise ValueError(f"Unknown fitness proxy: {fitness_proxy}")
    
    # Select top 10% of candidates (minimum 1, maximum population_size)
    top_k = max(1, int(len(candidate_indices) * 0.1))  # Top 10%, at least 1
    top_k = min(top_k, population_size)  # Cap at population_size
    
    # Get indices of top performers
    top_k_values, top_k_indices_in_candidates = torch.topk(fitness_values, top_k)
    top_k_global_indices = candidate_indices[top_k_indices_in_candidates]
    
    # Get genomes of top performers
    selected_genomes = pop_brain.get_weights()[top_k_global_indices]
    
    # Repeat selected genomes to fill population
    repeats = population_size // top_k + (1 if population_size % top_k != 0 else 0)
    repeated = selected_genomes.repeat(repeats, 1)[:population_size]
    child_genomes = repeated.clone()
    
    # Optional: Add logging to see selection statistics
    # print(f"Selected top {top_k} out of {len(candidate_indices)} candidates ({top_k/len(candidate_indices)*100:.1f}%)")
    
    return child_genomes, True


def _apply_recombination_and_mutation(child_genomes, recombination_enabled, 
                                       mutation_rate, mutation_mag, device):
    """
    Apply crossover (recombination) and mutation to child genomes.
    
    Args:
        child_genomes: torch.Tensor of shape (population_size, total_params)
    
    Returns:
        child_genomes: mutated genomes
    """
    # Recombination (averaging crossover)
    if recombination_enabled:
        child_genomes = _average_recombination(child_genomes)
    
    # Vectorized mutation
    mutation_mask = torch.rand_like(child_genomes) < mutation_rate
    noise = torch.randn_like(child_genomes) * mutation_mag
    child_genomes[mutation_mask] += noise[mutation_mask]
    
    return child_genomes


def _reproduce_population(pop_brain, env, reached_goal, strategy_params, device, gen_pbar=None, trial_idx=None, gen=None):
    """
    Main reproduction orchestration function.
    
    Returns:
        bool: True if reproduction succeeded (has parents), False otherwise
    """
    fitness_proxy = strategy_params.get('fitness_proxy', 'none')
    fitness_pool = strategy_params.get('fitness_pool', 'survivors_only')
    population_size = strategy_params['population_size']
    
    with torch.no_grad():
        # Select parents based on configuration
        if fitness_proxy == 'none':
            child_genomes, has_parents = _select_parents_by_survival(
                pop_brain, reached_goal, population_size, device
            )
        else:
            child_genomes, has_parents = _select_parents_by_fitness_proxy(
                pop_brain, env, reached_goal, fitness_proxy, fitness_pool,
                population_size, device
            )
            
            # Optional: Log selection info
            if has_parents and gen_pbar is not None:
                gen_pbar.write(f"[SELECTION] Trial {trial_idx} Gen {gen}: "
                              f"Using top 10% by {fitness_proxy} from {fitness_pool}")
        
        if not has_parents:
            return False
        
        # Apply recombination and mutation
        child_genomes = _apply_recombination_and_mutation(
            child_genomes,
            strategy_params['recombination_enabled'],
            strategy_params['mutation_rate'],
            strategy_params['mutation_mag'],
            device
        )
        
        # Update population with new genomes
        pop_brain.set_weights(child_genomes)
        
    return True


def _save_top_k_individuals(top_k_dir, pop_brain, reached_goal, success_steps,
                             success_energy, success_health, initial_distances,
                             initial_poses, reward_positions, 
                             save_top_k, gen, trial_idx, device):
    """
    Save top-k successful individuals to disk.
    """
    if save_top_k == 0 or not torch.any(reached_goal):
        return
    
    survivor_indices_cpu = torch.where(reached_goal)[0].cpu().tolist()
    random.shuffle(survivor_indices_cpu)
    top_k = min(save_top_k, len(survivor_indices_cpu))
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


def _should_promote_curriculum(success_rate, curriculum_streak, strategy_params):
    """
    Determine if curriculum should be promoted based on success rate.
    
    Returns:
        tuple: (new_streak, promoted, new_goal_distance)
    """
    threshold = strategy_params['curriculum_success_threshold'] * 100.0
    consecutive_needed = strategy_params['curriculum_consecutive_gens']
    increment = strategy_params['curriculum_distance_increment']
    
    if success_rate >= threshold:
        curriculum_streak += 1
    else:
        curriculum_streak = 0
    
    promoted = False
    new_distance = None
    
    if curriculum_streak >= consecutive_needed:
        promoted = True
        curriculum_streak = 0
        new_distance = increment  # This will be added to current distance
    
    return curriculum_streak, promoted, new_distance


# ---------------------------------------------------------------------------
# Main Trial Function
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
        
        # Store initial positions for logging (CPU)
        initial_poses = [(env.robot_x[i].item(), env.robot_y[i].item()) 
                        for i in range(strategy_params['population_size'])]
        reward_positions = [(env.goal_x[i].item(), env.goal_y[i].item()) 
                           for i in range(strategy_params['population_size'])]
        
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
                
                # Action selection based on configuration
                if strategy_params['action_space'] == "discrete":
                    if strategy_params['action_distribution'] == "deterministic":
                        actions = torch.argmax(output, dim=1)
                    else:
                        exp_output = torch.exp(output - torch.max(output, dim=1, keepdim=True)[0])
                        probs = exp_output / torch.sum(exp_output, dim=1, keepdim=True)
                        actions = torch.multinomial(probs, 1).squeeze(1)
                else:  # continuous
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
                
                # Update progress bar
                if steps % 10 == 0:
                    percent_done = (torch.sum(reached_goal).item() / strategy_params['population_size']) * 100
                    gen_pbar.set_postfix({"Step": steps, "%Done": f"{percent_done:.1f}%"})
                
                if env.render_flag:
                    env.render()
                
                if torch.all(dones):
                    break
        
        gen_duration = time.time() - gen_start_time
        
        # --- Curriculum Logic (if enabled) ---
        curriculum_active = (strategy_params['network_type'] in ["feedforward", "spiking"] 
                            and strategy_params['curriculum_enabled'])
        success_rate = (torch.sum(reached_goal).item() / strategy_params['population_size']) * 100
        curriculum_promoted = False
        
        if curriculum_active:
            curriculum_streak, curriculum_promoted, distance_increment = _should_promote_curriculum(
                success_rate, curriculum_streak, strategy_params
            )
            if curriculum_promoted:
                env.goal_spawn_dist += distance_increment
                gen_pbar.write(f"[CURRICULUM] Trial {trial_idx} Gen {gen}: goal_spawn_dist -> {env.goal_spawn_dist:.2f}")
        
        # --- Save top-k individuals ---
        if strategy_params['save_top_k'] > 0:
            _save_top_k_individuals(
                top_k_dir, pop_brain, reached_goal, success_steps, success_energy, success_health,
                initial_distances, initial_poses, reward_positions,
                strategy_params['save_top_k'], gen, trial_idx, device
            )
        
        # --- Log generation results ---
        has_survivors = strategy_instance.finalize_generation(
            env, reached_goal, success_steps, success_energy, success_health,
            curriculum_streak=curriculum_streak if curriculum_active else None,
            curriculum_promoted=curriculum_promoted
        )
        
        gen_pbar.set_postfix({
            "Success": f"{success_rate:.1f}%", 
            "Total_Steps": steps, 
            "Sec/Gen": f"{gen_duration:.1f}s"
        })
        
        # --- Reproduction for next generation ---
        # If no survivors, trial ends
        if not has_survivors:
            gen_pbar.write(f"[EXTINCT] Trial {trial_idx} Gen {gen} - No survivors.")
            break
        
        # Reproduce population for next generation
        reproduction_success = _reproduce_population(
            pop_brain, env, reached_goal, strategy_params, device, gen_pbar, trial_idx, gen
        )
        
        if not reproduction_success:
            gen_pbar.write(f"[EXTINCT] Trial {trial_idx} Gen {gen} - No suitable parents for reproduction.")
            break
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    env.close()
    # Finalize trial logging (mixin)
    strategy_instance._log_trial_complete(env)


# ---------------------------------------------------------------------------
# Neuroevolution Strategy Class
# ---------------------------------------------------------------------------

class NNStrategy(BaseStrategy, StrategyLoggingMixin):
    def __init__(self, 
                num_trials=3, 
                generations=20, 
                population_size=50, 
                mutation_rate=0.2, 
                mutation_mag=0.5, 
                recombination_enabled=False,
                n_steps=5,
                max_samples_per_gen=10,
                network_type="spiking",
                save_top_k=0,
                curriculum_enabled=False,
                curriculum_success_threshold=0.05,
                curriculum_consecutive_gens=3,
                curriculum_distance_increment=5.0,
                action_space="discrete",
                action_distribution="deterministic",
                strategy_name="nn",
                parallel_trials=False,
                device='cuda',
                fitness_proxy="none",
                fitness_pool="survivors_only",
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
            "device": device,
            "fitness_proxy": fitness_proxy,
            "fitness_pool": fitness_pool
        }
        
        super().__init__(strategy_name, params, parallel_trials=parallel_trials)
        
        # Store attributes
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
        self.fitness_proxy = fitness_proxy
        self.fitness_pool = fitness_pool

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
            'action_distribution': self.action_distribution,
            'fitness_proxy': self.fitness_proxy,
            'fitness_pool': self.fitness_pool
        }
        
        if self.parallel_trials:
            with mp.Pool(processes=self.num_trials) as pool:
                args_list = [(self, trial_idx+1, strategy_params, env_params, base_output_dir)
                            for trial_idx in range(self.num_trials)]
                pool.starmap(_run_single_trial, args_list)
        else:
            for trial_idx in range(1, self.num_trials + 1):
                _run_single_trial(self, trial_idx, strategy_params, env_params, base_output_dir)