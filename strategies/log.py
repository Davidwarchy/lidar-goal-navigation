"""
Logging mixin for strategies – clean hierarchical logging.
- Run level: metadata.json (already created by main.py)
- Trial level: trial_metadata.json, trial_summary.csv (includes curriculum)
- Generation level: log.json, agent_stats.csv
- No per-step logging, no duplicate files
"""

import os
import csv
import json
import time
from datetime import datetime
import numpy as np
import torch


class StrategyLoggingMixin:
    """
    Mixin class that adds logging capabilities to any strategy.
    Provides clean, organized logging without per-step overhead.
    """
    
    def setup_trial_logging(self, env, trial_num, total_trials, max_generations=None):
        """
        Set up logging directories and files for a trial.
        Assumes env.output_dir is already the trial directory.
        
        Args:
            env: The environment (has output_dir attribute)
            trial_num: Current trial number (1-indexed)
            total_trials: Total number of trials to run
            max_generations: Maximum generations per trial (if None, run until extinction)
        """
        # Create trial directory
        self.trial_num = trial_num
        self.total_trials = total_trials
        self.max_generations = max_generations
        
        # Use env.output_dir directly – it should already be the trial directory
        self.trial_dir = env.output_dir
        os.makedirs(self.trial_dir, exist_ok=True)
        
        # Trial-level metadata (only one metadata file per trial)
        self.trial_metadata = {
            "run_datetime": datetime.now().isoformat(),
            "strategy_name": self.name,
            "strategy_parameters": env.strategy_parameters,
            "trial_number": trial_num,
            "total_trials": total_trials,
            "max_generations": max_generations,
            "num_envs": env.num_envs,
            "max_steps": env.max_steps,
            "continue_after_goal": env.continue_after_goal,
            "initial_goal_spawn_distance": env.goal_spawn_dist,
            "environment_parameters": {
                "grid_width": env.grid_width,
                "grid_height": env.grid_height,
                "robot_radius": env.robot_radius,
                "num_rays": env.num_rays,
                "ray_length": env.ray_length,
                "map_image": os.path.basename(env.map_image_path),
                "use_lut": env.use_lut
            },
            "curriculum_enabled": False,  # Override in subclasses if needed
            "curriculum_success_threshold": None,
            "curriculum_consecutive_gens": None,
            "curriculum_distance_increment": None
        }
        
        # Save trial metadata to trial_metadata.json (single source)
        metadata_path = os.path.join(self.trial_dir, "trial_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.trial_metadata, f, indent=4)
        
        # Trial-level summary CSV (one row per generation)
        self.trial_summary_path = os.path.join(self.trial_dir, "trial_summary.csv")
        with open(self.trial_summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "generation", "success_rate_percent", "num_successful", 
                "avg_path_length", "avg_energy_remaining", "avg_health_remaining",
                "avg_initial_distance_to_reward", "gen_duration_seconds", 
                "total_steps_in_gen", "extinct", "goal_spawn_distance",
                "curriculum_streak", "curriculum_promoted"
            ])
        
        self.trial_start_time = time.time()
        self.trial_generations_completed = 0
        self.trial_extinct = False
        self.trial_curriculum_streak = 0
        self.trial_goal_distance_history = []  # Track goal distance over generations
    
    def set_curriculum_params(self, enabled, success_threshold, consecutive_gens, distance_increment):
        """
        Set curriculum parameters for the trial (called by strategies that use curriculum).
        
        Args:
            enabled: bool
            success_threshold: float (e.g., 0.05 for 5%)
            consecutive_gens: int
            distance_increment: float
        """
        self.trial_metadata["curriculum_enabled"] = enabled
        self.trial_metadata["curriculum_success_threshold"] = success_threshold
        self.trial_metadata["curriculum_consecutive_gens"] = consecutive_gens
        self.trial_metadata["curriculum_distance_increment"] = distance_increment
        
        # Update trial_metadata.json
        metadata_path = os.path.join(self.trial_dir, "trial_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.trial_metadata, f, indent=4)
    
    def setup_generation_logging(self, env, generation_num):
        """
        Set up logging for a single generation.
        
        Args:
            env: The environment
            generation_num: Current generation number (1-indexed)
        """
        self.generation_num = generation_num
        
        # Create generation directory
        self.gen_dir = os.path.join(self.trial_dir, f"gen_{generation_num}")
        os.makedirs(self.gen_dir, exist_ok=True)
        
        # CSV for per-agent final stats in this generation
        self.agent_stats_path = os.path.join(self.gen_dir, "agent_stats.csv")
        self.agent_stats_file = open(self.agent_stats_path, 'w', newline='')
        agent_writer = csv.writer(self.agent_stats_file)
        agent_writer.writerow([
            "agent_id", "goal_reached", "steps_to_success", 
            "final_energy", "final_health", "initial_distance_to_goal"
        ])
        self.agent_writer = agent_writer
        
        # Track state for this generation
        self.gen_start_time = time.time()
        self.gen_initial_distances = []
        
        # Track successes across trial
        if not hasattr(self, 'trial_goals_reached'):
            self.trial_goals_reached = set()
            self.trial_agent_success_data = {}
    
    def _log_initial_agent_data(self, env):
        """
        Log initial positions and distances for all agents.
        Args:
            env: The environment
        """
        self.gen_initial_distances = []
        for i in range(env.num_envs):
            dx = (env.robot_x[i] - env.goal_x[i]).cpu().item()
            dy = (env.robot_y[i] - env.goal_y[i]).cpu().item()
            dist = np.sqrt(dx*dx + dy*dy)
            self.gen_initial_distances.append(dist)
            
            energy_val = env.energy[i].cpu().item()
            health_val = env.health[i].cpu().item()
            
            # Write initial row for agent (goal_reached = False initially)
            self.agent_writer.writerow([
                i, False, -1, energy_val, health_val, dist
            ])
        self.agent_stats_file.flush()
    
    def finalize_generation(self, env, reached_goal, success_steps, success_energy, success_health, 
                           curriculum_streak=None, curriculum_promoted=False):
        """
        Called at generation end with GPU tensors. Writes all logs.
        
        Args:
            env: The environment
            reached_goal: (num_envs,) bool tensor on GPU
            success_steps: (num_envs,) long tensor on GPU
            success_energy: (num_envs,) float tensor on GPU
            success_health: (num_envs,) float tensor on GPU
            curriculum_streak: int (if using curriculum, otherwise None)
            curriculum_promoted: bool (if curriculum was promoted this generation)
        
        Returns:
            bool: True if there were survivors
        """
        gen_duration = time.time() - self.gen_start_time
        
        # Get successful agents
        survivor_indices_gpu = torch.where(reached_goal)[0]
        survivor_indices = survivor_indices_gpu.cpu().numpy()
        
        # Write success rows to agent_stats (one per successful agent)
        for idx in survivor_indices:
            steps = success_steps[idx].item()
            energy = success_energy[idx].item()
            health = success_health[idx].item()
            init_dist = self.gen_initial_distances[idx]
            # We need to update the row – easiest: write a new row with goal_reached=True
            # The CSV already has a row with False; we add a second row with True.
            self.agent_writer.writerow([
                idx, True, steps, energy, health, init_dist
            ])
        self.agent_stats_file.close()
        
        # Calculate generation statistics
        success_rate = (len(survivor_indices) / env.num_envs) * 100
        
        path_lengths = [success_steps[idx].item() for idx in survivor_indices if success_steps[idx].item() > 0]
        energies = [success_energy[idx].item() for idx in survivor_indices]
        healths = [success_health[idx].item() for idx in survivor_indices]
        initial_dists = [self.gen_initial_distances[idx] for idx in survivor_indices]
        
        # Get curriculum values
        if curriculum_streak is None:
            curriculum_streak_val = 0
            curriculum_promoted_val = False
        else:
            curriculum_streak_val = curriculum_streak
            curriculum_promoted_val = curriculum_promoted
            if curriculum_promoted:
                self.trial_curriculum_streak = 0
            else:
                self.trial_curriculum_streak = curriculum_streak
        
        # Track goal distance history
        self.trial_goal_distance_history.append(env.goal_spawn_dist)
        
        # Generation log.json (includes curriculum info)
        gen_stats = {
            "generation": self.generation_num,
            "trial": self.trial_num,
            "timestamp": datetime.now().isoformat(),
            "success_rate_percent": success_rate,
            "num_successful": len(survivor_indices),
            "population_size": env.num_envs,
            "avg_path_length": float(np.mean(path_lengths)) if path_lengths else 0,
            "avg_energy_remaining": float(np.mean(energies)) if energies else 0,
            "avg_health_remaining": float(np.mean(healths)) if healths else 0,
            "avg_initial_distance_to_reward": float(np.mean(initial_dists)) if initial_dists else 0,
            "generation_duration_seconds": gen_duration,
            "total_steps_in_gen": env.current_step,
            "extinct": len(survivor_indices) == 0,
            "goal_spawn_distance": float(env.goal_spawn_dist),
            "curriculum_streak": int(curriculum_streak_val),
            "curriculum_promoted": bool(curriculum_promoted_val)
        }
        
        with open(os.path.join(self.gen_dir, "log.json"), 'w') as f:
            json.dump(gen_stats, f, indent=2)
        
        # Append to trial summary CSV (includes curriculum columns)
        with open(self.trial_summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.generation_num, f"{success_rate:.2f}%", len(survivor_indices),
                f"{np.mean(path_lengths) if path_lengths else 0:.2f}",
                f"{np.mean(energies) if energies else 0:.2f}",
                f"{np.mean(healths) if healths else 0:.2f}",
                f"{np.mean(initial_dists) if initial_dists else 0:.2f}",
                f"{gen_duration:.2f}", env.current_step,
                len(survivor_indices) == 0,
                f"{env.goal_spawn_dist:.2f}",
                curriculum_streak_val,
                "yes" if curriculum_promoted_val else "no"
            ])
        
        # Update trial tracking
        self.trial_generations_completed += 1
        for idx in survivor_indices:
            if idx not in self.trial_goals_reached:
                self.trial_goals_reached.add(idx)
                self.trial_agent_success_data[idx] = {
                    "steps_to_success": success_steps[idx].item(),
                    "final_energy": success_energy[idx].item(),
                    "final_health": success_health[idx].item()
                }
        
        return len(survivor_indices) > 0
    
    def _log_trial_complete(self, env):
        """
        Log trial completion and save final statistics.
        Args:
            env: The environment
        """
        trial_duration = time.time() - self.trial_start_time
        
        trial_summary = {
            "trial_number": self.trial_num,
            "total_trials": self.total_trials,
            "completed_generations": self.trial_generations_completed,
            "max_generations": self.max_generations,
            "extinct": self.trial_extinct,
            "trial_duration_seconds": trial_duration,
            # Convert NumPy types to standard Python types here:
            "unique_successful_agents": int(len(self.trial_goals_reached)),
            "successful_agents_across_trial": [int(idx) for idx in self.trial_goals_reached],
            "final_curriculum_streak": self.trial_curriculum_streak,
            "final_goal_distance": env.goal_spawn_dist,
            "goal_distance_history": self.trial_goal_distance_history
        }
        
        # Calculate trial-level success metrics
        if self.trial_agent_success_data:
            steps = [data["steps_to_success"] for data in self.trial_agent_success_data.values()]
            energies = [data["final_energy"] for data in self.trial_agent_success_data.values()]
            healths = [data["final_health"] for data in self.trial_agent_success_data.values()]
            
            trial_summary["avg_success_steps"] = float(np.mean(steps))
            trial_summary["avg_success_energy"] = float(np.mean(energies))
            trial_summary["avg_success_health"] = float(np.mean(healths))
        
        # Update trial_metadata.json with completion info
        self.trial_metadata["completed_generations"] = self.trial_generations_completed
        self.trial_metadata["extinct"] = self.trial_extinct
        self.trial_metadata["trial_duration_seconds"] = trial_duration
        self.trial_metadata["unique_successful_agents"] = int(len(self.trial_goals_reached))
        self.trial_metadata["final_goal_distance"] = env.goal_spawn_dist
        self.trial_metadata["goal_distance_history"] = self.trial_goal_distance_history
        
        with open(os.path.join(self.trial_dir, "trial_metadata.json"), 'w') as f:
            json.dump(self.trial_metadata, f, indent=4)
        
        print(f"\n[TRIAL {self.trial_num}/{self.total_trials}] Complete. "
              f"Generations: {self.trial_generations_completed}, "
              f"Successful agents: {len(self.trial_goals_reached)}")
        
        if self.trial_metadata.get("curriculum_enabled", False):
            print(f"  Curriculum: Final goal distance = {env.goal_spawn_dist:.2f}")