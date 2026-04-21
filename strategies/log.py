"""
Logging mixin for strategies to provide consistent logging without natural selection.
Supports trials and generations structure.
"""

import os
import csv
import json
import time
from datetime import datetime
import numpy as np


class StrategyLoggingMixin:
    """
    Mixin class that adds logging capabilities to any strategy.
    Does NOT include natural selection logic - just iteration tracking and logs.
    Supports multiple trials and generations per trial.
    """
    
    def setup_trial_logging(self, env, trial_num, total_trials, max_generations=None):
        """
        Set up logging directories and files for a trial.
        Assumes env.output_dir is already the trial directory (e.g., base_output_dir/trial_X).
        
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
        
        # Trial-level metadata
        self.trial_metadata = {
            "run_datetime": datetime.now().isoformat(),
            "strategy_name": self.name,
            "trial_number": trial_num,
            "total_trials": total_trials,
            "max_generations": max_generations,
            "num_envs": env.num_envs,
            "max_steps": env.max_steps,
            "continue_after_goal": env.continue_after_goal,
            "environment_parameters": {
                "grid_width": env.grid_width,
                "grid_height": env.grid_height,
                "robot_radius": env.robot_radius,
                "num_rays": env.num_rays,
                "ray_length": env.ray_length,
                "map_image": os.path.basename(env.map_image_path),
                "use_lut": env.use_lut
            }
        }
        
        # Save trial metadata
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
                "total_steps_in_gen", "extinct"
            ])
        
        # Track trial-level stats
        self.trial_start_time = time.time()
        self.trial_generations_completed = 0
        self.trial_extinct = False
    
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
        
        # CSV log for this generation (step-level data)
        self.step_log_path = os.path.join(self.gen_dir, "step_log.csv")
        self.step_log_file = open(self.step_log_path, 'w', newline='')
        step_writer = csv.writer(self.step_log_file)
        step_writer.writerow(["step", "active_agents", "goals_reached_cumulative", "new_goals_this_step"])
        self.step_writer = step_writer
        
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
        self.gen_goals_reached = set()
        self.gen_initial_distances = []
        self.gen_agent_success_data = {}
        
        # Also track for trial-level cumulative stats
        if not hasattr(self, 'trial_goals_reached'):
            self.trial_goals_reached = set()
            self.trial_agent_success_data = {}
    
    def _log_step(self, step, active_mask, goal_reached_mask, env):
        """
        Log a single step of execution within a generation.
        
        Args:
            step: Current step number
            active_mask: Boolean array of which agents are still active
            goal_reached_mask: Boolean array of which agents reached goal this step
            env: The environment
        """
        active_count = np.sum(active_mask)
        
        # Track new goals in this generation
        new_goal_indices = np.where(goal_reached_mask)[0]
        for idx in new_goal_indices:
            if idx not in self.gen_goals_reached:
                self.gen_goals_reached.add(idx)
                
                # Store success data for this agent
                self.gen_agent_success_data[idx] = {
                    "steps_to_success": step,
                    "final_energy": env.energy[idx],
                    "final_health": env.health[idx]
                }
                
                # Write to agent stats CSV immediately
                init_dist = self.gen_initial_distances[idx] if idx < len(self.gen_initial_distances) else -1
                self.agent_writer.writerow([
                    idx, True, step, env.energy[idx], env.health[idx], init_dist
                ])
                self.agent_stats_file.flush()
        
        # Log step
        self.step_writer.writerow([
            step, active_count, len(self.gen_goals_reached), len(new_goal_indices)
        ])
        self.step_log_file.flush()
    
    def _log_initial_agent_data(self, env):
        """
        Log initial positions and distances for all agents in this generation.
        
        Args:
            env: The environment
        """
        self.gen_initial_distances = []
        for i in range(env.num_envs):
            dx = env.robot_x[i] - env.goal_x[i]
            dy = env.robot_y[i] - env.goal_y[i]
            dist = np.sqrt(dx*dx + dy*dy)
            self.gen_initial_distances.append(dist)
            
            # Write initial row for agent (goal_reached = False initially)
            self.agent_writer.writerow([
                i, False, -1, env.energy[i], env.health[i], dist
            ])
        self.agent_stats_file.flush()
    
    def _log_generation_complete(self, env):
        """
        Log generation completion and save statistics.
        
        Args:
            env: The environment
            
        Returns:
            bool: True if there were survivors (can continue to next generation), False if extinct
        """
        gen_duration = time.time() - self.gen_start_time
        
        # Close generation file handles
        self.step_log_file.close()
        self.agent_stats_file.close()
        
        # Calculate generation statistics
        survivor_indices = list(self.gen_goals_reached)
        success_rate = (len(survivor_indices) / env.num_envs) * 100
        
        path_lengths = [self.gen_agent_success_data[idx]["steps_to_success"] for idx in survivor_indices]
        energies = [self.gen_agent_success_data[idx]["final_energy"] for idx in survivor_indices]
        healths = [self.gen_agent_success_data[idx]["final_health"] for idx in survivor_indices]
        initial_dists = [self.gen_initial_distances[idx] for idx in survivor_indices]
        
        # Save generation log.json
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
            "extinct": len(survivor_indices) == 0
        }
        
        with open(os.path.join(self.gen_dir, "log.json"), 'w') as f:
            json.dump(gen_stats, f, indent=2)
        
        # Append to trial summary CSV
        with open(self.trial_summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.generation_num, f"{success_rate:.2f}%", len(survivor_indices),
                f"{np.mean(path_lengths) if path_lengths else 0:.2f}",
                f"{np.mean(energies) if energies else 0:.2f}",
                f"{np.mean(healths) if healths else 0:.2f}",
                f"{np.mean(initial_dists) if initial_dists else 0:.2f}",
                f"{gen_duration:.2f}", env.current_step,
                len(survivor_indices) == 0
            ])
        
        # Update trial tracking
        self.trial_generations_completed += 1
        for idx in survivor_indices:
            if idx not in self.trial_goals_reached:
                self.trial_goals_reached.add(idx)
                self.trial_agent_success_data[idx] = self.gen_agent_success_data[idx]
        
        return len(survivor_indices) > 0
    
    def _log_trial_complete(self, env):
        """
        Log trial completion and save final statistics.
        
        Args:
            env: The environment
        """
        trial_duration = time.time() - self.trial_start_time
        
        # Trial summary
        trial_summary = {
            "trial_number": self.trial_num,
            "total_trials": self.total_trials,
            "completed_generations": self.trial_generations_completed,
            "max_generations": self.max_generations,
            "extinct": self.trial_extinct,
            "trial_duration_seconds": trial_duration,
            # Convert NumPy types to standard Python types here:
            "unique_successful_agents": int(len(self.trial_goals_reached)),
            "successful_agents_across_trial": [int(idx) for idx in self.trial_goals_reached]
        }
        
        # Calculate trial-level success metrics
        if self.trial_agent_success_data:
            steps = [data["steps_to_success"] for data in self.trial_agent_success_data.values()]
            energies = [data["final_energy"] for data in self.trial_agent_success_data.values()]
            healths = [data["final_health"] for data in self.trial_agent_success_data.values()]
            
            trial_summary["avg_success_steps"] = float(np.mean(steps))
            trial_summary["avg_success_energy"] = float(np.mean(energies))
            trial_summary["avg_success_health"] = float(np.mean(healths))
        
        with open(os.path.join(self.trial_dir, "trial_complete.json"), 'w') as f:
            json.dump(trial_summary, f, indent=4)
        
        # Human-readable trial summary
        with open(os.path.join(self.trial_dir, "trial_summary.txt"), 'w') as f:
            f.write(f"=== Trial {self.trial_num}/{self.total_trials} Summary ===\n")
            f.write(f"Strategy: {self.trial_metadata['strategy_name']}\n")
            f.write(f"Duration: {trial_duration:.2f} seconds\n")
            f.write(f"Generations Completed: {self.trial_generations_completed}\n")
            f.write(f"Max Generations: {self.max_generations if self.max_generations else 'Unlimited'}\n")
            f.write(f"Extinct: {self.trial_extinct}\n")
            f.write(f"Unique Successful Agents: {len(self.trial_goals_reached)}\n")
            if self.trial_agent_success_data:
                f.write(f"Avg Success Steps: {trial_summary.get('avg_success_steps', 0):.2f}\n")
                f.write(f"Avg Success Energy Remaining: {trial_summary.get('avg_success_energy', 0):.2f}\n")
                f.write(f"Avg Success Health Remaining: {trial_summary.get('avg_success_health', 0):.2f}\n")
        
        print(f"\n[TRIAL {self.trial_num}/{self.total_trials}] Complete. "
              f"Generations: {self.trial_generations_completed}, "
              f"Successful agents: {len(self.trial_goals_reached)}")