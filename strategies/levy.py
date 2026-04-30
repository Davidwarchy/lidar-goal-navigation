import torch
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from .base_strategy import BaseStrategy
from .log import StrategyLoggingMixin
from env import VectorRobotExplorationEnv

class LevyWalkStrategy(BaseStrategy, StrategyLoggingMixin):
    def __init__(self, alpha=1.6, min_step=1.0, max_step=200.0, num_trials=1, max_generations=None, parallel_trials=False):
        parameters = {
            "alpha": alpha,
            "min_step": min_step,
            "max_step": max_step,
            "num_trials": num_trials,
            "max_generations": max_generations
        }
        super().__init__("levy", parameters, parallel_trials=parallel_trials)
        self.alpha = alpha
        self.min_step = min_step
        self.max_step = max_step
        self.num_trials = num_trials
        self.max_generations = max_generations

    def _sample_pareto(self, n):
        """Vectorized Pareto sampling with truncation."""
        u = np.random.uniform(size=n)
        x = self.min_step * (1 - u) ** (-1.0 / self.alpha)
        # Clip to max_step (simpler than rejection)
        x = np.clip(x, self.min_step, self.max_step)
        return np.maximum(1, np.round(x)).astype(int)
    
    def run(self, env_params, base_output_dir):
        if self.parallel_trials:
            trial_args = [(trial+1, env_params, base_output_dir, self.max_generations)
                          for trial in range(self.num_trials)]
            with mp.Pool(processes=self.num_trials) as pool:
                pool.starmap(_run_levy_trial, trial_args)
        else:
            for trial_num in range(1, self.num_trials + 1):
                self._run_single_trial(trial_num, env_params, base_output_dir, self.max_generations)
    
    def _run_single_trial(self, trial_num, env_params, base_output_dir, max_generations):
        trial_dir = os.path.join(base_output_dir, f"trial_{trial_num}")
        env_params_with_out = env_params.copy()
        env_params_with_out["output_dir"] = trial_dir
        env = VectorRobotExplorationEnv(**env_params_with_out)
        
        self.setup_trial_logging(env, trial_num, self.num_trials, max_generations)
        
        generation = 1
        trial_extinct = False
        
        # Run generations until extinction or max_generations
        gen_pbar = tqdm(desc=f"Trial {trial_num}/{self.num_trials}", unit="gen", position=0)
        
        while not trial_extinct:
            if max_generations and generation > max_generations:
                break
            
            # Setup generation logging
            self.setup_generation_logging(env, generation)

            # Reset environment for this generation
            obs = env.reset()
            self._log_initial_agent_data(env)
            
            # Per-agent state
            steps_left = torch.tensor(self._sample_pareto(env.num_envs), device=env.device)
            direction = torch.randint(0, 4, (env.num_envs,), device=env.device)
            
            # GPU tensors for tracking successes
            reached_goal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            success_steps = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
            success_energy = torch.full((env.num_envs,), -1.0, dtype=torch.float32, device=env.device)
            success_health = torch.full((env.num_envs,), -1.0, dtype=torch.float32, device=env.device)
            
            # Run this generation
            while not torch.all(env.done):
                obs, rewards, dones, info = env.step(direction, action_space="discrete")
                
                # GPU‑only goal tracking
                new_goals = info["goal_reached"] & ~reached_goal
                if torch.any(new_goals):
                    new_indices = torch.where(new_goals)[0]
                    reached_goal[new_indices] = True
                    success_steps[new_indices] = env.current_step
                    success_energy[new_indices] = env.energy[new_indices].float()
                    success_health[new_indices] = env.health[new_indices].float()
                
                # Decrement and check for new runs
                steps_left -= 1
                need_new = steps_left <= 0
                
                if torch.any(need_new):
                    need_new_indices = torch.where(need_new)[0]
                    new_directions = torch.randint(0, 4, (len(need_new_indices),), device=env.device)
                    direction[need_new_indices] = new_directions
                    steps_left[need_new_indices] = torch.tensor(
                        self._sample_pareto(len(need_new_indices)), device=env.device
                    )
                
                if env.render_flag:
                    env.render()
                
                if env.verbose or (env.current_step % 100 == 0):
                    percent_done = (torch.sum(reached_goal).item() / env.num_envs) * 100
                    gen_pbar.set_postfix({
                        "Gen": generation,
                        "Step": env.current_step,
                        "Success": f"{percent_done:.1f}%"
                    })
            
            # Generation ended – now write all success data at once
            has_survivors = self.finalize_generation(env, reached_goal, success_steps, success_energy, success_health)
            gen_pbar.update(1)
            
            if not has_survivors:
                trial_extinct = True
                self.trial_extinct = True
            generation += 1
        
        # Log trial completion
        gen_pbar.close()
        self._log_trial_complete(env)
        env.close()


def _run_levy_trial(trial_num, env_params, base_output_dir, max_generations):
    strategy = LevyWalkStrategy(num_trials=1, max_generations=max_generations, parallel_trials=False)
    strategy._run_single_trial(trial_num, env_params, base_output_dir, max_generations)