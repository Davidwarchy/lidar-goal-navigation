import torch
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from .base_strategy import BaseStrategy
from .log import StrategyLoggingMixin
from env import VectorRobotExplorationEnv

class UniformRunLengthStrategy(BaseStrategy, StrategyLoggingMixin):
    def __init__(self, min_step=1, max_step=200, num_trials=1, max_generations=None, parallel_trials=False):
        params = {
            "min_step": int(min_step), 
            "max_step": int(max_step),
            "num_trials": num_trials,
            "max_generations": max_generations
        }
        super().__init__("uniform", params, parallel_trials=parallel_trials)
        self.min_step = int(min_step)
        self.max_step = int(max_step)
        self.num_trials = num_trials
        self.max_generations = max_generations

    def run(self, env_params, base_output_dir):
        if self.parallel_trials:
            trial_args = [(trial+1, env_params, base_output_dir, self.max_generations)
                          for trial in range(self.num_trials)]
            with mp.Pool(processes=self.num_trials) as pool:
                pool.starmap(_run_uniform_trial, trial_args)
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
            steps_left = torch.randint(self.min_step, self.max_step + 1, (env.num_envs,), device=env.device)
            direction = torch.randint(0, 4, (env.num_envs,), device=env.device)
            # Track which agents have reached goal in this generation
            goal_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            
            while not torch.all(env.done):
                actions = direction.cpu().numpy()
                obs, rewards, dones, info = env.step(actions, action_space="discrete")
                
                # Track newly reached goals
                goal_reached_np = info["goal_reached"]
                new_goals = torch.from_numpy(goal_reached_np).bool().to(env.device) & ~goal_reached
                goal_reached = goal_reached | torch.from_numpy(goal_reached_np).bool().to(env.device)
                
                steps_left -= 1
                need_new = steps_left <= 0
                
                if torch.any(need_new):
                    need_new_indices = torch.where(need_new)[0]
                    new_directions = torch.randint(0, 4, (len(need_new_indices),), device=env.device)
                    direction[need_new_indices] = new_directions
                    steps_left[need_new_indices] = torch.randint(
                        self.min_step, self.max_step + 1, (len(need_new_indices),), device=env.device
                    )
                
                # Log this step
                self._log_step(
                    env.current_step, 
                    (~env.done).cpu().numpy(), 
                    new_goals.cpu().numpy(), 
                    env
                )
                
                if env.render_flag:
                    env.render()
                
                percent_done = (torch.sum(goal_reached).item() / env.num_envs) * 100
                # Only update tqdm every 100 steps (or every step if verbose)
                if env.verbose or (env.current_step % 100 == 0):
                    gen_pbar.set_postfix({
                        "Gen": generation, 
                        "Step": env.current_step, 
                        "Success": f"{percent_done:.1f}%"
                    })
            
            has_survivors = self._log_generation_complete(env)
            gen_pbar.update(1)
            
            if not has_survivors:
                trial_extinct = True
                self.trial_extinct = True
            generation += 1
        
        gen_pbar.close()
        self._log_trial_complete(env)
        env.close()

def _run_uniform_trial(trial_num, env_params, base_output_dir, max_generations):
    strategy = UniformRunLengthStrategy(num_trials=1, max_generations=max_generations, parallel_trials=False)
    strategy._run_single_trial(trial_num, env_params, base_output_dir, max_generations)