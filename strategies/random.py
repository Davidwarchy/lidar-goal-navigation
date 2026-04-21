import numpy as np
import os
import multiprocessing as mp
from .base_strategy import BaseStrategy
from .log import StrategyLoggingMixin
from env import VectorRobotExplorationEnv

class RandomWalkStrategy(BaseStrategy, StrategyLoggingMixin):
    def __init__(self, num_trials=1, max_generations=None, parallel_trials=False):
        params = {"num_trials": num_trials, "max_generations": max_generations}
        super().__init__("random", params, parallel_trials=parallel_trials)
        self.num_trials = num_trials
        self.max_generations = max_generations
    
    def run(self, env_params, base_output_dir):
        if self.parallel_trials:
            trial_args = [(trial+1, env_params, base_output_dir, self.max_generations)
                          for trial in range(self.num_trials)]
            with mp.Pool(processes=self.num_trials) as pool:
                pool.starmap(_run_random_trial, trial_args)
        else:
            for trial_num in range(1, self.num_trials + 1):
                self._run_single_trial(trial_num, env_params, base_output_dir, self.max_generations)
    
    def _run_single_trial(self, trial_num, env_params, base_output_dir, max_generations):
        trial_dir = os.path.join(base_output_dir, f"trial_{trial_num}")
        env_params_with_out = env_params.copy()
        env_params_with_out["output_dir"] = trial_dir
        env = VectorRobotExplorationEnv(**env_params_with_out)
        
        self.setup_trial_logging(env, trial_num, self.num_trials, max_generations)
        # ... rest of trial loop unchanged ...
        generation = 1
        trial_extinct = False
        # Run generations until extinction or max_generations
        while not trial_extinct:
            if max_generations and generation > max_generations:
                break
            # Setup generation logging
            self.setup_generation_logging(env, generation)
            # Reset environment for this generation
            obs = env.reset()
            self._log_initial_agent_data(env)
            # Track which agents have reached goal in this generation
            goal_reached = np.zeros(env.num_envs, dtype=bool)
            # Run this generation
            while not np.all(env.done):
                actions = np.random.randint(0, 4, size=env.num_envs)
                obs, rewards, dones, info = env.step(actions, action_space="discrete")
                # Track newly reached goals
                new_goals = info["goal_reached"] & ~goal_reached
                goal_reached |= info["goal_reached"]
                # Log this step
                active_mask = ~env.done
                self._log_step(env.current_step, active_mask, new_goals, env)
                if env.render_flag:
                    env.render()
                if env.verbose and env.current_step % 20 == 0:
                    active = np.sum(~env.done)
                    print(f"Trial {trial_num}, Gen {generation}, Step {env.current_step}, Active agents: {active}")
            # Log generation completion
            has_survivors = self._log_generation_complete(env)
            if not has_survivors:
                trial_extinct = True
                self.trial_extinct = True
            generation += 1
            # Log trial completion
        self._log_trial_complete(env)
        env.close()

def _run_random_trial(trial_num, env_params, base_output_dir, max_generations):
    strategy = RandomWalkStrategy(num_trials=1, max_generations=max_generations, parallel_trials=False)
    strategy._run_single_trial(trial_num, env_params, base_output_dir, max_generations)