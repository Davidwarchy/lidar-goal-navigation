import numpy as np
from .base_strategy import BaseStrategy
from .log import StrategyLoggingMixin


class UniformRunLengthStrategy(BaseStrategy, StrategyLoggingMixin):
    def __init__(self, min_step=1, max_step=200, num_trials=1, max_generations=None):
        params = {
            "min_step": int(min_step), 
            "max_step": int(max_step),
            "num_trials": num_trials,
            "max_generations": max_generations
        }
        super().__init__("uniform", params)
        self.min_step = int(min_step)
        self.max_step = int(max_step)
        self.num_trials = num_trials
        self.max_generations = max_generations

    def run(self, env):
        for trial in range(1, self.num_trials + 1):
            # Setup trial logging
            self.setup_trial_logging(env, self.name, trial, self.num_trials, self.max_generations)
            
            generation = 1
            trial_extinct = False
            
            # Run generations until extinction or max_generations
            while not trial_extinct:
                if self.max_generations and generation > self.max_generations:
                    break
                
                # Setup generation logging
                self.setup_generation_logging(env, generation)
                
                # Reset environment for this generation
                obs = env.reset()
                self._log_initial_agent_data(env)
                
                # Per-agent state
                steps_left = np.random.randint(self.min_step, self.max_step + 1, size=env.num_envs)
                direction = np.random.randint(0, 4, size=env.num_envs)
                
                # Track which agents have reached goal in this generation
                goal_reached = np.zeros(env.num_envs, dtype=bool)
                
                # Run this generation
                while not np.all(env.done):
                    actions = direction.copy()
                    obs, rewards, dones, info = env.step(actions)
                    
                    # Track newly reached goals
                    new_goals = info["goal_reached"] & ~goal_reached
                    goal_reached |= info["goal_reached"]
                    
                    steps_left -= 1
                    need_new = steps_left <= 0
                    
                    if np.any(need_new):
                        direction[need_new] = np.random.randint(0, 4, size=np.sum(need_new))
                        steps_left[need_new] = np.random.randint(self.min_step, self.max_step + 1, 
                                                                 size=np.sum(need_new))
                    
                    # Log this step
                    active_mask = ~env.done
                    self._log_step(env.current_step, active_mask, new_goals, env)
                    
                    if env.render_flag:
                        env.render()
                    
                    if env.verbose and env.current_step % 500 == 0:
                        active = np.sum(~env.done)
                        print(f"Trial {trial}, Gen {generation}, Step {env.current_step}, Active: {active}")
                
                # Log generation completion
                has_survivors = self._log_generation_complete(env)
                
                if not has_survivors:
                    trial_extinct = True
                    self.trial_extinct = True
                
                generation += 1
            
            # Log trial completion
            self._log_trial_complete(env)
        
        return env.current_step, None