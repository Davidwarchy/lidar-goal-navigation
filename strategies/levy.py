import numpy as np
from .base_strategy import BaseStrategy

class LevyWalkStrategy(BaseStrategy):
    def __init__(self, alpha=1.6, min_step=1.0, max_step=200.0):
        parameters = {
            "alpha": alpha,
            "min_step": min_step,
            "max_step": max_step
        }
        super().__init__("levy_walk", parameters)
        
        self.alpha = alpha
        self.min_step = min_step
        self.max_step = max_step

    def _sample_pareto(self, n):
        """Vectorized Pareto sampling with truncation."""
        u = np.random.uniform(size=n)
        x = self.min_step * (1 - u) ** (-1.0 / self.alpha)
        # Clip to max_step (simpler than rejection)
        x = np.clip(x, self.min_step, self.max_step)
        return np.maximum(1, np.round(x)).astype(int)
    
    def run(self, env):
        obs = env.reset()
        
        # Per-agent state
        steps_left = self._sample_pareto(env.num_envs)
        direction = np.random.randint(0, 4, size=env.num_envs)
        
        while not np.all(env.done):
            # Step with current directions
            actions = direction.copy()
            obs, rewards, dones, info = env.step(actions)
            
            # Decrement and check for new runs
            steps_left -= 1
            need_new = steps_left <= 0
            
            if np.any(need_new):
                direction[need_new] = np.random.randint(0, 4, size=np.sum(need_new))
                steps_left[need_new] = self._sample_pareto(np.sum(need_new))
            
            if env.render_flag:
                env.render()
            
            if env.verbose and env.current_step % 500 == 0:
                active = np.sum(~env.done)
                print(f"Step {env.current_step}, Active: {active}")
        
        return env.current_step, None