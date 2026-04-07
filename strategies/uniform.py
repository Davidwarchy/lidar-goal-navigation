import numpy as np
from .base_strategy import BaseStrategy

class UniformRunLengthStrategy(BaseStrategy):
    def __init__(self, min_step=1, max_step=200):
        params = {"min_step": int(min_step), "max_step": int(max_step)}
        super().__init__("uniform", params)
        self.min_step = int(min_step)
        self.max_step = int(max_step)

    def run(self, env):
        obs = env.reset()
        
        # Per-agent state
        steps_left = np.random.randint(self.min_step, self.max_step + 1, size=env.num_envs)
        direction = np.random.randint(0, 4, size=env.num_envs)
        
        while not np.all(env.done):
            actions = direction.copy()
            obs, rewards, dones, info = env.step(actions)
            
            steps_left -= 1
            need_new = steps_left <= 0
            
            if np.any(need_new):
                direction[need_new] = np.random.randint(0, 4, size=np.sum(need_new))
                steps_left[need_new] = np.random.randint(self.min_step, self.max_step + 1, 
                                                         size=np.sum(need_new))
            
            if env.render_flag:
                env.render()
            
            if env.verbose and env.current_step % 500 == 0:
                active = np.sum(~env.done)
                print(f"Step {env.current_step}, Active: {active}")
        
        return env.current_step, None