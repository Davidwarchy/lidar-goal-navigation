import numpy as np
from .base_strategy import BaseStrategy

class RandomWalkStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("random_walk", {})
    
    def run(self, env):
        obs = env.reset()
        
        while not np.all(env.done):
            actions = np.random.randint(0, 4, size=env.num_envs)
            obs, rewards, dones, info = env.step(actions)
            
            if env.render_flag:
                env.render()
            
            if env.verbose and env.current_step % 20 == 0:
                active = np.sum(~env.done)
                print(f"Step {env.current_step}, Active agents: {active}")
        
        return env.current_step, None