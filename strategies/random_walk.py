import random
from .base_strategy import BaseStrategy

class RandomWalkStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("random_walk", {})
    
    def run(self, env):
        obs = env.reset()
        done = False
        
        while not done:
            action = random.randint(0, 3)
            obs, reward, done, info = env.step(action)
            env.render()
            
            if env.current_step % 100 == 0:
                print(f"Step {env.current_step}, Action: {info['action']}")
            
        return env.current_step, env._get_coverage()