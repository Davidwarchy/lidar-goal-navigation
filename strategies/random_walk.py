import random
import asyncio
from .base_strategy import BaseStrategy

class RandomWalkStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("random_walk", {})
    
    async def run(self, env):
        obs = env.reset()
        done = False
        
        while not done:
            action = random.randint(0, 3)
            obs, reward, done, info = env.step(action)
            env.render()
            
            print(f"Step {env.current_step}, Action: {info['action']}, New cells: {reward}, Coverage: {info['coverage']:.2f}%")
            
            # Uncomment to slow down
            # await asyncio.sleep(1.0 / env.fps)
        
        return env.current_step, env._get_coverage()