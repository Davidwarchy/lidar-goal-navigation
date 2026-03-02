import pygame
import asyncio
from .base_strategy import BaseStrategy

class ManualControlStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("manual_control", {})
    
    async def run(self, env):
        obs = env.reset()
        done = False
        print("Manual control: hold arrow keys to move. Press ESC to quit.")

        while not done:
            env.render()

            # Handle quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    done = True

            # Continuous key state check
            keys = pygame.key.get_pressed()
            action = None

            # Movement mapping
            if keys[pygame.K_UP]:
                action = 0  # forward
            elif keys[pygame.K_DOWN]:
                action = 1  # backward
            elif keys[pygame.K_LEFT]:
                action = 2  # turn left
            elif keys[pygame.K_RIGHT]:
                action = 3  # turn right

            # Only step when a key is pressed
            if action is not None:
                obs, reward, done, info = env.step(action)
                print(f"Step {env.current_step}, Action: {info['action']}, "
                      f"New cells: {reward}, Coverage: {info['coverage']:.2f}%")

            await asyncio.sleep(1.0 / env.fps)

        return env.current_step, env._get_coverage()