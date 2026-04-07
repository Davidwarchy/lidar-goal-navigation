import pygame
import numpy as np
from .base_strategy import BaseStrategy

class ManualControlStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("manual_control", {})
    
    def run(self, env):
        # Force single environment for manual control
        if env.num_envs != 1:
            print("Warning: Manual control only works with 1 environment. Forcing num_envs=1")
            # This won't actually change env.num_envs, but we'll only use index 0
        
        obs = env.reset()
        done = False
        print("Manual control: arrow keys to move. ESC or Q to quit.")
        print("  ↑ = forward, ↓ = backward, ← = turn left, → = turn right")

        while not done:
            env.render()

            # Handle quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
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

            # Only step when a key is pressed (vectorized with single action)
            if action is not None:
                actions = np.array([action])  # Vectorized: array of 1 action
                obs, reward, dones, info = env.step(actions)
                done = dones[0]
                
                if env.verbose:
                    dist_to_goal = np.sqrt((env.robot_x[0] - env.goal_x[0])**2 + 
                                           (env.robot_y[0] - env.goal_y[0])**2)
                    print(f"Step {env.current_step}, Action: {action}, "
                          f"Dist to goal: {dist_to_goal:.1f}, "
                          f"Energy: {env.energy[0]}, Health: {env.health[0]}")

            env.clock.tick(env.fps)

        return env.current_step, None