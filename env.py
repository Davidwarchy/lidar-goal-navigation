import sys

def _dep_error_message(missing_import: str) -> str:
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    base = [
        f"Missing dependency: {missing_import!r}.",
        "",
        "Install dependencies with:",
        "  python -m pip install -r requirements.txt",
        "",
        f"Detected Python: {pyver}",
    ]
    return "\n".join(base)

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(_dep_error_message("numpy")) from e

import os
import csv
import random

try:
    import cv2
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(_dep_error_message("opencv-python (import name: cv2)")) from e

from datetime import datetime
from math import cos, sin, radians, sqrt
import json
from collections import OrderedDict
import warnings

class VectorRobotExplorationEnv:
    def __init__(self,
                 map_image_path,
                 num_envs=1,
                 grid_width=None, grid_height=None,
                 scale=2, fps=500,
                 robot_radius=5, num_rays=100, ray_length=200,
                 max_steps=1000,
                 wheel_base=4.0, wheel_radius=0.75, dt=0.02,
                 linear_speed=15.0, angular_speed=1.0,
                 output_dir=None, render=False,
                 strategy_name="unknown", strategy_parameters=None,
                 use_lut=True,
                 continue_after_goal=False,
                 verbose=False):

        self.num_envs = num_envs
        self.map_image_path = map_image_path
        self.map_image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if self.map_image is None:
            raise ValueError(f"Could not load map image from {map_image_path}")

        img_h, img_w = self.map_image.shape
        self.grid_width = grid_width if grid_width is not None else img_w
        self.grid_height = grid_height if grid_height is not None else img_h

        if self.map_image.shape[1] != self.grid_width or self.map_image.shape[0] != self.grid_height:
            self.map_image = cv2.resize(self.map_image, (self.grid_width, self.grid_height), interpolation=cv2.INTER_NEAREST)

        _, self.obstacle_map = cv2.threshold(self.map_image, 127, 1, cv2.THRESH_BINARY_INV)
        free_space_mask = (self.obstacle_map == 0).astype(np.uint8)
        # The distanceTransform calculates the distance from every free pixel to the nearest obstacle.
        # Purpose: It tells us if the entire volume of the robot (defined by robot_radius) can fit at a specific $(x, y)$ coordinate without overlapping a wall
        # Efficiency: Without this, you would have to check every single pixel under the robot's circular footprint every time it moves. With distanceTransform, we only check one pixel: if distance_map[y, x] < robot_radius, we know a collision has occurred
        self.distance_map = cv2.distanceTransform(free_space_mask, cv2.DIST_L2, 5) 

        self.map_height, self.map_width = self.map_image.shape
        self.scale = scale
        self.window_width = self.grid_width * scale
        self.window_height = self.grid_height * scale
        self.fps = fps
        self.num_rays = num_rays
        self.ray_length = ray_length
        self.max_steps = max_steps
        self.render_flag = render
        
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.dt = dt
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.robot_radius = robot_radius

        # Vectorized State variables
        self.robot_x = np.zeros(num_envs, dtype=np.float32)
        self.robot_y = np.zeros(num_envs, dtype=np.float32)
        self.robot_orientation = np.zeros(num_envs, dtype=np.float32)
        self.energy = np.full(num_envs, max_steps, dtype=np.int32)
        self.health = np.full(num_envs, 500, dtype=np.int32)
        self.done = np.zeros(num_envs, dtype=bool)
        self.current_step = 0
        
        # Survival/Goal parameters
        self.goal_x = np.zeros(num_envs, dtype=np.float32)
        self.goal_y = np.zeros(num_envs, dtype=np.float32)
        self.goal_success_dist = 6.0 
        self.goal_spawn_dist = 30.0 

        self.lidar_angles = np.linspace(-45, 45, self.num_rays)
        self.strategy_name = strategy_name
        self.strategy_parameters = strategy_parameters or {}

        # Output Setup
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.output_dir = output_dir or os.path.join("output", f"{timestamp}_{strategy_name}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.use_lut = use_lut
        self.lidar_lut = None
        if self.use_lut:
            self._load_lut()

        self.continue_after_goal = continue_after_goal
        self.verbose = verbose
        
        # Pygame attributes
        self.screen = None
        self._pygame_initialized = False
        self._save_metadata()

    def _load_lut(self):
        base_name = os.path.splitext(os.path.basename(self.map_image_path))[0]
        lut_path = os.path.join("environments", "luts", f"{base_name}_360.npy")
        if os.path.exists(lut_path):
            self.lidar_lut = np.load(lut_path)
        else:
            self.use_lut = False

    def _save_metadata(self):
        """Save comprehensive run metadata for the vectorized session."""
        metadata = {
            "run_datetime": datetime.now().isoformat(),
            "strategy_name": self.strategy_name,
            "num_envs": self.num_envs,
            "max_steps": self.max_steps,
            "continue_after_goal": self.continue_after_goal,
            "environment_parameters": {
                "grid_width": self.grid_width,
                "grid_height": self.grid_height,
                "robot_radius": self.robot_radius,
                "num_rays": self.num_rays,
                "ray_length": self.ray_length,
                "map_image": os.path.basename(self.map_image_path),
                "use_lut": self.use_lut
            },
            "output_directory": self.output_dir
        }
        
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def reset(self):
        center_x, center_y = self.map_width // 2, self.map_height // 2
        for i in range(self.num_envs):
            self.robot_x[i], self.robot_y[i] = self._find_free_position(center_x, center_y)
            gx, gy = self._spawn_reward(self.robot_x[i], self.robot_y[i], self.goal_spawn_dist)
            self.goal_x[i], self.goal_y[i] = gx, gy
            
        self.robot_orientation.fill(0)
        self.current_step = 0
        self.energy.fill(self.max_steps)
        self.health.fill(500)
        self.done.fill(False)

        return self._get_observation()

    def _find_free_position(self, start_x, start_y, max_radius=100):
        for radius in range(0, max_radius, 5):
            for angle in np.linspace(0, 2*np.pi, 36):
                x, y = int(start_x + radius * np.cos(angle)), int(start_y + radius * np.sin(angle))
                if self._is_position_free(x, y): return x, y
        return start_x, start_y

    def _is_position_free(self, x, y):
        ix, iy = int(x), int(y)
        if not (0 <= ix < self.map_width and 0 <= iy < self.map_height): return False
        return self.distance_map[iy, ix] >= self.robot_radius

    def _spawn_reward(self, start_x, start_y, distance):
        for _ in range(200):
            angle = np.random.uniform(0, 2 * np.pi)
            tx, ty = start_x + distance * np.cos(angle), start_y + distance * np.sin(angle)
            ix, iy = int(round(tx)), int(round(ty))
            if (0 <= ix < self.map_width and 0 <= iy < self.map_height and self.obstacle_map[iy, ix] == 0):
                return tx, ty
        return start_x, start_y

    def step(self, actions):
        # Only decrement health for robots that are still active (not done)
        active_mask = ~self.done
        
        v_left = np.zeros(self.num_envs)
        v_right = np.zeros(self.num_envs)

        # Vectorized action mapping
        v_left[actions == 0] = v_right[actions == 0] = self.linear_speed # Up
        v_left[actions == 1] = v_right[actions == 1] = -self.linear_speed # Down
        v_left[actions == 2], v_right[actions == 2] = self.linear_speed/4, -self.linear_speed/4 # Left
        v_left[actions == 3], v_right[actions == 3] = -self.linear_speed/4, self.linear_speed/4 # Right

        # Update positions
        self._update_robot_positions(v_left, v_right)
        # 2. Get Perception
        obs = self._get_observation()

        # 3. Check for Success (Goal reached)
        dx = self.robot_x - self.goal_x
        dy = self.robot_y - self.goal_y
        goal_reached = (dx*dx + dy*dy) <= self.goal_success_dist ** 2
        
        # 4. Success handling
        # Reward is only for the step they actually find it
        rewards = np.where(goal_reached & active_mask, 1.0, 0.0)

        # 5. Resource Depletion
        # In 'Life on Silicon', every step costs 1 energy point 
        # We only decrement if they haven't finished yet
        self.energy[active_mask] -= 1
        
        # 6. Final Termination Check
        # If NOT continuing after goal, reaching it marks them as done immediately
        if not self.continue_after_goal:
            # Done if: No energy OR No health OR Reached Max Steps OR Reached Goal
            new_dones = (self.energy <= 0) | (self.health <= 0) | \
                        (self.current_step >= self.max_steps) | goal_reached
        else:
            # If continuing, they only stop when they physically 'die' or time runs out
            new_dones = (self.current_step >= self.max_steps)

        self.current_step += 1
        self.done |= new_dones

        return obs, rewards, self.done.copy(), {"goal_reached": goal_reached}

    def _update_robot_positions(self, v_left, v_right):
        lin_vel = (v_left + v_right) / 2 * self.wheel_radius
        ang_vel = np.degrees((v_right - v_left) / self.wheel_base * self.wheel_radius)
        
        new_x = self.robot_x + lin_vel * np.cos(np.radians(self.robot_orientation)) * self.dt
        new_y = self.robot_y + lin_vel * np.sin(np.radians(self.robot_orientation)) * self.dt
        new_orient = (self.robot_orientation + ang_vel * self.dt) % 360
        
        # Collision check for all robots
        ixs, iys = new_x.astype(int), new_y.astype(int)
        valid = (ixs >= 0) & (ixs < self.map_width) & (iys >= 0) & (iys < self.map_height)
        
        # Safe distance transform lookup
        dist_vals = np.zeros(self.num_envs)
        dist_vals[valid] = self.distance_map[iys[valid], ixs[valid]]
        
        free_mask = dist_vals >= self.robot_radius
        
        # Update only if free and not already done
        update_mask = free_mask & ~self.done
        self.robot_x[update_mask] = new_x[update_mask]
        self.robot_y[update_mask] = new_y[update_mask]
        self.robot_orientation[update_mask] = new_orient[update_mask]
        
        # Penalty for collision
        self.health[~free_mask & ~self.done] -= 1

    def _get_observation(self):
        """
        Get LIDAR distances for the entire population. 
        Prioritizes the O(1) LUT but falls back to vectorized ray marching.
        """
        if self.use_lut:
            if self.lidar_lut is None:
                raise ValueError("use_lut is True but lidar_lut was not loaded. Check your paths.")
            
            # 1. Prepare indices (H, W)
            ixs = np.round(self.robot_x).astype(int)
            iys = np.round(self.robot_y).astype(int)
            
            # Boundary clipping for safety
            ixs = np.clip(ixs, 0, self.map_width - 1)
            iys = np.clip(iys, 0, self.map_height - 1)
            
            # 2. Map world angles to LUT indices
            # Shape: (num_envs, num_rays)
            angles = (self.robot_orientation[:, None] + self.lidar_angles) % 360
            angle_idxs = angles.astype(int)
            
            # 3. Vectorized LUT retrieval
            # We use advanced indexing to pull distances for all robots at once
            return self.lidar_lut[iys, ixs, :][np.arange(self.num_envs)[:, None], angle_idxs]

        # FALLBACK: Vectorized Ray Marching (if LUT is not used/available)
        # Raise warning if LUT was intended but not loaded
        if self.use_lut:
            
            warnings.warn("use_lut is True but lidar_lut was not loaded. Falling back to vectorized ray marching.")

        return self._compute_vectorized_ray_marching()

    def _compute_vectorized_ray_marching(self):
        """
        Computes LIDAR for all robots simultaneously without a LUT.
        Complexity: O(num_envs * num_rays * ray_length)
        """
        # Create ray steps: shape (ray_length,)
        steps = np.arange(self.ray_length)
        
        # Absolute angles for every ray of every robot: (num_envs, num_rays)
        angles_rad = np.radians(self.robot_orientation[:, None] + self.lidar_angles)
        
        # Unit vectors: (num_envs, num_rays, 1)
        dx = np.cos(angles_rad)[:, :, None]
        dy = np.sin(angles_rad)[:, :, None]
        
        # Compute all potential ray coordinates: (num_envs, num_rays, ray_length)
        # x_points = robot_x + dx * steps
        ray_x = (self.robot_x[:, None, None] + dx * steps).astype(np.int32)
        ray_y = (self.robot_y[:, None, None] + dy * steps).astype(np.int32)
        
        # Clip to map boundaries to avoid index errors
        ray_x_clp = np.clip(ray_x, 0, self.map_width - 1)
        ray_y_clp = np.clip(ray_y, 0, self.map_height - 1)
        
        # Check hits against the obstacle map
        # Resulting 'hits' shape: (num_envs, num_rays, ray_length)
        hits = self.obstacle_map[ray_y_clp, ray_x_clp] == 1
        
        # Find the first hit along the 'ray_length' axis
        # argmax returns the first index of 'True'
        hit_indices = np.argmax(hits, axis=2)
        
        # Identify rays that never hit anything
        no_hits = ~np.any(hits, axis=2)
        
        # Final distances: use hit index or default to max ray_length
        distances = hit_indices.astype(np.float32)
        distances[no_hits] = float(self.ray_length)
        
        return distances

    def render(self):
        if not self.render_flag: return
        if not self._pygame_initialized:
            import pygame
            self.pygame = pygame
            self.pygame.init()
            self.screen = self.pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = self.pygame.time.Clock()
            self._pygame_initialized = True
        
        self.screen.fill((255, 255, 255))
        # Draw base map
        map_surf = self.pygame.surfarray.make_surface(np.transpose(np.stack([self.map_image]*3, axis=-1), (1,0,2)))
        self.screen.blit(self.pygame.transform.scale(map_surf, (self.window_width, self.window_height)), (0,0))
        
        # Draw goals for all active robots
        for i in range(self.num_envs):
            if not self.done[i]:
                goal_x = int(self.goal_x[i] * self.scale)
                goal_y = int(self.goal_y[i] * self.scale)
                radius = int(self.goal_success_dist * self.scale)
                
                # Outer ring (bright green)
                self.pygame.draw.circle(self.screen, (0, 255, 0), (goal_x, goal_y), radius, 2)
                # Inner circle (lighter green)
                self.pygame.draw.circle(self.screen, (100, 255, 100), (goal_x, goal_y), max(2, radius // 2))
                # Center bullseye
                self.pygame.draw.circle(self.screen, (255, 255, 255), (goal_x, goal_y), max(2, radius // 4))
        
        # Draw active robots
        for i in range(self.num_envs):
            if not self.done[i]:
                center_x = int(self.robot_x[i] * self.scale)
                center_y = int(self.robot_y[i] * self.scale)
                radius = int(self.robot_radius * self.scale)
                
                # Robot body
                self.pygame.draw.circle(self.screen, (0, 0, 255), (center_x, center_y), radius)
                
                # Heading direction (red line)
                arrow_len = radius * 1.5
                angle_rad = np.radians(self.robot_orientation[i])
                end_x = int(center_x + arrow_len * np.cos(angle_rad))
                end_y = int(center_y + arrow_len * np.sin(angle_rad))
                self.pygame.draw.line(self.screen, (255, 0, 0), (center_x, center_y), (end_x, end_y), 2)
                
                # Tip dot
                self.pygame.draw.circle(self.screen, (255, 255, 0), (end_x, end_y), 2)
        
        self.pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self._pygame_initialized: self.pygame.quit()