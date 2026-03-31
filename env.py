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

class RobotExplorationEnv:
    def __init__(self,
                 map_image_path,
                 grid_width=None, grid_height=None,
                 scale=2, fps=10,
                 robot_radius=5, num_rays=100, ray_length=200,
                 max_steps=int(1000e3),
                 wheel_base=4.0, wheel_radius=0.75, dt=0.2,
                 linear_speed=15.0, angular_speed=1.0,
                 output_dir=None, render=False,
                 strategy_name="unknown", strategy_parameters=None,
                 cache_size=1000,
                 enable_coverage=False,
                 use_lut=False,
                 verbose=False
                 ): 

        # ------------------------------------------------------------------
        # 1. Load the image **first**
        # ------------------------------------------------------------------
        self.map_image_path = map_image_path
        self.map_image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if self.map_image is None:
            raise ValueError(f"Could not load map image from {map_image_path}")

        # ------------------------------------------------------------------
        # 2. Derive grid size from the image (user can still override)
        # ------------------------------------------------------------------
        img_h, img_w = self.map_image.shape
        self.grid_width  = grid_width  if grid_width  is not None else img_w
        self.grid_height = grid_height if grid_height is not None else img_h

        # Resize if needed
        if self.map_image.shape[1] != self.grid_width or self.map_image.shape[0] != self.grid_height:
            self.map_image = cv2.resize(self.map_image,
                                        (self.grid_width, self.grid_height),
                                        interpolation=cv2.INTER_NEAREST)

        # ------------------------------------------------------------------
        # 3. Build the binary obstacle map (0 = free, 1 = obstacle)
        # ------------------------------------------------------------------
        _, self.obstacle_map = cv2.threshold(self.map_image,
                                             127, 1, cv2.THRESH_BINARY_INV)
        free_space_mask = (self.obstacle_map == 0).astype(np.uint8)
        self.distance_map = cv2.distanceTransform(free_space_mask, cv2.DIST_L2, 5)

        # ------------------------------------------------------------------
        # 4. Parameters
        # ------------------------------------------------------------------
        self.map_height, self.map_width = self.map_image.shape
        self.scale = scale
        self.window_width  = self.grid_width  * scale
        self.window_height = self.grid_height * scale
        self.fps = fps
        self.num_rays = num_rays
        self.ray_length = ray_length
        self.max_steps = max_steps
        self.render_flag = render
        self.cache_size = cache_size
        
        # Survival/Goal parameters
        self.energy = self.max_steps
        self.health = 200
        self.goal_x = None
        self.goal_y = None
        self.goal_success_dist = 1.0 
        self.goal_spawn_dist = 30.0 

        # Enable coverage if render enabled (because coverage updates are expensive and only needed for visualization/logging)
        self.enable_coverage = self.render_flag
        
        # Coarse grid for coverage (divide indices by 2)
        self.cov_scale = 2
        self.cov_width = self.grid_width // self.cov_scale
        self.cov_height = self.grid_height // self.cov_scale
        
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.dt = dt
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.robot_radius = robot_radius
        
        # State variables
        self.robot_x = None
        self.robot_y = None
        self.robot_orientation = None
        self.current_step = 0
        self.lidar_angles = np.linspace(-45, 45, self.num_rays)
        
        # Exploration grid (-1=unexplored, 0=free, 1=obstacle)
        self.exploration_grid = None
        
        # Strategy information
        self.strategy_name = strategy_name
        self.strategy_parameters = strategy_parameters or {}
        
        # Episode Tracking
        self.episode = 0

        # Output
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.output_dir = output_dir or os.path.join("output", f"{timestamp}_{strategy_name}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_buffer = []
        
        # Rendering attributes (only used if render_flag is True)
        self.screen = None
        self._pygame_initialized = False
        self.clock = None
        self.pygame = None
        
        # LIDAR cache
        self.lidar_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

        self._steps = np.arange(0, self.ray_length)
        self._base_angles = np.linspace(-45, 45, self.num_rays)

        self.verbose = verbose

        if self.strategy_name in ["ga", "spike_nn"]: 
            self.verbose = False

        # lut variables
        self.use_lut = use_lut
        self.lidar_lut = None

        if self.use_lut:
            self._load_lut()
    
        # Save metadata immediately
        self._save_metadata()

    def _load_lut(self):
        """Automatically locate and load the matching LUT file."""
        base_name = os.path.splitext(os.path.basename(self.map_image_path))[0]
        # Assuming 360 angular resolution as generated by your script
        lut_path = os.path.join("environments", "luts", f"{base_name}_360.npy")
        
        if os.path.exists(lut_path):
            self.lidar_lut = np.load(lut_path)
            if self.verbose:
                print(f"[INFO] Loaded LUT from {lut_path}")
        else:
            print(f"[WARNING] LUT file {lut_path} not found. Falling back to ray marching.")
            self.use_lut = False

    def _get_cache_key(self, x, y, orientation):
        """
        Generate a cache key from robot position and orientation.
        Uses quantization to handle floating point precision and similar positions.
        """
        # Quantize to reduce cache size and handle similar positions
        qx = int(round(x))
        qy = int(round(y))
        qo = int(round(orientation / 5) * 5)
        return (qx, qy, qo)

    def _get_cached_lidar(self, robot_x, robot_y, orientation):
        """
        Get LIDAR readings from cache if available, otherwise compute them.
        """
        cache_key = self._get_cache_key(robot_x, robot_y, orientation)
        
        if cache_key in self.lidar_cache:
            self.cache_hits += 1
            self.lidar_cache.move_to_end(cache_key)
            # Unpack intersections and precomputed distances
            cached_inter, cached_dist = self.lidar_cache[cache_key]
            return list(cached_inter), cached_dist.copy()  # Return a copy to prevent modification
        
        self.cache_misses += 1
        # Compute LIDAR readings and distances
        intersections, distances = self._compute_lidar_rays(robot_x, robot_y, orientation)
        
        # Manage cache size (LRU-like behavior with simple timestamp)
        if len(self.lidar_cache) >= self.cache_size:
            self.lidar_cache.popitem(last=False)
            self.cache_evictions += 1
        
        # Store in cache with timestamp
        self.lidar_cache[cache_key] = (intersections, distances)
        return intersections, distances

    def get_cache_stats(self):
        """Return cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
            "evictions": self.cache_evictions,
            "current_size": len(self.lidar_cache),
            "max_size": self.cache_size
        }

    def _save_metadata(self):
        """Save comprehensive run metadata before starting"""
        metadata = {
            "run_datetime": datetime.now().isoformat(),
            "strategy_name": self.strategy_name,
            "strategy_parameters": self.strategy_parameters,
            "total_steps_scheduled": self.max_steps,
            "environment_parameters": {
                "grid_width": self.grid_width,
                "grid_height": self.grid_height,
                "robot_radius": self.robot_radius,
                "num_rays": self.num_rays,
                "ray_length": self.ray_length,
                "max_steps": self.max_steps,
                "map_image": os.path.basename(self.map_image_path),
                "cache_size": self.cache_size,
                "goal_location": {"x": self.goal_x, "y": self.goal_y}
            },
            "output_directory": self.output_dir
        }
        
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        if self.verbose: 
            print(f"[INFO] Metadata saved to {metadata_path}")

    def reset(self):
        # Start robot in a free area (find first free pixel from center)
        center_x, center_y = self.map_width // 2, self.map_height // 2
        self.robot_x, self.robot_y = self._find_free_position(center_x, center_y)
        self.robot_orientation = 0
        self.current_step = 0
        
        # Survival Reset
        self.energy = self.max_steps
        self.health = 200
        self.goal_x, self.goal_y = self._spawn_reward(self.robot_x, self.robot_y, self.goal_spawn_dist)
        self._save_metadata()

        # Increment episode count
        self.episode += 1

        # Coarse coverage grid initialization
        self.exploration_grid = np.full((self.cov_width, self.cov_height), -1, dtype=int) 
        # Initialize log buffer
        self.log_buffer = []
        
        # NEW: Clear cache on reset (optional - depends on your needs)
        self.lidar_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

        return self._get_observation()

    def _spawn_reward(self, start_x, start_y, distance):
        """Finds a free position approximately 'distance' units away for the survival object"""
        for _ in range(200):
            angle = random.uniform(0, 2 * np.pi)
            tx = start_x + distance * np.cos(angle)
            ty = start_y + distance * np.sin(angle)
            
            # Safely round to nearest pixel integer
            ix, iy = int(round(tx)), int(round(ty))
            
            # Ensure strictly within array bounds before checking obstacle map
            if (0 <= ix < self.map_width and 0 <= iy < self.map_height and 
                self.obstacle_map[iy, ix] == 0):
                return tx, ty
                
        # If we failed to find a valid spot after 200 tries (e.g., map is too small), 
        # recursively decrease the distance by 10.0 and try again until we find a free spot.
        if distance > 10.0:
            new_distance = distance - 10.0
            print(f"[WARNING] Could not spawn reward at distance {distance:.1f}. Decreasing to {new_distance:.1f}...")
            return self._spawn_reward(start_x, start_y, new_distance)
            
        return start_x, start_y

    def _check_goal_reached(self):
        """Check if robot is within the success distance of the reward"""
        dx = self.robot_x - self.goal_x
        dy = self.robot_y - self.goal_y
        return dx*dx + dy*dy <= self.goal_success_dist ** 2

    def _find_free_position(self, start_x, start_y, max_radius=100):
        for radius in range(0, max_radius, 5):
            for angle in np.linspace(0, 2*np.pi, 36):
                x = int(start_x + radius * np.cos(angle))
                y = int(start_y + radius * np.sin(angle))
                if self._is_position_free(x, y):
                    return x, y
        # Fallback to start position if no free position found
        print("[WARNING] No free position found within radius, using start coordinates")
        return start_x, start_y

    def _is_position_free(self, x, y):
        """Check if position is free of obstacles considering robot radius"""

        ix, iy = int(x), int(y)
        if not (0 <= ix < self.map_width and 0 <= iy < self.map_height):
            return False
        return self.distance_map[iy, ix] >= self.robot_radius

    def step(self, action, extra_info=None):
        extra_info = extra_info or {}

        # Map action to left/right wheel velocities
        if action == 0:  # up
            v_left = v_right = self.linear_speed
        elif action == 1:  # down
            v_left = v_right = -self.linear_speed
        elif action == 2:  # left
            v_left = +self.linear_speed / 4 # negative because of image coordinates
            v_right = -self.linear_speed / 4
        else:  # right
            v_left = -self.linear_speed / 4 # negative because of image coordinates 
            v_right = +self.linear_speed / 4 

        # Update robot position
        self.robot_x, self.robot_y, self.robot_orientation = self._update_robot_position(
            self.robot_x, self.robot_y, self.robot_orientation, v_left, v_right
        )

        # Cast LIDAR using cached version
        intersections, distances = self.cast_lidar_rays_optimized(
            self.robot_x, self.robot_y, self.robot_orientation
        )

        # IDEA 1A: Throttle expensive map updates or disable entirely via flag
        if self.enable_coverage and self.current_step % 5 == 0:
            new_cells = self._update_map(intersections=intersections)
        else:
            new_cells = 0

        # Observation
        obs = self._get_observation(intersections, distances)
        
        # Reward logic: Check if goal is reached
        goal_reached = self._check_goal_reached()
        reward = 1.0 if goal_reached else 0.0

        # Step bookkeeping
        self.current_step += 1
        self.energy -= 1
        
        done = (self.energy <= 0 or self.health <= 0 or 
                self.current_step >= self.max_steps or goal_reached)
        
        info = {
            "new_cells": new_cells, 
            "coverage": self._get_coverage(), 
            "health": self.health,
            "energy": self.energy,
            "goal_reached": goal_reached,
            "action": action
        }

        return obs, reward, done, info

    def render(self):
        if not self.render_flag:
            return
        
        # Lazy import and initialization of pygame
        if not self._pygame_initialized:
            import pygame
            self.pygame = pygame
            self._pygame_initialized = True
            self.clock = pygame.time.Clock()
        
        if self.screen is None:
            self.pygame.init()
            self.screen = self.pygame.display.set_mode((self.window_width, self.window_height))
            self.pygame.display.set_caption("Robot Exploration - Image Map")
        
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.close()
        
        self.screen.fill((255, 255, 255))
        self._draw_map(self.pygame)
        self._draw_reward(self.pygame)
        self._draw_robot(self.pygame)
        self._draw_lidar(self.pygame)
        self.pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        # Log cache statistics at the end
        if hasattr(self, 'cache_hits'):
            stats = self.get_cache_stats()
            print(f"[INFO] Cache statistics: {stats}")
            
            # Save cache stats to file
            stats_path = os.path.join(self.output_dir, "cache_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
        
        if self._pygame_initialized and self.screen:
            self.pygame.quit()
            self.screen = None

    def _update_robot_position(self, x, y, orientation, v_left, v_right):
        linear_velocity = (v_left + v_right) / 2 * self.wheel_radius
        angular_velocity = (v_right - v_left) / self.wheel_base * self.wheel_radius
        angular_velocity = np.degrees(angular_velocity)
        
        new_x = x + linear_velocity * cos(radians(orientation)) * self.dt
        new_y = y + linear_velocity * sin(radians(orientation)) * self.dt
        new_orientation = (orientation + angular_velocity * self.dt) % 360
        
        if not self._is_position_free(new_x, new_y):
            self.health -= 1
            return x, y, orientation
        return new_x, new_y, new_orientation

    def _draw_lidar(self, pygame):
        intersections, _ = self.cast_lidar_rays_optimized(self.robot_x, self.robot_y, self.robot_orientation)
        angles = self.robot_orientation + self.lidar_angles
        
        for i, (inter, angle_deg) in enumerate(zip(intersections, angles)):
            angle_rad = np.radians(angle_deg)
            
            if inter:
                # Draw to obstacle
                end_x, end_y = inter[0], inter[1]
                color = (255, 255, 0)  # Yellow
            else:
                # Draw to max range
                end_x = self.robot_x + self.ray_length * np.cos(angle_rad)
                end_y = self.robot_y + self.ray_length * np.sin(angle_rad)
                color = (128, 128, 128)  # Gray
            
            pygame.draw.line(
                self.screen, color,
                (int(self.robot_x * self.scale), int(self.robot_y * self.scale)),
                (int(end_x * self.scale), int(end_y * self.scale)), 1
            )

    def cast_lidar_rays_optimized(self, robot_x, robot_y, orientation):
        """
        Public method that selects the fastest available method. [cite: 54]
        """
        # If LUT is available and we don't need intersections (no rendering/coverage), 
        # use the ultra-fast O(1) path. 
        if self.use_lut and self.lidar_lut is not None and not self.render_flag and not self.enable_coverage:
            return self._get_lut_distances_only(robot_x, robot_y, orientation)
        
        # Otherwise, fall back to standard cached ray marching for full data. [cite: 55]
        return self._get_cached_lidar(robot_x, robot_y, orientation)

    def _get_lut_distances_only(self, x, y, orientation):
        """Ultra-fast path: Returns only the distance array. """
        ix, iy = int(round(x)), int(round(y))
        
        # Boundary check [cite: 59]
        if not (0 <= ix < self.map_width and 0 <= iy < self.map_height):
            return None, np.full(self.num_rays, self.ray_length, dtype=np.float32)

        # Vectorized LUT indexing [cite: 177, 186]
        # (orientation + lidar_angles) broadcasts the robot's heading across the sensor fan. [cite: 176]
        angle_indices = ((orientation + self.lidar_angles) % 360).astype(np.int32)
        
        # Direct retrieval from the precomputed [H, W, Angle] table. [cite: 167, 178]
        distances = self.lidar_lut[iy, ix, angle_indices]
        
        return None, distances

    def _compute_lidar_rays(self, robot_x, robot_y, orientation, num_rays=None, max_range=None):
        """
        Fully vectorized LIDAR ray marching.
        No Bresenham.
        No Python loops over pixels.
        Fast.
        """
        if num_rays is None:
            num_rays = self.num_rays
        if max_range is None:
            max_range = self.ray_length

        # Compute ray angles
        angles_rad = np.radians(orientation + self._base_angles)

        # Unit direction vectors for all rays
        dx = np.cos(angles_rad)
        dy = np.sin(angles_rad)

            # Expand for broadcasting: shape (num_rays, max_range)
        ray_x = robot_x + np.outer(dx, self._steps)
        
        ray_y = robot_y + np.outer(dy, self._steps)

        # Truncate to integer grid coordinates
        ray_x = ray_x.astype(np.int32)
        ray_y = ray_y.astype(np.int32)

        # Validity mask to ignore out-of-bounds
        valid_mask = (
            (ray_x >= 0) &
            (ray_x < self.map_width) &
            (ray_y >= 0) &
            (ray_y < self.map_height)
        )

        # Clip coordinates for safe indexing (prevents index errors)
        ray_x_clip = np.clip(ray_x, 0, self.map_width - 1)
        ray_y_clip = np.clip(ray_y, 0, self.map_height - 1)

        # Fetch obstacle hits in one batched operation
        hits_raw = self.obstacle_map[ray_y_clip, ray_x_clip]
        hits = (hits_raw == 1) & valid_mask

        # Find index of first hit per ray (argmax on bool returns index of first True)
        hit_indices = np.argmax(hits, axis=1)

        # Check if there was actually a hit (avoids false positives where argmax=0 but no hit)
        has_hit = np.any(hits, axis=1)

        # Build intersections list
        intersections = [None] * num_rays
        
        # IDEA 2C: Precompute exact distances using the hit indices directly
        distances = np.full(num_rays, max_range, dtype=np.float32)

        for i in range(num_rays):
            if has_hit[i]:
                idx = hit_indices[i]
                intersections[i] = (ray_x[i, idx], ray_y[i, idx])
                distances[i] = idx # step index == pixel distance 

        return intersections, distances

    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm for efficient pixel traversal"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        
        points.append((x, y))
        return points

    def _get_observation(self, intersections=None, distances=None):
        """Get LIDAR distances as observation"""
        # IDEA 2C: Instantly return the precomputed distance array without calculating 
        if distances is not None:
            return distances

        if intersections is None:
            intersections, distances = self.cast_lidar_rays_optimized(
                self.robot_x, self.robot_y, self.robot_orientation
            )
            return distances

        # IDEA 2B: Fallback to Vectorized NumPy if only intersections are supplied
        xs, ys, mask = [], [], []
        for inter in intersections:
            if inter is None:
                xs.append(0.0)
                ys.append(0.0)
                mask.append(False)
            else:
                xs.append(inter[0])
                ys.append(inter[1])
                mask.append(True)
                
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        mask = np.array(mask, dtype=bool)

        dx = xs - self.robot_x
        dy = ys - self.robot_y
        d = np.empty_like(dx)
        
        d[mask] = np.sqrt(dx[mask] * dx[mask] + dy[mask] * dy[mask])
        d[~mask] = self.ray_length
        return d

    def _get_coverage(self):
        """Calculate percentage of explored cells (free + obstacle). 0 - free, 1 - obstacle, -1 - unexplored"""
        # Coverage derived from coarse grid (Idea 1D)
        if not self.enable_coverage:
            return 0.0
        return 100 * np.sum(self.exploration_grid >= 0) / (self.cov_width * self.cov_height)

    def _update_map(self, intersections):
        """
        Update exploration grid using precomputed LIDAR intersections.
        Only called if rendering/logging is needed.
        """
        new_cells = 0
        
        # IDEA 1C: Use fewer rays for coverage (skip by step=4)
        ray_step = 4
        angles = np.linspace(self.robot_orientation - 45, self.robot_orientation + 45, self.num_rays)

        for i in range(0, len(intersections), ray_step):
            inter = intersections[i]
            angle_rad = np.radians(angles[i])

            if inter is not None:
                ox, oy = int(inter[0]), int(inter[1])
                line_points = self._bresenham_line(int(self.robot_x), int(self.robot_y), ox, oy)
                
                for px, py in line_points[:-1]:
                    # IDEA 1D: Coarse grid division
                    cx, cy = px // self.cov_scale, py // self.cov_scale
                    if 0 <= cx < self.cov_width and 0 <= cy < self.cov_height:
                        if self.exploration_grid[cx, cy] == -1:
                            self.exploration_grid[cx, cy] = 0
                            new_cells += 1

                cx, cy = ox // self.cov_scale, oy // self.cov_scale
                if 0 <= cx < self.cov_width and 0 <= cy < self.cov_height:
                    if self.exploration_grid[cx, cy] == -1:
                        self.exploration_grid[cx, cy] = 1
                        new_cells += 1
            else:
                end_x = int(self.robot_x + self.ray_length * np.cos(angle_rad))
                end_y = int(self.robot_y + self.ray_length * np.sin(angle_rad))
                line_points = self._bresenham_line(int(self.robot_x), int(self.robot_y), end_x, end_y)

                for px, py in line_points:
                    cx, cy = px // self.cov_scale, py // self.cov_scale
                    if 0 <= cx < self.cov_width and 0 <= cy < self.cov_height:
                        if self.exploration_grid[cx, cy] == -1:
                            self.exploration_grid[cx, cy] = 0
                            new_cells += 1

        return new_cells

    def _draw_map(self, pygame):
        """Simple drawing that shows exploration progress"""
        # Draw base map
        base_surface = pygame.surfarray.make_surface(np.transpose(
            np.stack([self.map_image] * 3, axis=-1), (1, 0, 2)
        ))
        if self.scale != 1:
            base_surface = pygame.transform.scale(base_surface, (self.window_width, self.window_height))
        self.screen.blit(base_surface, (0, 0))
        
        # Draw exploration overlay
        if not self.enable_coverage:
            return
            
        # Adjust block drawing size to compensate for coarse grid (Idea 1D)
        block_size = self.cov_scale * self.scale
        
        for cx in range(self.cov_width):
            for cy in range(self.cov_height):
                if self.exploration_grid[cx, cy] == 0:  
                    color = (0, 255, 0)
                    pygame.draw.rect(self.screen, color, 
                                (cx * block_size, cy * block_size, block_size, block_size))
                elif self.exploration_grid[cx, cy] == 1: 
                    color = (255, 0, 0)
                    pygame.draw.rect(self.screen, color,
                                (cx * block_size, cy * block_size, block_size, block_size))
                    
    def _draw_robot(self, pygame):
        """Draw robot as circle with orientation"""
        # Scale coordinates for display
        display_x = int(self.robot_x * self.scale)
        display_y = int(self.robot_y * self.scale)
        display_radius = int(self.robot_radius * self.scale)
        
        # Draw robot body
        pygame.draw.circle(self.screen, (0, 0, 255), (display_x, display_y), display_radius)
        
        # Draw orientation line
        end_x = display_x + display_radius * 1.5 * np.cos(np.radians(self.robot_orientation))
        end_y = display_y + display_radius * 1.5 * np.sin(np.radians(self.robot_orientation))
        pygame.draw.line(self.screen, (255, 255, 0), (display_x, display_y), (int(end_x), int(end_y)), 3)
        
        # Draw robot center
        pygame.draw.circle(self.screen, (255, 255, 0), (display_x, display_y), 3)

    def _draw_reward(self, pygame):
        """Draw the survival object"""
        if self.goal_x is not None:
            gx, gy = int(self.goal_x * self.scale), int(self.goal_y * self.scale)
            pygame.draw.circle(self.screen, (0, 0, 255), (gx, gy), int(self.goal_success_dist * self.scale))