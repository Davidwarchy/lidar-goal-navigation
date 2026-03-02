import numpy as np
import os
import csv
import cv2
from datetime import datetime
from math import cos, sin, radians, sqrt
import json
import hashlib

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
                 cache_size=1000):  # New parameter for cache size

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

        # ------------------------------------------------------------------
        # 4. Parameters
        # ------------------------------------------------------------------
        self.map_height, self.map_width = self.map_image.shape   # now = grid size
        self.scale = scale
        self.window_width  = self.grid_width  * scale
        self.window_height = self.grid_height * scale
        self.fps = fps
        self.num_rays = num_rays
        self.ray_length = ray_length
        self.max_steps = max_steps
        self.render_flag = render
        self.cache_size = cache_size  # Maximum number of cached readings
        
        # Robot parameters
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
        
        # NEW: LIDAR cache
        self.lidar_cache = {}  # Dictionary: key -> (intersections, timestamp)
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        
        # Save metadata immediately
        self._save_metadata()

    def _get_cache_key(self, x, y, orientation):
        """
        Generate a cache key from robot position and orientation.
        Uses quantization to handle floating point precision and similar positions.
        """
        # Quantize to reduce cache size and handle similar positions
        quantize_factor = 1  # Adjust based on your needs (1 = no quantization)
        
        qx = round(x / quantize_factor) * quantize_factor
        qy = round(y / quantize_factor) * quantize_factor
        qo = round(orientation / 5) * 5  # Quantize orientation to nearest 5 degrees
        
        # Create a hash for faster dictionary lookups
        key_str = f"{qx:.1f}_{qy:.1f}_{qo:.1f}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_lidar(self, robot_x, robot_y, orientation):
        """
        Get LIDAR readings from cache if available, otherwise compute them.
        """
        cache_key = self._get_cache_key(robot_x, robot_y, orientation)
        
        if cache_key in self.lidar_cache:
            self.cache_hits += 1
            intersections, _ = self.lidar_cache[cache_key]
            return intersections.copy()  # Return a copy to prevent modification
        
        self.cache_misses += 1
        
        # Compute LIDAR readings
        intersections = self._compute_lidar_rays(robot_x, robot_y, orientation)
        
        # Manage cache size (LRU-like behavior with simple timestamp)
        if len(self.lidar_cache) >= self.cache_size:
            # Remove oldest entry (simple approach)
            oldest_key = min(self.lidar_cache.keys(), 
                           key=lambda k: self.lidar_cache[k][1])
            del self.lidar_cache[oldest_key]
            self.cache_evictions += 1
        
        # Store in cache with timestamp
        self.lidar_cache[cache_key] = (intersections, self.current_step)
        
        return intersections 

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

    # [All other methods remain exactly the same as before]
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
                "cache_size": self.cache_size  # Added cache info to metadata
            },
            "output_directory": self.output_dir
        }
        
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"[INFO] Metadata saved to {metadata_path}")

    def reset(self):
        # Start robot in a free area (find first free pixel from center)
        center_x, center_y = self.map_width // 2, self.map_height // 2
        self.robot_x, self.robot_y = self._find_free_position(center_x, center_y)
        self.robot_orientation = 0
        self.current_step = 0
        
        # Increment episode count
        self.episode += 1

        # Initialize exploration grid
        self.exploration_grid = np.full((self.grid_width, self.grid_height), -1, dtype=int) 
        
        # Initialize log buffer
        self.log_buffer = []
        
        # NEW: Clear cache on reset (optional - depends on your needs)
        self.lidar_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

        return self._get_observation()
    def _find_free_position(self, start_x, start_y, max_radius=100):
        """Find a free position for the robot starting from given coordinates"""
        for radius in range(0, max_radius, 5):
            for angle in np.linspace(0, 2*np.pi, 36):
                x = int(start_x + radius * np.cos(angle))
                y = int(start_y + radius * np.sin(angle))
                if (0 <= x < self.map_width and 0 <= y < self.map_height and 
                    self._is_position_free(x, y)):
                    return x, y
        # Fallback to start position if no free position found
        print("[WARNING] No free position found within radius, using start coordinates")
        return start_x, start_y

    def _is_position_free(self, x, y):
        """Check if position is free of obstacles considering robot radius"""
        # Check robot center and surrounding area
        check_radius = self.robot_radius
        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                if dx*dx + dy*dy <= check_radius*check_radius: # pythagorean check for circular area
                    check_x, check_y = int(x + dx), int(y + dy)
                    if (0 <= check_x < self.map_width and 0 <= check_y < self.map_height):
                        if self.obstacle_map[check_y, check_x] == 1:
                            return False
        return True

    def _calculate_reward(self):
        """
        Base reward function. Override in subclasses (e.g. BlobEnv) for RL.
        """
        return 0.0

    def step(self, action, extra_info=None):
        if not 0 <= action <= 3:
            raise ValueError("Action must be 0-3")
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

        # Store old position for cache invalidation if needed
        old_x, old_y, old_orientation = self.robot_x, self.robot_y, self.robot_orientation

        # Update robot position
        self.robot_x, self.robot_y, self.robot_orientation = self._update_robot_position(
            self.robot_x, self.robot_y, self.robot_orientation, v_left, v_right
        )

        # Cast LIDAR using cached version
        intersections = self.cast_lidar_rays_optimized(
            self.robot_x, self.robot_y, self.robot_orientation
        )

        # Update map only if rendering (reuse intersections)
        new_cells = 0
        if self.render_flag:
            new_cells = self._update_map(intersections=intersections)

        # Observation
        obs = self._get_observation(intersections)

        # Reward
        reward = self._calculate_reward()

        self._log_step(action, intersections, reward, extra_info)

        # Step bookkeeping
        self.current_step += 1
        done = self.current_step >= self.max_steps
        info = {
            "new_cells": new_cells, 
            "coverage": self._get_coverage(), 
            "action": action, 
            "reward": reward,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_evictions": self.cache_evictions,
            "cache_size": len(self.lidar_cache)
        }

        return obs, reward, done, info

    def _log_step(self, action, intersections, reward, extra_info):
        lidar_distances = [sqrt((inter[0] - self.robot_x)**2 + (inter[1] - self.robot_y)**2) if inter else self.ray_length
                        for inter in intersections]
        
        # Row Order: step, strategy, action, run_start, run_length, [lidar], episode, reward, x, y, orientation
        row = [
            self.current_step,
            self.strategy_name,
            action,
            extra_info.get('run_start', ''),
            extra_info.get('run_length', '')
        ] + lidar_distances + [
            self.episode,
            reward,
            self.robot_x,
            self.robot_y,
            self.robot_orientation
        ]
        
        self.log_buffer.append(row)

        # Save buffer periodically
        if len(self.log_buffer) == 100:
            path = os.path.join(self.output_dir, f"log_{self.current_step + 1}.csv")
            header = ['step', 'strategy', 'action', 'run_start', 'run_length'] + \
                    [f'ray_{i}' for i in range(self.num_rays)] + \
                    ['episode', 'reward', 'x', 'y', 'orientation']
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self.log_buffer)
            self.log_buffer = []

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
        self._draw_robot(self.pygame)
        self._draw_lidar(self.pygame)
        self.pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        # Save any remaining steps in the buffer
        if self.log_buffer:
            final_path = os.path.join(self.output_dir, f"log_{self.current_step}.csv")
            header = ['step', 'strategy', 'action', 'run_start', 'run_length'] + \
                     [f'ray_{i}' for i in range(self.num_rays)] + \
                     ['episode', 'reward', 'x', 'y', 'orientation']
            with open(final_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self.log_buffer)
        
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
        
        if self._check_collision(new_x, new_y):
            return x, y, orientation
        return new_x, new_y, new_orientation

    def _check_collision(self, x, y):
        """Check if robot at position (x,y) collides with obstacles"""
        return not self._is_position_free(x, y)

    def _draw_lidar(self, pygame):
        intersections = self.cast_lidar_rays_optimized(self.robot_x, self.robot_y, self.robot_orientation)
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

    def cast_lidar_rays_optimized(self, robot_x, robot_y, orientation, num_rays=None, max_range=None):
        """
        Public method that uses cache for LIDAR readings.
        Maintains backward compatibility.
        """
        if num_rays is not None or max_range is not None:
            # If parameters differ from defaults, don't use cache
            return self._compute_lidar_rays(robot_x, robot_y, orientation, num_rays, max_range)
        
        return self._get_cached_lidar(robot_x, robot_y, orientation)
    
    def _compute_lidar_rays(self, robot_x, robot_y, orientation, num_rays=None, max_range=None):
        """
        Original LIDAR computation logic (renamed from cast_lidar_rays_optimized)
        """
        if num_rays is None:
            num_rays = self.num_rays
        if max_range is None:
            max_range = self.ray_length
            
        intersections = []
        angles = orientation + self.lidar_angles
        
        for angle_deg in angles:
            angle_rad = np.radians(angle_deg)
            end_x = int(robot_x + max_range * np.cos(angle_rad))
            end_y = int(robot_y + max_range * np.sin(angle_rad))
            
            line_points = self._bresenham_line(int(robot_x), int(robot_y), end_x, end_y)
            
            closest_intersection = None
            for point in line_points:
                x, y = point
                if not (0 <= x < self.map_width and 0 <= y < self.map_height):
                    break
                    
                if self.obstacle_map[y, x] == 1:
                    closest_intersection = (x, y)
                    break
            
            intersections.append(closest_intersection)
        
        return intersections

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

    def _get_observation(self, intersections=None):
        """Get LIDAR distances as observation"""
        if intersections is None:
            intersections = self.cast_lidar_rays_optimized(
                self.robot_x, self.robot_y, self.robot_orientation
            )
        distances = [sqrt((inter[0] - self.robot_x)**2 + (inter[1] - self.robot_y)**2) if inter else self.ray_length
                     for inter in intersections]
        return np.array(distances, dtype=np.float32)

    def _get_coverage(self):
        """Calculate percentage of explored cells (free + obstacle). 0 - free, 1 - obstacle, -1 - unexplored"""
        return 100 * np.sum(self.exploration_grid >= 0) / (self.grid_width * self.grid_height)

    def _update_map(self, intersections):
        """
        Update exploration grid using precomputed LIDAR intersections.
        Only called if rendering/logging is needed.
        """
        new_cells = 0
        angles = np.linspace(self.robot_orientation - 45, self.robot_orientation + 45, self.num_rays)

        for angle_deg, inter in zip(angles, intersections):
            angle_rad = np.radians(angle_deg)

            if inter is not None:
                # Ray hit obstacle → mark obstacle and free path
                ox, oy = int(inter[0]), int(inter[1])

                # Mark free path to obstacle
                line_points = self._bresenham_line(int(self.robot_x), int(self.robot_y), ox, oy)
                for px, py in line_points[:-1]:
                    if 0 <= px < self.grid_width and 0 <= py < self.grid_height:
                        if self.exploration_grid[px, py] == -1:
                            self.exploration_grid[px, py] = 0
                            new_cells += 1

                # Mark obstacle
                if 0 <= ox < self.grid_width and 0 <= oy < self.grid_height:
                    if self.exploration_grid[ox, oy] == -1:
                        self.exploration_grid[ox, oy] = 1
                        new_cells += 1
            else:
                # No intersection → mark full ray as free
                end_x = int(self.robot_x + self.ray_length * np.cos(angle_rad))
                end_y = int(self.robot_y + self.ray_length * np.sin(angle_rad))
                line_points = self._bresenham_line(int(self.robot_x), int(self.robot_y), end_x, end_y)

                for px, py in line_points:
                    if 0 <= px < self.grid_width and 0 <= py < self.grid_height:
                        if self.exploration_grid[px, py] == -1:
                            self.exploration_grid[px, py] = 0
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
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if self.exploration_grid[x, y] == 0:  # Free space
                    color = (0, 255, 0)  # Solid green
                    pygame.draw.rect(self.screen, color, 
                                (x * self.scale, y * self.scale, self.scale, self.scale))
                elif self.exploration_grid[x, y] == 1:  # Obstacle
                    color = (255, 0, 0)  # Solid red
                    pygame.draw.rect(self.screen, color,
                                (x * self.scale, y * self.scale, self.scale, self.scale))
                    
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