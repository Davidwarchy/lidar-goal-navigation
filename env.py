import torch
import torch.nn.functional as F
import sys
import os
import csv
import random
import cv2
import json
from datetime import datetime
from math import cos, sin, radians, sqrt
import warnings
import numpy as np

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

try:
    import cv2
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(_dep_error_message("opencv-python (import name: cv2)")) from e

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
                goal_spawn_dist=30.0,
                verbose=False,
                device='cuda',
                # Noise parameters
                lidar_noise_type="none",
                lidar_noise_std=5.0,
                lidar_noise_max=10.0,
                lidar_dropout_p=0.05,
                motor_noise_type="none",
                motor_slip_prob=0.1,
                motor_slip_mag=0.5,
                motor_deadzone_threshold=0.1,
                observation_delay=0):

        self.num_envs = num_envs
        self.map_image_path = map_image_path
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        # Load map
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
        
        # Convert to torch tensors on GPU
        self.obstacle_map_tensor = torch.from_numpy(self.obstacle_map).float().to(self.device)
        self.distance_map_tensor = torch.from_numpy(self.distance_map).float().to(self.device)

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

        # Vectorized State variables (on GPU)
        self.robot_x = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.robot_y = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.robot_orientation = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.energy = torch.full((num_envs,), max_steps, dtype=torch.int32, device=self.device)
        self.health = torch.full((num_envs,), 500, dtype=torch.int32, device=self.device)
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.current_step = 0
        
        # Survival/Goal parameters
        self.goal_x = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.goal_y = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.goal_success_dist = 6.0 
        self.goal_spawn_dist = goal_spawn_dist

        self.lidar_angles = torch.linspace(-45, 45, self.num_rays, device=self.device)
        self.strategy_name = strategy_name
        self.strategy_parameters = strategy_parameters or {}

        # Output Setup – now output_dir is exactly the trial directory
        self.output_dir = output_dir if output_dir is not None else "output"
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

        # ---------- Pre-allocated buffers for GPU memory reuse ----------
        # These tensors are reused every step to avoid new allocations.
        self._v_left = torch.zeros(num_envs, device=self.device)
        self._v_right = torch.zeros(num_envs, device=self.device)
        self._new_x = torch.zeros(num_envs, device=self.device)
        self._new_y = torch.zeros(num_envs, device=self.device)
        self._new_orient = torch.zeros(num_envs, device=self.device)
        self._ixs = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._iys = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._dist_vals = torch.zeros(num_envs, device=self.device)
        self._free_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._update_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._collision_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._linear_vel = torch.zeros(num_envs, device=self.device)
        self._angular_vel = torch.zeros(num_envs, device=self.device)
        # LIDAR distance buffer (kept on GPU, copied to CPU only when needed)
        self._lidar_distances = torch.zeros((num_envs, num_rays), device=self.device)

        # ---------- Noise parameters ----------
        self.lidar_noise_type = lidar_noise_type
        self.lidar_noise_std = lidar_noise_std
        self.lidar_noise_max = lidar_noise_max
        self.lidar_dropout_p = lidar_dropout_p
        self.motor_noise_type = motor_noise_type
        self.motor_slip_prob = motor_slip_prob
        self.motor_slip_mag = motor_slip_mag
        self.motor_deadzone_threshold = motor_deadzone_threshold
        self.observation_delay = observation_delay
        
        # Latency buffer (will be initialized in reset)
        self.obs_buffer = None
        self.obs_buffer_idx = 0
    def _load_lut(self):
        base_name = os.path.splitext(os.path.basename(self.map_image_path))[0]
        lut_path = os.path.join("environments", "luts", f"{base_name}_360.npy")
        if os.path.exists(lut_path):
            lut_np = np.load(lut_path, mmap_mode='r')
            self.lidar_lut = torch.from_numpy(lut_np.copy()).to(self.device)
        else:
            self.use_lut = False
            print(f"Warning: LUT not found at {lut_path}, falling back to ray marching")

    def _save_metadata(self):
        """Save comprehensive run metadata for the vectorized session."""
        metadata = {
            "run_datetime": datetime.now().isoformat(),
            "strategy_name": self.strategy_name,
            "num_envs": self.num_envs,
            "max_steps": self.max_steps,
            "continue_after_goal": self.continue_after_goal,
            "device": str(self.device),
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
            x, y = self._find_free_position(center_x, center_y)
            self.robot_x[i] = x
            self.robot_y[i] = y
            gx, gy = self._spawn_reward(x, y, self.goal_spawn_dist)
            self.goal_x[i] = gx
            self.goal_y[i] = gy
            
        self.robot_orientation.zero_()
        self.current_step = 0
        self.energy.fill_(self.max_steps)
        self.health.fill_(500)
        self.done.fill_(False)
        
        # Initialize latency buffer if needed
        if self.observation_delay > 0:
            self.obs_buffer = torch.zeros(self.observation_delay + 1, self.num_envs, self.num_rays, device=self.device)
            self.obs_buffer_idx = 0
            # Fill buffer with initial observation
            initial_obs = self._get_observation_no_noise()
            for i in range(self.observation_delay + 1):
                self.obs_buffer[i] = initial_obs

        return self._get_observation()

    def _find_free_position(self, start_x, start_y, max_radius=100):
        for radius in range(0, max_radius, 5):
            for angle in np.linspace(0, 2*np.pi, 36):
                x, y = int(start_x + radius * np.cos(angle)), int(start_y + radius * np.sin(angle))
                if self._is_position_free(x, y):
                    return x, y
        return start_x, start_y

    def _is_position_free(self, x, y):
        ix, iy = int(x), int(y)
        if not (0 <= ix < self.map_width and 0 <= iy < self.map_height):
            return False
        return self.distance_map[iy, ix] >= self.robot_radius

    def _spawn_reward(self, start_x, start_y, distance):
        # ---- 1. sanity check: is the radius even possible in the map? ----
        if distance <= 0:
            raise ValueError("Distance must be positive")

        max_radius = min(
            start_x, self.map_width - start_x,
            start_y, self.map_height - start_y
        )

        if distance > max_radius:
            raise ValueError(
                f"Distance {distance} is too large for map bounds. "
                f"Max feasible radius from start is {max_radius}."
            )

        # ---- 2. sampling attempts ----
        for _ in range(200):
            angle = np.random.uniform(0, 2 * np.pi)
            tx = start_x + distance * np.cos(angle)
            ty = start_y + distance * np.sin(angle)

            ix, iy = int(round(tx)), int(round(ty))

            if (0 <= ix < self.map_width and
                0 <= iy < self.map_height and
                self.obstacle_map[iy, ix] == 0):
                return tx, ty

        # ---- 3. failure case ----
        raise RuntimeError(
            f"Failed to spawn reward after 200 attempts at distance {distance}"
        )

    def step(self, actions, action_space="discrete"):
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        actions : tensor or array-like
            For discrete: (num_envs,) tensor of integers in {0,1,2,3}
            For continuous: (num_envs, 2) tensor of [linear_velocity, angular_velocity]
        action_space : str
            "discrete" or "continuous"
        
        Returns:
        --------
        obs : torch.Tensor (num_envs, num_rays) on GPU
        rewards : torch.Tensor (num_envs) on GPU
        done : torch.Tensor (num_envs) bool on GPU
        info : dict with 'goal_reached' as GPU tensor
        """
        with torch.no_grad():
            # Convert actions to torch tensor on device if needed
            if not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions, device=self.device)
            
            # Only decrement health for robots that are still active (not done)
            active_mask = ~self.done
            
            if action_space == "discrete":
                # Reuse pre‑allocated v_left/v_right
                self._v_left.zero_()
                self._v_right.zero_()
                # Up (action 0)
                mask0 = (actions == 0)
                self._v_left[mask0] = self.linear_speed
                self._v_right[mask0] = self.linear_speed
                # Down (action 1)
                mask1 = (actions == 1)
                self._v_left[mask1] = -self.linear_speed
                self._v_right[mask1] = -self.linear_speed
                # Left (action 2)
                mask2 = (actions == 2)
                self._v_left[mask2] = self.linear_speed / 4
                self._v_right[mask2] = -self.linear_speed / 4
                # Right (action 3)
                mask3 = (actions == 3)
                self._v_left[mask3] = -self.linear_speed / 4
                self._v_right[mask3] = self.linear_speed / 4
                v_left = self._v_left
                v_right = self._v_right
            else:  # continuous action space
                # Actions shape: (num_envs, 2) - [linear_velocity, angular_velocity]
                linear_vel = actions[:, 0]
                angular_vel = actions[:, 1]  # degrees per second
                v_left = linear_vel - (angular_vel * self.wheel_base / 2) / self.wheel_radius
                v_right = linear_vel + (angular_vel * self.wheel_base / 2) / self.wheel_radius
            
            # Update positions
            self._update_robot_positions(v_left, v_right)
            
            # Get Perception
            obs = self._get_observation()   # GPU tensor

            # Check for Success (Goal reached)
            dx = self.robot_x - self.goal_x
            dy = self.robot_y - self.goal_y
            goal_reached = (dx*dx + dy*dy) <= self.goal_success_dist ** 2
            
            # Success handling
            # Reward is only for the step they actually find it
            rewards = torch.where(goal_reached & active_mask, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))

            # Resource Depletion
            # In 'Life on Silicon', every step costs 1 energy point 
            # We only decrement if they haven't finished yet
            self.energy[active_mask] -= 1
            
            # Final Termination Check
            # If NOT continuing after goal, reaching it marks them as done immediately
            if not self.continue_after_goal:
                # Done if: No energy OR No health OR Reached Max Steps OR Reached Goal
                new_dones = (self.energy <= 0) | (self.health <= 0) | \
                            (self.current_step >= self.max_steps) | goal_reached
            else:
                # If continuing, they only stop when they physically 'die' or time runs out
                new_dones = (self.current_step >= self.max_steps)

            self.current_step += 1
            self.done = self.done | new_dones

        # Return all as GPU tensors
        return obs, rewards, self.done, {"goal_reached": goal_reached}

    def _update_robot_positions(self, v_left, v_right):
        with torch.no_grad():
            # --- Motor Noise: Wheel Slip ---
            if self.motor_noise_type == "slip":
                slip_mask = torch.rand_like(v_left) < self.motor_slip_prob
                # Slip reduces wheel velocity by random amount
                slip_effect = 1.0 - torch.rand_like(v_left) * self.motor_slip_mag
                v_left = torch.where(slip_mask, v_left * slip_effect, v_left)
                v_right = torch.where(slip_mask, v_right * slip_effect, v_right)
            
            # --- Motor Noise: Dead Zone ---
            elif self.motor_noise_type == "deadzone":
                # Small commands have no effect
                left_small = torch.abs(v_left) < self.motor_deadzone_threshold
                right_small = torch.abs(v_right) < self.motor_deadzone_threshold
                v_left = torch.where(left_small, torch.zeros_like(v_left), v_left)
                v_right = torch.where(right_small, torch.zeros_like(v_right), v_right)
            
            # Compute linear and angular velocities (use buffers)
            self._linear_vel = (v_left + v_right) / 2 * self.wheel_radius
            self._angular_vel = torch.rad2deg((v_right - v_left) / self.wheel_base * self.wheel_radius)
            
            # Compute new candidate states (reuse pre‑allocated buffers)
            self._new_x = self.robot_x + self._linear_vel * torch.cos(torch.deg2rad(self.robot_orientation)) * self.dt
            self._new_y = self.robot_y + self._linear_vel * torch.sin(torch.deg2rad(self.robot_orientation)) * self.dt
            self._new_orient = (self.robot_orientation + self._angular_vel * self.dt) % 360
            
            # Collision check for all robots
            self._ixs = self._new_x.long()
            self._iys = self._new_y.long()
            valid = (self._ixs >= 0) & (self._ixs < self.map_width) & (self._iys >= 0) & (self._iys < self.map_height)
            
            # Safe distance transform lookup (reuse self._dist_vals)
            self._dist_vals.zero_()
            valid_indices = torch.where(valid)
            self._dist_vals[valid_indices] = self.distance_map_tensor[self._iys[valid_indices], self._ixs[valid_indices]]
            
            self._free_mask = self._dist_vals >= self.robot_radius
            self._update_mask = self._free_mask & ~self.done
    
            # Update only if free and not already done
            self.robot_x[self._update_mask] = self._new_x[self._update_mask]
            self.robot_y[self._update_mask] = self._new_y[self._update_mask]
            self.robot_orientation[self._update_mask] = self._new_orient[self._update_mask]
            
            # Penalty for collision
            self._collision_mask = ~self._free_mask & ~self.done
            self.health[self._collision_mask] -= 1

    def _get_observation(self):
        """
        Get LIDAR distances for the entire population. 
        Prioritises the O(1) LUT but falls back to vectorized ray marching.
        Returns GPU tensor of shape (num_envs, num_rays).
        """
        distances = self._get_observation_no_noise()
        
        # Apply perceptual noise
        if self.lidar_noise_type != "none":
            distances = self._apply_lidar_noise(distances)
        
        # Apply latency delay
        if self.observation_delay > 0:
            distances = self._apply_latency(distances)
        
        return distances

    def _get_observation_no_noise(self):
        """Get raw LIDAR distances without noise or latency."""
        if self.use_lut and self.lidar_lut is not None:
            with torch.no_grad():
                # 1. Prepare indices (H, W)
                ixs = torch.round(self.robot_x).long()
                iys = torch.round(self.robot_y).long()
                
                # Boundary clipping for safety
                ixs = torch.clamp(ixs, 0, self.map_width - 1)
                iys = torch.clamp(iys, 0, self.map_height - 1)
                
                # 2. Map world angles to LUT indices
                # Shape: (num_envs, num_rays)
                angles = (self.robot_orientation[:, None] + self.lidar_angles) % 360
                # FIX: Clip to [0, 359] to avoid index 360 due to floating point precision
                angle_idxs = torch.clamp(angles.long(), 0, 359)
                
                # Direct GPU indexing into LUT
                self._lidar_distances[:] = self.lidar_lut[iys[:, None], ixs[:, None], angle_idxs]
                return self._lidar_distances   # GPU tensor

        # FALLBACK: Vectorized Ray Marching (if LUT is not used/available)
        # Raise warning if LUT was intended but not loaded
        if self.use_lut and self.lidar_lut is None:
            warnings.warn("use_lut is True but lidar_lut was not loaded. Falling back to vectorized ray marching.")

        return self._compute_vectorized_ray_marching()

    def _compute_vectorized_ray_marching(self):
        """
        Computes LIDAR for all robots simultaneously without a LUT.
        Complexity: O(num_envs * num_rays * ray_length)
        Returns GPU tensor.
        """
        with torch.no_grad():
            # Create ray steps on GPU
            steps = torch.arange(self.ray_length, device=self.device)
            
            # Absolute angles for every ray of every robot: (num_envs, num_rays)
            angles_rad = torch.deg2rad(self.robot_orientation[:, None] + self.lidar_angles)
            
            # Unit vectors: (num_envs, num_rays, 1)
            dx = torch.cos(angles_rad)[:, :, None]
            dy = torch.sin(angles_rad)[:, :, None]
            
            # Compute all potential ray coordinates: (num_envs, num_rays, ray_length)
            # x_points = robot_x + dx * steps
            ray_x = (self.robot_x[:, None, None] + dx * steps).long()
            ray_y = (self.robot_y[:, None, None] + dy * steps).long()
            
            # Clip to map boundaries to avoid index errors
            ray_x_clp = torch.clamp(ray_x, 0, self.map_width - 1)
            ray_y_clp = torch.clamp(ray_y, 0, self.map_height - 1)
            
            # Check hits against the obstacle map
            # Resulting 'hits' shape: (num_envs, num_rays, ray_length)
            hits = self.obstacle_map_tensor[ray_y_clp, ray_x_clp] == 1
            
            # Find the first hit along the 'ray_length' axis
            # argmax returns the first index of 'True'
            hit_indices = torch.argmax(hits.float(), dim=2)
            
            # Identify rays that never hit anything
            no_hits = ~torch.any(hits, dim=2)
            
            # Final distances: use hit index or default to max ray_length
            distances = hit_indices.float()
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
        
        # Draw goals for all active robots (need CPU values for pygame)
        robot_x_cpu = self.robot_x.cpu().numpy() if self.robot_x.is_cuda else self.robot_x.numpy()
        robot_y_cpu = self.robot_y.cpu().numpy() if self.robot_y.is_cuda else self.robot_y.numpy()
        goal_x_cpu = self.goal_x.cpu().numpy() if self.goal_x.is_cuda else self.goal_x.numpy()
        goal_y_cpu = self.goal_y.cpu().numpy() if self.goal_y.is_cuda else self.goal_y.numpy()
        done_cpu = self.done.cpu().numpy() if self.done.is_cuda else self.done.numpy()
        robot_orient_cpu = self.robot_orientation.cpu().numpy() if self.robot_orientation.is_cuda else self.robot_orientation.numpy()
        
        for i in range(self.num_envs):
            if not done_cpu[i]:
                goal_x = int(goal_x_cpu[i] * self.scale)
                goal_y = int(goal_y_cpu[i] * self.scale)
                radius = int(self.goal_success_dist * self.scale)
                
                # Outer ring (bright green)
                self.pygame.draw.circle(self.screen, (0, 255, 0), (goal_x, goal_y), radius, 2)
                # Inner circle (lighter green)
                self.pygame.draw.circle(self.screen, (100, 255, 100), (goal_x, goal_y), max(2, radius // 2))
                # Center bullseye
                self.pygame.draw.circle(self.screen, (255, 255, 255), (goal_x, goal_y), max(2, radius // 4))
        
        # Draw active robots
        for i in range(self.num_envs):
            if not done_cpu[i]:
                center_x = int(robot_x_cpu[i] * self.scale)
                center_y = int(robot_y_cpu[i] * self.scale)
                radius = int(self.robot_radius * self.scale)
                
                # Robot body
                self.pygame.draw.circle(self.screen, (0, 0, 255), (center_x, center_y), radius)
                
                # Heading direction (red line)
                arrow_len = radius * 1.5
                angle_rad = np.radians(robot_orient_cpu[i])
                end_x = int(center_x + arrow_len * np.cos(angle_rad))
                end_y = int(center_y + arrow_len * np.sin(angle_rad))
                self.pygame.draw.line(self.screen, (255, 0, 0), (center_x, center_y), (end_x, end_y), 2)
                
                # Tip dot
                self.pygame.draw.circle(self.screen, (255, 255, 0), (end_x, end_y), 2)
        
        self.pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self._pygame_initialized: self.pygame.quit()
        # Optional: clear GPU cache (not called automatically, but safe to add here)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_gpu_cache(self):
        """Explicitly clear GPU memory cache. Can be called between generations."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _apply_lidar_noise(self, distances):
        """Add noise to LiDAR readings."""
        if self.lidar_noise_type == "gaussian":
            noise = torch.randn_like(distances) * self.lidar_noise_std
            distances = distances + noise
            distances = torch.clamp(distances, 0, self.ray_length)
            
        elif self.lidar_noise_type == "uniform":
            noise = (torch.rand_like(distances) * 2 - 1) * self.lidar_noise_max
            distances = distances + noise
            distances = torch.clamp(distances, 0, self.ray_length)
            
        elif self.lidar_noise_type == "dropout":
            mask = torch.rand_like(distances) > self.lidar_dropout_p
            distances = torch.where(mask, distances, torch.full_like(distances, self.ray_length))
        
        return distances

    def _apply_latency(self, distances):
        """Apply observation delay using circular buffer."""
        # Store current observation in buffer
        self.obs_buffer[self.obs_buffer_idx] = distances
        
        # Retrieve delayed observation
        delayed_idx = (self.obs_buffer_idx - self.observation_delay) % (self.observation_delay + 1)
        delayed_obs = self.obs_buffer[delayed_idx]
        
        # Advance buffer index
        self.obs_buffer_idx = (self.obs_buffer_idx + 1) % (self.observation_delay + 1)
        
        return delayed_obs