import cv2
import numpy as np
import os
import time
import random

def run_lut_system(env_name="6.png", num_angles=360, num_samples=1000, render=True, scale=6):
    """
    Finds the map and LUT, then runs a benchmark or visualization.
    """
    # 1. Path Resolution
    image_path = os.path.join("environments", "images", env_name)
    base_name = os.path.splitext(env_name)[0]
    lut_filename = f"{base_name}_{num_angles}.npy"
    lut_path = os.path.join("environments", "luts", lut_filename)

    # Check existence
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return
    if not os.path.exists(lut_path):
        print(f"[ERROR] LUT not found: {lut_path}. Run generator for {num_angles} angles first.")
        return

    # 2. Load Resources
    map_img = cv2.imread(image_path)
    lut = np.load(lut_path)
    h, w, _ = lut.shape
    
    # 3. Identify Valid Free Space
    gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    _, obstacle_map = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV) 
    free_y, free_x = np.where(obstacle_map == 0)
    free_points = list(zip(free_x, free_y))

    # --- OPTIMIZATION: PRE-COMPUTATION ---
    # Instead of calling linspace 100,000 times, we define the sensor geometry once.
    num_rays = 100
    sensor_fan = np.linspace(-45, 45, num_rays) 
    # Pre-calculate unit vectors for rendering to avoid sin/cos in the loop
    fan_rad = np.radians(sensor_fan)
    unit_cols = np.cos(fan_rad)
    unit_rows = np.sin(fan_rad)
    scale_factor = num_angles / 360.0
    
    print(f"[INFO] Running sequential benchmark for {num_samples} samples...")

    # 4. Execution Loop
    start_time = time.time()
    for i in range(num_samples):
        # Sample valid pose
        rx, ry = random.choice(free_points)
        yaw = random.uniform(0, 360)
        
        # Sequential Step 2: Observation (The optimized part)
        # We broadcast the single 'yaw' across the pre-calculated 'sensor_fan'
        ray_angles = (yaw + sensor_fan) % 360
        angle_indices = (ray_angles * scale_factor).astype(np.int32, copy=False)
        
        # O(1) LUT Retrieval
        distances = lut[ry, rx, angle_indices]

        if render:
            # Visualization logic remains sequential for the "single robot" feel
            display = cv2.resize(map_img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
            # Use pre-calculated trig for the current yaw
            yaw_rad = np.radians(yaw)
            c, s = np.cos(yaw_rad), np.sin(yaw_rad)
            
            for j in range(num_rays):
                dist = distances[j]
                ang_total_rad = np.radians(ray_angles[j])
                
                s_pt = (int(rx * scale), int(ry * scale))
                e_pt = (int((rx + dist * np.cos(ang_total_rad)) * scale),
                        int((ry + dist * np.sin(ang_total_rad)) * scale))
                
                # Yellow for obstacle hits, Gray for open space 
                color = (0, 255, 255) if dist < (lut.max() - 1) else (120, 120, 120)
                cv2.line(display, s_pt, e_pt, color, 1)

            # Draw robot circle and heading 
            cv2.circle(display, (int(rx * scale), int(ry * scale)), int(2 * scale), (255, 0, 0), -1)
            cv2.imshow("Sequential LUT Visualization", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    end_time = time.time()
    print(f"\n=== Results ===\nAvg per Pose: {((end_time - start_time) / num_samples) * 1000:.4f}ms")

if __name__ == "__main__":
    run_lut_system("6.png", 360, num_samples=100_000, render=False)

    #  python .\luts\load_iterate.py (0.030ms per step)