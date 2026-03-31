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

    print(f"[INFO] Map: {env_name} ({w}x{h})")
    print(f"[INFO] LUT: {lut_filename} | Angles: {num_angles}")
    print(f"[INFO] Mode: {'Rendering' if render else 'Headless Benchmark'}")

    # 4. Execution Loop
    start_time = time.time()
    for i in range(num_samples):
        # Sample valid pose
        rx, ry = random.choice(free_points)
        yaw = random.uniform(0, 360)
        
        # Calculate LIDAR fan (standard 90-degree spread) 
        # In this env, lidar_angles are typically -45 to 45 
        ray_angles = np.linspace(yaw - 45, yaw + 45, 100)
        angle_indices = ((ray_angles % 360) / 360 * num_angles).astype(np.int32) % num_angles
        
        # Constant-time O(1) retrieval
        distances = lut[ry, rx, angle_indices]

        if render:
            display = cv2.resize(map_img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
            for ang, dist in zip(ray_angles, distances):
                angle_rad = np.radians(ang % 360)
                # Scale coordinates for the window 
                s_pt = (int(rx * scale), int(ry * scale))
                e_pt = (int((rx + dist * np.cos(angle_rad)) * scale),
                        int((ry + dist * np.sin(angle_rad)) * scale))
                
                # Yellow for obstacle hits, Gray for open space 
                color = (0, 255, 255) if dist < (lut.max() - 1) else (120, 120, 120)
                cv2.line(display, s_pt, e_pt, color, 1)

            # Draw robot circle and heading 
            cv2.circle(display, (int(rx * scale), int(ry * scale)), int(2 * scale), (255, 0, 0), -1)
            
            cv2.putText(display, f"Pose {i+1}/{num_samples}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("LUT Visualization", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    end_time = time.time()
    avg_ms = ((end_time - start_time) / num_samples) * 1000
    print(f"\n=== Results ===")
    print(f"Total Time: {end_time - start_time:.4f}s")
    print(f"Avg per Pose: {avg_ms:.4f}ms")
    
    if render:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parameters to change
    TARGET_IMAGE = "6.png"
    ANGULAR_RES = 360
    
    # Run high-speed headless test
    run_lut_system(TARGET_IMAGE, ANGULAR_RES, num_samples=100_000, render=False)

    print("\nNow running with rendering enabled. Press 'q' to quit visualization.")
    
    # Run visual test
    # run_lut_system(TARGET_IMAGE, ANGULAR_RES, num_samples=100, render=True)