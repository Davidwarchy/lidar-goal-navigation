import cv2
import numpy as np
import os
import random

def visualize_lut_scaled(image_path, lut_path, scale=4, num_rays=100, ray_spread=90):
    """
    Visualizes the 3D LUT at a higher scale for better clarity.
    """
    # 1. Load the original map and LUT
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"[ERROR] Could not load image {image_path}")
        return
    
    # Load LUT: Shape is (H, W, Angles) 
    lut = np.load(lut_path)
    h, w, num_angles = lut.shape
    
    # 2. Scale the base image for visibility 
    window_w, window_h = w * scale, h * scale
    display_img = cv2.resize(original_img, (window_w, window_h), interpolation=cv2.INTER_NEAREST)

    # 3. Find a random free position in original coordinates 
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    free_y, free_x = np.where(gray > 127) # Assuming white is free space 
    if len(free_x) == 0:
        print("[ERROR] No free space found in map.")
        return
        
    idx = random.randint(0, len(free_x) - 1)
    rx, ry = free_x[idx], free_y[idx]
    
    # 4. Random Orientation (Yaw)
    yaw_deg = random.uniform(0, 360) 
    
    # 5. Determine Ray Angles (Spread around the yaw) 
    half_spread = ray_spread / 2
    ray_angles = np.linspace(yaw_deg - half_spread, yaw_deg + half_spread, num_rays)
    
    # 6. Draw Rays using LUT data scaled to the display window
    for angle in ray_angles:
        angle_wrapped = angle % 360
        # Map degree to LUT index based on angular resolution 
        angle_idx = int((angle_wrapped / 360) * num_angles) % num_angles
        
        # Get pre-computed distance from LUT
        distance = lut[ry, rx, angle_idx]
        
        # Calculate end point in original coordinates, then scale 
        angle_rad = np.radians(angle_wrapped)
        ex_orig = rx + distance * np.cos(angle_rad)
        ey_orig = ry + distance * np.sin(angle_rad)
        
        # Scale for display 
        start_pt = (int(rx * scale), int(ry * scale))
        end_pt = (int(ex_orig * scale), int(ey_orig * scale))
        
        # Color: Yellow if it hit an obstacle, Gray if it reached max range 
        color = (0, 255, 255) if distance < (lut.max() - 1) else (150, 150, 150)
        cv2.line(display_img, start_pt, end_pt, color, 1)

    # 7. Draw Robot and Heading 
    center_scaled = (int(rx * scale), int(ry * scale))
    cv2.circle(display_img, center_scaled, int(3 * scale), (255, 0, 0), -1) # Blue robot
    
    # Heading indicator (Red line)
    head_x = int((rx + 8) * scale * np.cos(np.radians(yaw_deg))) # Relative shift
    # Simplified heading line for visual check
    head_end = (int((rx + 10 * np.cos(np.radians(yaw_deg))) * scale),
                int((ry + 10 * np.sin(np.radians(yaw_deg))) * scale))
    cv2.line(display_img, center_scaled, head_end, (0, 0, 255), 2)

    # 8. Show Result
    print(f"[INFO] Rendering at {scale}x scale. Press any key to close.")
    cv2.imshow(f"3D LUT Visualization ({scale}x Scale)", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MAP_PATH = "environments/images/6.png" 
    LUT_PATH = "environments/luts/6_360.npy"
    
    if os.path.exists(LUT_PATH):
        visualize_lut_scaled(MAP_PATH, LUT_PATH, scale=6)
    else:
        print(f"File not found: {LUT_PATH}. Run your generator first.")