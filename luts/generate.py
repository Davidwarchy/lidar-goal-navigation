import cv2
import numpy as np
import os
from tqdm import tqdm

def generate_3d_lidar_lut(image_path, num_angles=360, ray_length=200, output_dir="environments/luts"):
    """
    Generates a 3D LUT: [height, width, angle_index] storing distance to obstacles.
    Includes dual-level progress bars for tracking.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load and process map
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    if img is None: 
        print(f"Error: Could not load {image_path}")
        return
    
    h, w = img.shape 
    _, obstacle_map = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV) 
    
    # 2. Setup angles and steps
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    angles_rad = np.radians(angles) 
    steps = np.arange(ray_length) 
    
    # 3. Initialize 3D LUT (initialized to max ray length)
    lidar_lut = np.full((h, w, num_angles), ray_length, dtype=np.float32)

    # Precompute unit vectors for all angles
    dx = np.cos(angles_rad) 
    dy = np.sin(angles_rad) 

    print(f"[INFO] Generating 3D LUT: {h}x{w} map | {num_angles} angles | {ray_length}px range")
    
    # 4. Iterate through every pixel with Row-level tqdm
    for y in tqdm(range(h), desc="Processing Rows", unit="row"):
        # Optional: Nested tqdm for columns if the map is very large, 
        # but usually row-level is enough for visibility.
        for x in range(w):
            # Only compute for free space (0 = free in our obstacle_map) 
            if obstacle_map[y, x] == 1:
                continue
            
            # Vectorized ray casting for all angles at this specific (x, y)
            # Shapes: (num_angles, ray_length)
            ray_x = (x + np.outer(dx, steps)).astype(np.int32) 
            ray_y = (y + np.outer(dy, steps)).astype(np.int32) 

            # Valid mask for boundaries 
            valid = (ray_x >= 0) & (ray_x < w) & (ray_y >= 0) & (ray_y < h) 
            
            # Clip for safe indexing 
            rx_c = np.clip(ray_x, 0, w - 1) 
            ry_c = np.clip(ray_y, 0, h - 1) 
            
            # Hit detection 
            hits = (obstacle_map[ry_c, rx_c] == 1) & valid 
            
            # Vectorized search for first hit index per angle
            # argmax returns the first index of 'True' 
            has_hit = np.any(hits, axis=1) 
            first_hits = np.argmax(hits, axis=1) 
            
            # Assign distances: if no hit, it remains the default ray_length
            lidar_lut[y, x, has_hit] = first_hits[has_hit]

    # 5. Save
    base_name = os.path.splitext(os.path.basename(image_path))[0] 
    output_path = os.path.join(output_dir, f"{base_name}_{num_angles}.npy")
    np.save(output_path, lidar_lut)
    print(f"\n[SUCCESS] Saved 3D LUT to: {output_path}")

if __name__ == "__main__":
    # Ensure this points to your actual environments directory
    generate_3d_lidar_lut("environments/images/6.png", num_angles=360)