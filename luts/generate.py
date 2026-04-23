# luts/generate.py - Updated with PyTorch GPU support
import torch
import cv2
import numpy as np
import os
from tqdm import tqdm

def generate_3d_lidar_lut(image_path, num_angles=360, ray_length=200, output_dir="environments/luts", device='cuda'):
    """
    Generates a 3D LUT: [height, width, angle_index] storing distance to obstacles.
    Uses PyTorch for GPU acceleration.
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
    
    # Convert to torch tensors on GPU
    obstacle_map_tensor = torch.from_numpy(obstacle_map).float().to(device)
    
    # 2. Setup angles and steps
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    angles_rad = np.radians(angles) 
    steps = np.arange(ray_length) 
    
    # 3. Initialize 3D LUT (initialized to max ray length)
    lidar_lut = torch.full((h, w, num_angles), ray_length, dtype=torch.float32, device=device)

    # Precompute unit vectors for all angles
    dx = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device=device)
    dy = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device=device)
    steps_tensor = torch.tensor(steps, dtype=torch.float32, device=device)

    print(f"[INFO] Generating 3D LUT on {device}: {h}x{w} map | {num_angles} angles | {ray_length}px range")
    
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
            ray_x = (x + dx[:, None] * steps_tensor[None, :]).long()
            ray_y = (y + dy[:, None] * steps_tensor[None, :]).long()

            # Valid mask for boundaries 
            valid = (ray_x >= 0) & (ray_x < w) & (ray_y >= 0) & (ray_y < h) 
            
            # Clip for safe indexing 
            rx_c = torch.clamp(ray_x, 0, w - 1)
            ry_c = torch.clamp(ray_y, 0, h - 1)
            
            # Hit detection 
            hits = (obstacle_map_tensor[ry_c, rx_c] == 1) & valid 
            
            # Vectorized search for first hit index per angle
            # argmax returns the first index of 'True' 
            has_hit = torch.any(hits, dim=1) 
            first_hits = torch.argmax(hits.float(), dim=1) 
            
            # Assign distances: if no hit, it remains the default ray_length
            lidar_lut[y, x, has_hit] = first_hits[has_hit].float()

    # 5. Save as numpy for compatibility
    lidar_lut_np = lidar_lut.cpu().numpy()
    base_name = os.path.splitext(os.path.basename(image_path))[0] 
    output_path = os.path.join(output_dir, f"{base_name}_{num_angles}.npy")
    np.save(output_path, lidar_lut_np)
    print(f"\n[SUCCESS] Saved 3D LUT to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    generate_3d_lidar_lut("environments/images/6.png", num_angles=360, device=args.device)