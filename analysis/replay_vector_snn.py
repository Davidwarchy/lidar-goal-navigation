"""
Simple replay script - just edit the paths at the bottom and run.
"""

import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import RobotExplorationEnv
from strategies.vector_snn import VectorSNN


def replay_individual(weight_path, map_name="6.png", render=True, max_steps=2000):
    """Simple function to replay a saved individual."""
    
    # Load weights
    with open(weight_path, 'r') as f:
        data = json.load(f)
    
    # Create environment
    map_path = os.path.join("environments", "images", map_name)
    env = RobotExplorationEnv(
        map_image_path=map_path,
        robot_radius=3,
        render=render,
        max_steps=max_steps,
        strategy_name="replay"
    )
    
    # Create SNN
    snn = VectorSNN(num_envs=1, input_size=env.num_rays, hidden_size=64, output_size=4, n_steps=5)
    snn.set_genomes([np.array(data["weights"])])
    
    # Print info
    print(f"\n=== Replaying Individual ===")
    print(f"From: {weight_path}")
    print(f"Original steps to success: {data.get('steps_to_success', 'N/A')}")
    print(f"Initial distance to reward: {data.get('initial_distance_to_reward', 'N/A')}")
    
    # Run
    obs = env.reset()
    snn.reset_state()
    done = False
    step = 0
    
    try:
        while not done and step < max_steps:
            output = snn.forward((obs / env.ray_length).reshape(1, -1))
            action = np.argmax(output[0])
            obs, reward, done, info = env.step(action)
            step += 1
            
            if render:
                env.render()
            
            if info.get("goal_reached", False):
                print(f"\n✓ SUCCESS! Goal reached at step {step}")
                print(f"  Energy: {info['energy']}, Health: {info['health']}")
                break
            
            if step % 500 == 0:
                dist = np.sqrt((env.robot_x - env.goal_x)**2 + (env.robot_y - env.goal_y)**2)
                print(f"  Step {step}: distance = {dist:.1f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        print(f"\nFinal - Steps: {step}, Coverage: {info.get('coverage', 0):.1f}%")
        env.close()


if __name__ == "__main__":
    # EDIT THESE PATHS:
    WEIGHT_PATH = r"C:\Users\HP\Desktop\Projects\navigation\10-lidar-goal-navigation\output\2026-04-02-114646_vec_spike_nn\trial_1\gen_10\samples\ind_78.json"
    MAP_NAME = "6.png"
    
    replay_individual(WEIGHT_PATH, MAP_NAME, render=True)