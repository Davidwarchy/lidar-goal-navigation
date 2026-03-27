import sys
import os

# Add the project root (two levels up from this script) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import os
import json
import numpy as np
from env import RobotExplorationEnv
from strategies.genetic_algorithm import NeuralNetwork

def run_trained_robot(map_path, weight_file_path, render=True, max_steps=1000):
    """
    Loads a specific neural network weight file and runs it in the environment.
    """
    # 1. Initialize the Environment
    env = RobotExplorationEnv(
        map_image_path=map_path,
        render=render,
        max_steps=max_steps,
        robot_radius=3,
        strategy_name="loaded_weights_eval"
    )

    # 2. Load Weights from JSON
    if not os.path.exists(weight_file_path):
        raise FileNotFoundError(f"Weight file not found: {weight_file_path}")

    with open(weight_file_path, 'r') as f:
        data = json.load(f)

    # 3. Reconstruct the Neural Network 
    # Input size matches the number of lidar rays in the environment 
    nn = NeuralNetwork(input_size=env.num_rays, hidden_sizes=[16, 16], output_size=4)
    nn.W1 = np.array(data["W1"])
    nn.b1 = np.array(data["b1"])
    nn.W2 = np.array(data["W2"])
    nn.b2 = np.array(data["b2"])
    nn.W3 = np.array(data["W3"])
    nn.b3 = np.array(data["b3"])

    # 4. Run the Simulation Loop 
    obs = env.reset()
    done = False
    total_reward = 0

    print(f"Starting evaluation of {weight_file_path}...")

    try:
        while not done:
            # Normalize observation before feeding to NN 
            normalized_obs = obs / env.ray_length
            
            # Get the best action based on current weights 
            action = nn.get_action(normalized_obs)
            
            # Step the environment 
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if render:
                env.render()

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
    finally:
        # 5. Clean up and report stats
        print("\n=== Evaluation Complete ===")
        print(f"Steps taken: {env.current_step}")
        print(f"Goal Reached: {info.get('goal_reached', False)}")
        print(f"Final Coverage: {info.get('coverage', 0):.2f}%")
        env.close()

if __name__ == "__main__":
    # Example usage:
    MAP = "environments/images/6.png"
    WEIGHTS = r"C:\Users\HP\Desktop\Projects\navigation\10-lidar-goal-navigation\output\2026-03-23-171230_ga\gen_12_weights\rank_2_fitness_0.1852.json"
    
    run_trained_robot(MAP, WEIGHTS)