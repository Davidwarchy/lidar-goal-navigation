import argparse
import os
import sys
import time
import numpy as np
from vec_env import VectorRobotExplorationEnv

IMAGES_DIR = "environments/images"

def get_map_path(env_filename):
    return os.path.join(IMAGES_DIR, env_filename)

def parse_args():
    parser = argparse.ArgumentParser(description="Vectorized Robot Exploration")
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "ga", "spike_nn"])
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=10, help="Number of parallel robots")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--env", type=str, default="6.png")
    parser.add_argument("--use_lut", action="store_true")
    parser.add_argument(    
        "--continue_after_goal",
        action="store_true",
        help="Continue simulation even after reaching the reward until max_steps"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize Vectorized Env
    env = VectorRobotExplorationEnv(
        map_image_path=get_map_path(args.env),
        num_envs=args.num_envs,
        render=args.render,
        max_steps=args.max_steps,
        use_lut=args.use_lut,
        strategy_name=args.strategy,
        continue_after_goal=args.continue_after_goal
    )

    print(f"Running vectorized {args.strategy} with {args.num_envs} robots...")
    
    env.reset()
    start_time = time.time()
    total_steps = 0

    try:
        step = 0
        while True:
            # Simple random actions for demo; replace with strategy.get_actions(obs)
            actions = np.random.randint(0, 4, size=args.num_envs)
            
            obs, rewards, dones, info = env.step(actions)
            total_steps += args.num_envs
            step += 1
            
            # 👇 print number of completed envs at this step
            num_dones = np.sum(dones)

            print(f"step {step}: {num_dones} dones")
            
            if env.render_flag:
                env.render()
            
            if np.all(dones):
                break
                
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start_time
    print(f"\n=== Complete ===")
    print(f"Total Steps (All Envs): {total_steps}")
    print(f"Steps Per Second: {total_steps / elapsed:.2f}")
    
    env.close()

if __name__ == "__main__":
    main()