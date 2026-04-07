import argparse
import os
import sys
import time
import numpy as np
from env import VectorRobotExplorationEnv

IMAGES_DIR = "environments/images"

def get_map_path(env_filename):
    return os.path.join(IMAGES_DIR, env_filename)

def load_strategy(name, 
                  alpha=None, 
                  min_step=None, 
                  max_step=None, 
                  num_trials=3, 
                  num_generations=30, 
                  population_size=100, 
                  mutation_rate=0.1,
                  load_weights=False,
                  weights_dir="weights"
                  ):
    """Load strategy class based on name."""
    if name == "random":
        from strategies.random import RandomWalkStrategy
        return RandomWalkStrategy()

    if name == "levy":
        from strategies.levy import LevyWalkStrategy
        return LevyWalkStrategy(alpha=1.6, min_step=1.0, max_step=200.0)

    if name == "levy_custom":
        from strategies.levy import LevyWalkStrategy
        return LevyWalkStrategy(alpha=alpha, min_step=min_step, max_step=max_step)

    if name == "manual":
        from strategies.manual import ManualControlStrategy
        return ManualControlStrategy()
    
    if name == "uniform":
        from strategies.uniform import UniformRunLengthStrategy
        return UniformRunLengthStrategy(min_step=1, max_step=10)

    if name == "spiking":
        from strategies.nn import NNStrategy
        return NNStrategy(
            population_size=population_size,
            generations=num_generations,
            num_trials=num_trials,
            mutation_rate=mutation_rate,
            mutation_mag=0.5,
            network_type="spiking"
        )
    
    if name == "random_nn":
        from strategies.nn import NNStrategy
        return NNStrategy(
            population_size=population_size,
            generations=num_generations,
            num_trials=num_trials,
            mutation_rate=mutation_rate,
            mutation_mag=0.5,
            network_type="feedforward"
        )

    raise ValueError(f"Unknown strategy: {name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Vectorized Robot Exploration")
    parser.add_argument("--strategy", type=str, default="random", 
                        choices=["random", "levy", "levy_custom", "uniform", "manual", "spiking", "random_nn"],
                        help="Exploration strategy")
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

    # New Evolutionary/Trial Arguments
    parser.add_argument("--trials", type=int, default=20, help="Number of independent trials") 
    parser.add_argument("--generations", type=int, default=50, help="Generations per trial") 
    parser.add_argument("--population", type=int, default=1000, help="Individuals per generation") 


    # Custom Lévy walk parameters
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--min_step", type=float, default=1.0)
    parser.add_argument("--max_step_len", type=float, default=200.0)
    parser.add_argument("--load_weights", type=str, default=None, help="Path to weights")
    parser.add_argument("--weights_dir", type=str, default="ga_weights", help="Directory to save/load weights")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate for genetic algorithm")

    parser.add_argument("--verbose", action="store_true", help="Verbose output during trials")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Manual control override
    if args.strategy == "manual":
        args.num_envs = 1
        args.render = True  # Force render for manual
        print("Manual control mode - using 1 environment")
    
    # Load strategy
    strategy = load_strategy(
        args.strategy,
        population_size=args.population,
        num_generations=args.generations,
        num_trials=args.trials,
        mutation_rate=args.mutation_rate,
        alpha=args.alpha,
        min_step=args.min_step,
        max_step=args.max_step_len
    )
    
    # Initialize Environment
    env = VectorRobotExplorationEnv(
        map_image_path=get_map_path(args.env),
        num_envs=args.num_envs,
        robot_radius=3,
        render=args.render,
        max_steps=args.max_steps,
        strategy_name=strategy.name,
        use_lut=args.use_lut,
        continue_after_goal=args.continue_after_goal,
        verbose=args.verbose
    )
    
    # Run
    strategy.run(env)
    env.close()

if __name__ == "__main__":
    main()

    # example commands
    # --- RANDOM WALK --- 
    # python main.py --strategy random --max_steps 1000 --env 6.png
    # python main.py --strategy random --env 6.png --use_lut --max_steps 100000 --continue_after_goal
    # --- SPIKING --- 
    # python main.py --strategy spike_nn --trials 50 --generations 50 --population 1000 --max_steps 1000 --env 6.png
    # --- GA --- 
    # python main.py --strategy ga --trials 5 --generations 5 --population 10 --max_steps 1000 --env 6.png
    # --- PROFILING ---
    # python -m cProfile -s tottime main.py --strategy random --env 6.png --use_lut --max_steps 100_000 --continue_after_goal > xprofile.txt