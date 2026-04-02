import argparse
import os
import sys
import time
import numpy as np
from vec_env import VectorRobotExplorationEnv

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
        from strategies.random_walk import RandomWalkStrategy
        return RandomWalkStrategy()

    if name == "levy":
        from strategies.levy_walk import LevyWalkStrategy
        return LevyWalkStrategy(alpha=1.6, min_step=1.0, max_step=200.0)

    if name == "levy_custom":
        from strategies.levy_walk import LevyWalkStrategy
        return LevyWalkStrategy(alpha=alpha, min_step=min_step, max_step=max_step)

    if name == "manual":
        from strategies.manual_control import ManualControlStrategy
        return ManualControlStrategy()
    if name == "uniform":
        from strategies.uniform import UniformRunLengthStrategy
        return UniformRunLengthStrategy(min_step=1, max_step=10)

    if name == "spike_nn":
        from strategies.spiking import SpikeNNGeneticStrategy
        return SpikeNNGeneticStrategy(
            population_size=population_size,
            generations=num_generations,
            num_trials=num_trials,
            mutation_rate=0.3,
            weights_dir="spike_weights"
        )
    if name == "ga":
        from strategies.genetic_algorithm import RNNGeneticStrategy
        return RNNGeneticStrategy(
            num_trials=num_trials,
            generations=num_generations,
            population_size=population_size, 
            mutation_rate=mutation_rate,
            load_weights=load_weights,
            weights_dir=weights_dir
        )
    
    if name == "snn":
        from strategies.vector_snn import VectorSpikeNNStrategy
        return VectorSpikeNNStrategy(
            population_size=population_size,
            generations=num_generations,
            num_trials=num_trials,
            mutation_rate=0.2,
            mutation_mag=0.5
        )

    raise ValueError(f"Unknown strategy: {name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Vectorized Robot Exploration")
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "ga", "spike_nn", "snn"])
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

    # Load strategy with user-defined evolutionary parameters
    strategy = load_strategy(
        args.strategy,
        population_size=args.population,
        num_generations=args.generations,
        num_trials=args.trials
    )

    # Initialize Environment with correct num_envs
    env = VectorRobotExplorationEnv(
        map_image_path=get_map_path(args.env),
        num_envs=args.population,
        robot_radius=3,
        render=args.render,
        max_steps=args.max_steps,
        strategy_name=strategy.name,
        use_lut=args.use_lut,
        continue_after_goal=args.continue_after_goal,
        verbose=args.verbose
    )

    print(f"Running vectorized {args.strategy} on {args.env}...")
    print(f"Output directory: {env.output_dir}")
    
    # Clean execution call
    strategy.run(env)
    
    env.close()
    
if __name__ == "__main__":
    main()