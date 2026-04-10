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
                  weights_dir="weights",
                  ga_curriculum_enabled=False,
                  ga_curriculum_success_threshold=0.05,
                  ga_curriculum_consecutive_gens=3,
                  ga_curriculum_distance_increment=5.0, 
                  action_space="discrete",
                  action_distribution="deterministic"
                  ):
    """Load strategy class based on name."""
    if name == "random":
        from strategies.random import RandomWalkStrategy
        return RandomWalkStrategy(num_trials=num_trials, max_generations=num_generations)

    if name == "levy":
        from strategies.levy import LevyWalkStrategy
        return LevyWalkStrategy(alpha=1.6, min_step=1.0, max_step=200.0, 
                                num_trials=num_trials, max_generations=num_generations)

    if name == "levy_custom":
        from strategies.levy import LevyWalkStrategy
        return LevyWalkStrategy(alpha=alpha, min_step=min_step, max_step=max_step,
                                num_trials=num_trials, max_generations=num_generations)

    if name == "manual":
        from strategies.manual import ManualControlStrategy
        return ManualControlStrategy()
    
    if name == "uniform":
        from strategies.uniform import UniformRunLengthStrategy
        return UniformRunLengthStrategy(min_step=1, max_step=10,
                                        num_trials=num_trials, max_generations=num_generations)

    if name == "spiking":
        from strategies.nn import NNStrategy
        return NNStrategy(
            population_size=population_size,
            generations=num_generations,
            num_trials=num_trials,
            mutation_rate=mutation_rate,
            mutation_mag=0.5,
            network_type="spiking",
            action_space=action_space,  
            action_distribution=action_distribution 
        )

    if name == "random_nn":
        from strategies.nn import NNStrategy
        return NNStrategy(
            population_size=population_size,
            generations=num_generations,
            num_trials=num_trials,
            mutation_rate=mutation_rate,
            mutation_mag=0.5,
            network_type="feedforward",
            curriculum_enabled=ga_curriculum_enabled,
            curriculum_success_threshold=ga_curriculum_success_threshold,
            curriculum_consecutive_gens=ga_curriculum_consecutive_gens,
            curriculum_distance_increment=ga_curriculum_distance_increment,
            action_space=action_space, 
            action_distribution=action_distribution 
        )

    raise ValueError(f"Unknown strategy: {name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Vectorized Robot Exploration")
    parser.add_argument("--strategy", type=str, default="random", 
                        choices=["random", "levy", "levy_custom", "uniform", "manual", "spiking", "random_nn"],
                        help="Exploration strategy")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--population", type=int, default=10, help="Number of parallel robots / population size")  # Changed default to 10, removed num_envs
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

    # Custom Lévy walk parameters
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--min_step", type=float, default=1.0)
    parser.add_argument("--max_step_len", type=float, default=200.0)
    parser.add_argument("--load_weights", type=str, default=None, help="Path to weights")
    parser.add_argument("--weights_dir", type=str, default="ga_weights", help="Directory to save/load weights")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate for genetic algorithm")
    parser.add_argument(
        "--ga_curriculum",
        action="store_true",
        help="Enable generation-based curriculum for feedforward GA (`random_nn`) only"
    )
    parser.add_argument(
        "--ga_curriculum_success_threshold",
        type=float,
        default=0.05,
        help="Minimum generation success rate to count toward curriculum streak (default 0.05)"
    )
    parser.add_argument(
        "--ga_curriculum_consecutive_gens",
        type=int,
        default=3,
        help="Consecutive qualifying generations needed before increasing goal distance"
    )
    parser.add_argument(
        "--ga_curriculum_distance_increment",
        type=float,
        default=5.0,
        help="Goal distance increase after curriculum promotion"
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose output during trials")

    parser.add_argument(
        "--save_top_k", 
        type=int, 
        default=5, 
        help="Number of top individuals to save per generation (0 = don't save weights)"
    )

    parser.add_argument(
        "--action_space",
        type=str,
        default="discrete",
        choices=["discrete", "continuous"],
        help="Action space type: discrete (4 actions) or continuous (linear/angular velocity)"
    )

    parser.add_argument(
        "--action_distribution",
        type=str,
        default="deterministic",
        choices=["deterministic", "stochastic"],
        help="Action distribution: deterministic (argmax) or stochastic (sample from softmax)"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Manual control override
    if args.strategy == "manual":
        args.population = 1  # Force single robot for manual
        args.render = True
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
        max_step=args.max_step_len,
        ga_curriculum_enabled=args.ga_curriculum,
        ga_curriculum_success_threshold=args.ga_curriculum_success_threshold,
        ga_curriculum_consecutive_gens=args.ga_curriculum_consecutive_gens,
        ga_curriculum_distance_increment=args.ga_curriculum_distance_increment, 
        action_space=args.action_space,
        action_distribution=args.action_distribution 
    )
    
    # Initialize Environment with population size
    env = VectorRobotExplorationEnv(
        map_image_path=get_map_path(args.env),
        num_envs=args.population,  # Use population here
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
# python main.py --strategy random --max_steps 1000 --population 10 --env 6.png
# python main.py --strategy random --env 6.png --use_lut --max_steps 100000 --continue_after_goal --population 10
# --- SPIKING --- 
# python main.py --strategy spiking --trials 50 --generations 50 --population 100 --max_steps 1000 --env 6.png
# --- PROFILING ---
# python -m cProfile -s tottime main.py --strategy random --env 6.png --use_lut --max_steps 100_000 --continue_after_goal --population 10 > xprofile.txt