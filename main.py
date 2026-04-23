# main.py - Updated with device argument
import argparse
import os
import sys
import time
import numpy as np
from datetime import datetime
import json
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
                  action_distribution="deterministic",
                  parallel_trials=False,
                  device='cuda'):
    """Load strategy class based on name."""
    # Normalize strategy names
    if name == "spiking":
        name = "nn_spiking"
    elif name == "random_nn":
        name = "nn_random"
    
    if name == "random":
        from strategies.random import RandomWalkStrategy
        return RandomWalkStrategy(num_trials=num_trials, max_generations=num_generations, parallel_trials=parallel_trials)

    if name == "levy":
        from strategies.levy import LevyWalkStrategy
        return LevyWalkStrategy(alpha=1.6, min_step=1.0, max_step=200.0, 
                                num_trials=num_trials, max_generations=num_generations, parallel_trials=parallel_trials)

    if name == "levy_custom":
        from strategies.levy import LevyWalkStrategy
        return LevyWalkStrategy(alpha=alpha, min_step=min_step, max_step=max_step,
                                num_trials=num_trials, max_generations=num_generations, parallel_trials=parallel_trials)

    if name == "manual":
        from strategies.manual import ManualControlStrategy
        return ManualControlStrategy()
    
    if name == "uniform":
        from strategies.uniform import UniformRunLengthStrategy
        return UniformRunLengthStrategy(min_step=1, max_step=10,
                                        num_trials=num_trials, max_generations=num_generations, parallel_trials=parallel_trials)

    if name == "nn_spiking":
        from strategies.nn import NNStrategy
        return NNStrategy(
            population_size=population_size,
            generations=num_generations,
            num_trials=num_trials,
            mutation_rate=mutation_rate,
            mutation_mag=0.5,
            network_type="spiking",
            action_space=action_space,  
            action_distribution=action_distribution, 
            strategy_name="nn_spiking",
            parallel_trials=parallel_trials,
            device=device
        )

    if name == "nn_random":
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
            action_distribution=action_distribution, 
            strategy_name="nn_random",
            parallel_trials=parallel_trials,
            device=device
        )

    raise ValueError(f"Unknown strategy: {name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Vectorized Robot Exploration")
    parser.add_argument("--strategy", type=str, default="random", 
                        choices=["random", "levy", "levy_custom", "uniform", "manual", "spiking", "random_nn"],
                        help="Exploration strategy")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--population", type=int, default=10, help="Number of parallel robots / population size")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--env", type=str, default="6.png")
    parser.add_argument(
        "--no_lut", 
        dest="use_lut", 
        action="store_false", 
        default=True,
        help="Disable the pre-calculated Look-Up Table and use ray marching instead (default: LUT is enabled)"
    )
    parser.add_argument(    
        "--continue_after_goal",
        action="store_true",
        help="Continue simulation even after reaching the reward until max_steps"
    )

    # New Evolutionary/Trial Arguments
    parser.add_argument("--trials", type=int, default=1, help="Number of independent trials") 
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
        default=0, 
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

        
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base output directory (default: './output')")
    
    parser.add_argument("--parallel", action="store_true",
                        help="Run trials in parallel using multiprocessing")
    
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run on (cuda or cpu)")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"Using device: {args.device}")
    
    # Manual control override
    if args.strategy == "manual":
        args.population = 1  # Force single robot for manual
        args.render = True
        print("Manual control mode - using 1 environment")
    
    # Create overall output directory
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    base_output_dir = args.output_dir if args.output_dir is not None else "output"
    run_dir = os.path.join(base_output_dir, f"{timestamp}_{args.strategy}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save top-level metadata
    overall_metadata = {
        "run_datetime": datetime.now().isoformat(),
        "strategy": args.strategy,
        "num_trials": args.trials,
        "generations_per_trial": args.generations,
        "population_size": args.population,
        "max_steps": args.max_steps,
        "map": args.env,
        "use_lut": args.use_lut,
        "continue_after_goal": args.continue_after_goal,
        "parallel_trials": args.parallel,
        "device": args.device,
        "command_line_args": vars(args)
    }
    with open(os.path.join(run_dir, "metadata.json"), 'w') as f:
        json.dump(overall_metadata, f, indent=4)
    
    # Environment parameters (picklable)
    env_params = {
        "map_image_path": get_map_path(args.env),
        "num_envs": args.population,
        "robot_radius": 3,
        "render": args.render,
        "max_steps": args.max_steps,
        "strategy_name": args.strategy,
        "use_lut": args.use_lut,
        "continue_after_goal": args.continue_after_goal,
        "verbose": args.verbose,
        "device": args.device
        # Note: output_dir will be added per trial inside worker
    }
    
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
        action_distribution=args.action_distribution,
        parallel_trials=args.parallel,
        device=args.device
    )
    
    # Run
    strategy.run(env_params, run_dir)

if __name__ == "__main__":
    import torch  # Import here to avoid circular imports
    main()

# example commands
# --- RANDOM WALK --- 
# python main.py --strategy random --max_steps 1000 --population 10 --env 6.png
# python main.py --strategy random --env 6.png --max_steps 100000 --continue_after_goal --population 10
# --- SPIKING --- 
# python main.py --strategy spiking --trials 50 --generations 50 --population 100 --max_steps 1000 --env 6.png
#
# --- PROFILING ---
# python -m cProfile -s tottime main.py --strategy random --env 6.png --max_steps 100_000 --continue_after_goal --population 10 > xprofile.txt
# 
# --- PARALLEL TRIALS ---
# python main.py --strategy spiking --trials 50 --generations 50 --population 100 --max_steps 1000 --env 6.png
# 