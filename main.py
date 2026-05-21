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
                  mutation_mag=0.5,
                  recombination=False,
                  lif_steps=5, 
                  load_weights=False,
                  weights_dir="weights",
                  hidden_sizes=[64],
                  ga_curriculum_enabled=False,
                  ga_curriculum_success_threshold=0.05,
                  ga_curriculum_consecutive_gens=3,
                  ga_curriculum_distance_increment=5.0, 
                  action_space="discrete",
                  action_distribution="deterministic",
                  parallel_trials=False,
                  device='cuda',
                  fitness_proxy="none",
                  fitness_pool="survivors_only"):
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

    # Parse hidden_sizes from command line argument
    hidden_sizes_str = hidden_sizes
    hidden_sizes_list = [int(x.strip()) for x in hidden_sizes_str.split(",")] if hidden_sizes_str else [64]
    network_params = {"hidden_sizes": hidden_sizes_list}

    if name == "nn_spiking":
        from strategies.nn import NNStrategy
        hidden_sizes_str = hidden_sizes
        hidden_sizes_list = [int(x.strip()) for x in hidden_sizes_str.split(",")] if hidden_sizes_str else [64]
        network_params = {"hidden_sizes": hidden_sizes_list}
        return NNStrategy(
            population_size=population_size,
            generations=num_generations,
            num_trials=num_trials,
            mutation_rate=mutation_rate,
            mutation_mag=mutation_mag,
            recombination_enabled=recombination,
            n_steps=lif_steps,
            network_type="spiking",
            curriculum_enabled=ga_curriculum_enabled,
            curriculum_success_threshold=ga_curriculum_success_threshold,
            curriculum_consecutive_gens=ga_curriculum_consecutive_gens,
            curriculum_distance_increment=ga_curriculum_distance_increment,
            action_space=action_space,
            action_distribution=action_distribution,
            strategy_name="nn_spiking",
            parallel_trials=parallel_trials,
            device=device,
            fitness_proxy=fitness_proxy,
            fitness_pool=fitness_pool,
            **network_params
        )

    if name == "nn_random":
        from strategies.nn import NNStrategy
        hidden_sizes_str = hidden_sizes
        hidden_sizes_list = [int(x.strip()) for x in hidden_sizes_str.split(",")] if hidden_sizes_str else [64]
        network_params = {"hidden_sizes": hidden_sizes_list}
        return NNStrategy(
            population_size=population_size,
            generations=num_generations,
            num_trials=num_trials,
            mutation_rate=mutation_rate,
            mutation_mag=mutation_mag,
            recombination_enabled=recombination,
            network_type="feedforward",
            curriculum_enabled=ga_curriculum_enabled,
            curriculum_success_threshold=ga_curriculum_success_threshold,
            curriculum_consecutive_gens=ga_curriculum_consecutive_gens,
            curriculum_distance_increment=ga_curriculum_distance_increment,
            action_space=action_space,
            action_distribution=action_distribution,
            strategy_name="nn_random",
            parallel_trials=parallel_trials,
            device=device,
            fitness_proxy=fitness_proxy,
            fitness_pool=fitness_pool,
            **network_params
        )
    raise ValueError(f"Unknown strategy: {name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Vectorized Robot Exploration")
    parser.add_argument("--strategy", type=str, default="random", 
                        choices=["random", "levy", "levy_custom", "uniform", "manual", "spiking", "random_nn", "nn_spiking", "nn_random"],
                        help="Exploration strategy")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--population", type=int, default=10, help="Number of parallel robots / population size")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--env", type=str, default="6.png")
    parser.add_argument("--no_lut", dest="use_lut", action="store_false", default=True)
    parser.add_argument("--continue_after_goal", action="store_true")
    
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--generations", type=int, default=1)
    
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--min_step", type=float, default=1.0)
    parser.add_argument("--max_step_len", type=float, default=200.0)
    parser.add_argument("--load_weights", type=str, default=None)
    parser.add_argument("--weights_dir", type=str, default="ga_weights")
    parser.add_argument("--mutation_rate", type=float, default=0.1)
    parser.add_argument("--ga_curriculum", action="store_true")
    parser.add_argument("--ga_curriculum_success_threshold", type=float, default=0.05)
    parser.add_argument("--ga_curriculum_consecutive_gens", type=int, default=3)
    parser.add_argument("--ga_curriculum_distance_increment", type=float, default=5.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_top_k", type=int, default=0)
    parser.add_argument("--action_space", type=str, default="discrete", choices=["discrete", "continuous"])
    parser.add_argument("--action_distribution", type=str, default="deterministic", choices=["deterministic", "stochastic"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mutation_mag", type=float, default=0.5)
    parser.add_argument("--goal_spawn_dist", type=float, default=30.0)
    parser.add_argument("--recombination", action="store_true")
    parser.add_argument("--lif_steps", type=int, default=5)
    parser.add_argument("--robot_radius", type=int, default=3)
    parser.add_argument("--fitness_proxy", type=str, default="none", choices=["none", "health", "energy"])
    parser.add_argument("--fitness_pool", type=str, default="survivors_only", choices=["all_agents", "survivors_only"])
    
    # --- Noise Arguments ---
    parser.add_argument("--lidar_noise", type=str, default="none",
                        choices=["none", "gaussian", "uniform", "dropout"],
                        help="LiDAR noise type")
    parser.add_argument("--lidar_noise_std", type=float, default=5.0,
                        help="Gaussian noise standard deviation (pixels)")
    parser.add_argument("--lidar_noise_max", type=float, default=10.0,
                        help="Uniform noise max offset (pixels)")
    parser.add_argument("--lidar_dropout_p", type=float, default=0.05,
                        help="Ray dropout probability")
    parser.add_argument("--motor_noise", type=str, default="none",
                        choices=["none", "slip", "deadzone"],
                        help="Motor noise type")
    parser.add_argument("--motor_slip_prob", type=float, default=0.1,
                        help="Wheel slip probability")
    parser.add_argument("--motor_slip_mag", type=float, default=0.5,
                        help="Wheel slip magnitude (0-1)")
    parser.add_argument("--motor_deadzone", type=float, default=0.1,
                        help="Motor deadzone threshold")
    parser.add_argument("--obs_delay", type=int, default=0,
                        help="Observation delay in steps (latency)")
    
    # Additional argument for neural network hidden layer sizes
    parser.add_argument("--hidden_sizes", type=str, default="64",
                    help="Comma-separated hidden layer sizes (e.g., '64' or '64,32'). "
                         "For spiking networks, each value is a LIF layer.")
    
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
        "mutation_rate": args.mutation_rate,
        "mutation_magnitude": args.mutation_mag,
        "action_space": args.action_space,
        "action_distribution": args.action_distribution,
        # Noise metadata
        "lidar_noise_type": args.lidar_noise,
        "lidar_noise_std": args.lidar_noise_std,
        "lidar_noise_max": args.lidar_noise_max,
        "lidar_dropout_p": args.lidar_dropout_p,
        "motor_noise_type": args.motor_noise,
        "motor_slip_prob": args.motor_slip_prob,
        "motor_slip_mag": args.motor_slip_mag,
        "motor_deadzone_threshold": args.motor_deadzone,
        "observation_delay": args.obs_delay,
        "hidden_sizes": args.hidden_sizes, 
        "command_line_args": vars(args)
    }
    with open(os.path.join(run_dir, "metadata.json"), 'w') as f:
        json.dump(overall_metadata, f, indent=4)
    
    # Environment parameters (picklable)
    env_params = {
        "map_image_path": get_map_path(args.env),
        "num_envs": args.population,
        "robot_radius": args.robot_radius,
        "render": args.render,
        "max_steps": args.max_steps,
        "strategy_name": args.strategy,
        "use_lut": args.use_lut,
        "continue_after_goal": args.continue_after_goal,
        "verbose": args.verbose,
        "device": args.device,
        "goal_spawn_dist": args.goal_spawn_dist,
        # Noise parameters
        "lidar_noise_type": args.lidar_noise,
        "lidar_noise_std": args.lidar_noise_std,
        "lidar_noise_max": args.lidar_noise_max,
        "lidar_dropout_p": args.lidar_dropout_p,
        "motor_noise_type": args.motor_noise,
        "motor_slip_prob": args.motor_slip_prob,
        "motor_slip_mag": args.motor_slip_mag,
        "motor_deadzone_threshold": args.motor_deadzone,
        "observation_delay": args.obs_delay,
    }
    
    strategy = load_strategy(
        args.strategy,
        population_size=args.population,
        num_generations=args.generations,
        num_trials=args.trials,
        mutation_rate=args.mutation_rate,
        mutation_mag=args.mutation_mag,
        recombination=args.recombination,
        lif_steps=args.lif_steps,
        hidden_sizes=args.hidden_sizes,
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
        device=args.device,
        fitness_proxy=args.fitness_proxy,
        fitness_pool=args.fitness_pool
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
# python main.py --strategy spiking --trials 1 --generations 10 --population 1000 --max_steps 1000 --env 6.png
#
# --- PROFILING ---
# python -m cProfile -s tottime main.py --strategy random --env 6.png --max_steps 100_000 --continue_after_goal --population 10 > xprofile.txt
# 
# --- PARALLEL TRIALS ---
# python main.py --strategy spiking --trials 50 --generations 50 --population 100 --max_steps 1000 --env 6.png
# 
# python main.py --strategy spiking --trials 1 --generations 2 --population 100 
# 
# --- CURRICULUM LOGIC ---
# python main.py --strategy random_nn --trials 1 --generations 2 --population 100 --max_steps 1000 --env 6.png --ga_curriculum
# 
# --- CUSTOM LEVY WALK ---
# python main.py --strategy random_nn --trials 1 --generations 2 --population 1000 --max_steps 1000 --env 6.png --fitness_proxy none
# 
# --- ACTION SPACE / DISTRIBUTION --- 
# python main.py --strategy nn_random --trials 1 --generations 2 --population 10 --max_steps 1000 --env 6.png --action_space continuous --action_distribution stochastic