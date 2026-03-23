import argparse
import os
import sys
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import RobotExplorationEnv

# Base path for environment images
IMAGES_DIR = "environments/images"

def get_map_path(env_filename):
    """Return full path to environment image."""
    return os.path.join(IMAGES_DIR, env_filename)


def load_strategy(name, alpha=None, min_step=None, max_step=None):
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
        from strategies import SpikeNNGeneticStrategy
        return SpikeNNGeneticStrategy(
            population_size=50,
            generations=20,
            energy_per_agent=150,
            elite_count=10,
            mutation_rate=0.3,
            weights_dir="spike_weights"
        )


    raise ValueError(f"Unknown strategy: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Robot Exploration Runner")

    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "levy", "manual", "levy_custom", "uniform", "spike_nn"],
        help="Exploration strategy"
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Max number of steps to run (default 1000)"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering"
    )

    # Custom Lévy walk parameters
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--min_step", type=float, default=1.0)
    parser.add_argument("--max_step_len", type=float, default=200.0)

    # New argument: choose environment
    parser.add_argument(
        "--env",
        type=str,
        default="6.png",
        help="Environment image filename (from environments/images)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.strategy == "manual":
        args.render = True

    strategy = load_strategy(
        args.strategy,
        alpha=args.alpha,
        min_step=args.min_step,
        max_step=args.max_step_len
    )

    env = RobotExplorationEnv(
        map_image_path=get_map_path(args.env),
        robot_radius=3,
        render=args.render,
        max_steps=args.max_steps,
        strategy_name=strategy.name,
        strategy_parameters=strategy.parameters
    )

    print(f"\nRunning {strategy.name} on {args.env}...")
    print(f"Parameters: {strategy.parameters}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output dir: {env.output_dir}")

    start_time = time.time()

    steps, coverage = strategy.run(env)

    elapsed = time.time() - start_time
    time_per_step = elapsed / steps if steps > 0 else 0
    steps_per_second = steps / elapsed if elapsed > 0 else 0

    print("\n=== Simulation Complete ===")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Steps taken: {steps}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Time per step: {time_per_step*1000:.2f}ms")
    print(f"Steps per second: {steps_per_second:.2f}")
    print(f"Saved to: {env.output_dir}")

    env.close()


if __name__ == "__main__":
    main()

    # example command: python main.py --strategy random --max_steps 1000 --env 6.png
