"""
Mass experiment runner for all five strategies.
Runs each strategy with:
- 10 trials
- 100 generations  
- 1000 population size
- LUT enabled
"""

import subprocess
import sys
import os
from datetime import datetime

# Configuration
STRATEGIES = [
    "random",
    "levy", 
    "uniform",
    "random_nn",
    "spiking"
]

TRIALS = 10
GENERATIONS = 100
POPULATION = 1000
MAX_STEPS = 1000
MAP = "6.png"
USE_LUT = True
CONTINUE_AFTER_GOAL = False  # Set to True if you want robots to continue after reaching goal

def get_command(strategy):
    """Build command for each strategy with specific parameters"""
    
    base_cmd = [
        sys.executable, "main.py",
        "--strategy", strategy,
        "--trials", str(TRIALS),
        "--generations", str(GENERATIONS),
        "--population", str(POPULATION),
        "--env", MAP,
        "--max_steps", str(MAX_STEPS),
    ]
    
    # Add LUT if enabled
    if USE_LUT:
        base_cmd.append("--use_lut")
    
    # Add continue_after_goal if enabled
    if CONTINUE_AFTER_GOAL:
        base_cmd.append("--continue_after_goal")
    
    # Strategy-specific parameters
    if strategy == "levy":
        base_cmd.extend(["--alpha", "1.6", "--min_step", "1.0", "--max_step_len", "200.0"])
    
    elif strategy == "random_nn":
        base_cmd.extend([
            "--mutation_rate", "0.1",
            "--save_top_k", "5"
        ])
    
    elif strategy == "spiking":
        base_cmd.extend([
            "--mutation_rate", "0.1",
            "--save_top_k", "5"
        ])
    
    return base_cmd

def run_experiments():
    """Run all strategies sequentially"""
    
    # Create timestamp for this mass run
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    base_output_dir = f"output/mass-run-{timestamp}"
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Log file for this mass run
    log_path = os.path.join(base_output_dir, "run_log.txt")
    
    results = {}
    
    with open(log_path, 'w') as log:
        log.write(f"Mass Experiment Run: {timestamp}\n")
        log.write(f"Date: {datetime.now().isoformat()}\n")
        log.write(f"Trials per strategy: {TRIALS}\n")
        log.write(f"Generations per trial: {GENERATIONS}\n")
        log.write(f"Population size: {POPULATION}\n")
        log.write(f"Map: {MAP}\n")
        log.write(f"Use LUT: {USE_LUT}\n")
        log.write(f"Continue after goal: {CONTINUE_AFTER_GOAL}\n")
        log.write("="*80 + "\n\n")
        
        print(f"\n{'='*80}")
        print(f"MASS EXPERIMENT RUN - {timestamp}")
        print(f"Trials: {TRIALS}, Generations: {GENERATIONS}, Population: {POPULATION}")
        print(f"Output directory: {base_output_dir}")
        print(f"{'='*80}\n")
        
        for i, strategy in enumerate(STRATEGIES, 1):
            print(f"\n[{i}/{len(STRATEGIES)}] Running strategy: {strategy.upper()}")
            print("-"*50)
            
            # Set output directory for this strategy
            strategy_output = os.path.join(base_output_dir, strategy)
            cmd = get_command(strategy)
            cmd.extend(["--output_dir", strategy_output])
            
            # Log command
            cmd_str = " ".join(cmd)
            log.write(f"\n{'='*60}\n")
            log.write(f"Strategy: {strategy}\n")
            log.write(f"Command: {cmd_str}\n")
            log.write(f"Start time: {datetime.now().isoformat()}\n")
            log.flush()
            
            print(f"Command: {cmd_str}")
            print(f"Output: {strategy_output}")
            
            try:
                # Run the command
                start_time = datetime.now()
                result = subprocess.run(cmd, capture_output=True, text=True)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Log results
                log.write(f"End time: {end_time.isoformat()}\n")
                log.write(f"Duration: {duration:.2f} seconds\n")
                log.write(f"Return code: {result.returncode}\n")
                
                if result.returncode == 0:
                    log.write("Status: SUCCESS\n")
                    print(f"✓ SUCCESS - Duration: {duration:.2f}s")
                else:
                    log.write("Status: FAILED\n")
                    log.write(f"STDERR:\n{result.stderr}\n")
                    print(f"✗ FAILED - Duration: {duration:.2f}s")
                    print(f"  Error: {result.stderr[:200]}...")
                
                results[strategy] = {
                    "status": "success" if result.returncode == 0 else "failed",
                    "duration": duration,
                    "output_dir": strategy_output
                }
                
            except Exception as e:
                log.write(f"EXCEPTION: {str(e)}\n")
                print(f"✗ EXCEPTION: {e}")
                results[strategy] = {
                    "status": "exception",
                    "error": str(e),
                    "output_dir": strategy_output
                }
            
            log.write("\n")
            log.flush()
            
            print(f"{'-'*50}\n")
        
        # Write summary
        log.write("\n" + "="*80 + "\n")
        log.write("FINAL SUMMARY\n")
        log.write("="*80 + "\n")
        
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        
        for strategy, result in results.items():
            status = result["status"]
            duration = result.get("duration", 0)
            output_dir = result["output_dir"]
            log.write(f"\n{strategy.upper()}:\n")
            log.write(f"  Status: {status}\n")
            log.write(f"  Duration: {duration:.2f}s\n")
            log.write(f"  Output: {output_dir}\n")
            
            # Print to console with color indication
            status_icon = "✓" if status == "success" else "✗"
            print(f"{status_icon} {strategy.upper():12} - {status:10} - {duration:8.2f}s - {output_dir}")
        
        log.write(f"\nLog saved to: {log_path}\n")
    
    print(f"\nLog saved to: {log_path}")
    print(f"All outputs in: {base_output_dir}")
    
    return results

def run_single_strategy(strategy, output_dir=None):
    """Run a single strategy with the same parameters"""
    
    cmd = get_command(strategy)
    
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mass experiment runner for robot exploration strategies")
    parser.add_argument("--strategy", type=str, choices=STRATEGIES + ["all"], default="all",
                       help="Strategy to run (default: all)")
    parser.add_argument("--trials", type=int, default=TRIALS,
                       help=f"Number of trials (default: {TRIALS})")
    parser.add_argument("--generations", type=int, default=GENERATIONS,
                       help=f"Generations per trial (default: {GENERATIONS})")
    parser.add_argument("--population", type=int, default=POPULATION,
                       help=f"Population size (default: {POPULATION})")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    
    args = parser.parse_args()
    
    # Override defaults if provided
    TRIALS = args.trials
    GENERATIONS = args.generations
    POPULATION = args.population
    
    if args.dry_run:
        print("DRY RUN - Commands to be executed:")
        print("="*60)
        if args.strategy == "all":
            for strategy in STRATEGIES:
                cmd = get_command(strategy)
                print(f"\n{strategy}:")
                print(f"  {' '.join(cmd)}")
        else:
            cmd = get_command(args.strategy)
            print(f"{' '.join(cmd)}")
        sys.exit(0)
    
    if args.strategy == "all":
        run_experiments()
    else:
        # Run single strategy
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        output_dir = f"output/single-{args.strategy}-{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Running {args.strategy} with {TRIALS} trials, {GENERATIONS} generations, {POPULATION} population")
        print(f"Output: {output_dir}")
        
        cmd = get_command(args.strategy)
        cmd.extend(["--output_dir", output_dir])
        
        result = subprocess.run(cmd)
        sys.exit(result.returncode)