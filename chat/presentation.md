# Lidar Navigation 
## Life on Silicon 

Transfer life to silicon 

What's needed: perception, motion, manipulation, selection

Long-term autonomous systems. A good yardstick would be: would the systems we create survive without people? That would need to consider things like how do robots make more of themselves, how do they get to know. 

Natural selection provides a good platform for doing this. 

## Experimental Setup 

- Have a 2d environment 
- Goal is to reach a reward 

Approaches that tend to optimize for a given parameter tend to perform poorly at scale. 

Our main hypothesis is that we can use neural networks to learn everything that we need to have, without optimizing an explicit fitness function. 

- survival as the sole selector
- no hand-crafted fitness function, no explicit reward for coverage/exploration/distance/etc.

## Fitness 

We have created an environment which runs and gives us feedback on whether the reward has been reached or not. 

## 
We shall have 
- runs
- generations 
- individuals 

Each run will have a few generations. Run ends when we don't have living members for the next generation. 

## Subgoal 

What we want to have is to setup our experiments with constraints. 

Let's not undermine the integrity of the experiment (by introducing handcrafted fitness function). 

We will set up our experiment and then check the results. 

We don't want to prematurely optimize for results. We've already hypothesized that a neural network subjected to natural selection will give us the result. 

So we want to take care to setup the experiments right. 

# Meta 
## Github 
tips for being a good manager of a github repo with collaborators 
- we have the goal of creating a object goal navigation system...

branches are getting too wild... with too many features that haven't been fixed? 

I've gotten some advice: 
- No direct push to main (You can automate this so that there's no need for manual enforcement). 
    main → always stable, runnable
    dev → integration branch (optional but useful for your scale)
    feature/* → new work
    fix/* → bug fixes
- Have other branches 
- Kill long-lived branches. Long-lived branches accumulate merge conflicts and lose context.
- Update main via pull requests only from other branches 
- Regular merges in development branch (dev/, feature/*) 

I want to know how to pace developments, features, etc without the whole project spiralling out of control. I think that I want to define a clear endpoint. 

## Clarity 
Problem of clarity 

I'm doing a life on silicon project with a few people. 

I think that there's a problem with telling people that we don't want to have a fitness function for anything. 

WE just want to have teh system, and surivval is the only metric for fitness. 

I have repeated this very many times and it appears that all my people don't seem to understand this. 

I think that it could be because of a few reasons: 
- That I'm not communicating well
- That the concept is difficult to grasp 
- That prior experience is making people fall back to classic reasoning. 

## Experiments 

Experiments with survival rates. 
--- 
- Pure Random Walk 
- Natural Selection 
- Random Neural Network 
- - High level Control Architecture: Centralized vs Decentralized 
- - Action Space: Continuous vs Discrete steps 
- - Action Distribution: Stochastic policy vs Deterministic policy
- - Neural Network Depth: shallow vs not shallow 
- - Neural Network Breadth: Narrow vs wide 
- - Neural Net Architecture: Simple Feed Forward, Recurring Neural Network
- - Maximum number of steps, maximum generations, individuals per generation
- - Curriculum learning (like Peter suggested) 
- - LUTs vs Ray Marching 
- Spiking Neural Network (same as above) 

All the while, we also want to keep track of the options we have used for each run 
- run date 
- strategy name 
- output direction
- strategy specific information. In the case of genetic algorithm: 

```json
{
    "run_datetime": "2026-03-19T11:50:18.200638",
    "strategy_name": "ga",
    "strategy_parameters": {},
    "max_steps": 10000,
    "environment_parameters": {
        "grid_width": 100,
        "grid_height": 100,
        "robot_radius": 3,
        "num_rays": 100,
        "ray_length": 200,
        "max_steps": 10000,
        "map_image": "6.png",
        "cache_size": 1000,
        "goal_location": {
            "x": 57.36092428883696,
            "y": 61.44588212042851
        }
    },
    "output_directory": "output\\2026-03-19-115018_random_walk",
    "strategy_parameters": {
        "population_size": 100,
        "num_generations": 30,
        "mutation_rate": 0.1,
        "mutation_scale": 0.1,
        "survivor_ratio": 0.2,
        "load_weights": null,
        "neural_net_layers": [100, 50, 10, 2],
        "use_curriculum": "True"
    },
    "params" : {
        "render": "False" , 
        "ray_casting_method": "LUT"
    },
    "timing_parameters": {
        "total_steps": 20123, 
        "total_time": 200188,
        "time_per_step": 0.1
    }
}
``` 

We might want to have an option to give a goal location to a robot. 

# 

Outputs to a folder called `output` 

Output for a run should be timestamped `YYYY-MM-DD-HHMMSS`

We want to save experiment metadata in a json file, so that it's quite easy to refer to the experiments during ablation analysis. 

Run. Generations. Population. Individual. 

# Also some nice experiments to do 
- Have a system that does selection for network connections 
- Have a system that kinda reproduces neurons 

# Naming of Iteration Objects 

One _run_ (main.py) conducts multiple _trials_.

A _trial_ goes like this: we start with one _generation_, then two. If no survivors at any point between moving to the next generation, we stop that _trial_. We call that trial _extinct_, . 

A _generation_ has many _individuals_. A group of _individuals_ in the same _generation_ is called a _population_. 

Each _individual_ runs in the env until done. 
