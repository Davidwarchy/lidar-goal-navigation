# Standard 
We want to have the following standard for the configuration tests 
- _trials_ 7 
- _generations_ 200 (arrived at by checking the generations at which the rate of success improves by less than a certain value in 10 generations)
- _population_ 1000 
- _max\_steps_ 1000 

- standard (random_nn, spiking_nn) - **done on David PC** 
- standard (random, levy, uniform) - **done on OMEN**

# Test Configurations

We are going to test these configurations. 

## Generations (**Done on David PC**)
- We did for about [500 generations](output/experiments/standard/random_nn/2026-04-21-214740_random_nn) and observed that there's little change after 200 generations, so we decided to use this as the base. 

- The folder is in David's local machine, and might not be accessible from here. 

## Time 
Time pressure (max steps) - 100, 500, 1_000, 10_000 (**Done on OMEN**) 

--- 
## Natural Selection 
- Mutation rate - 0.05, 0.1, 0.3, 0.5, 0.75 (**Done on OMEN**)
- Mutation magnitude - 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0 (**Done on OMEN**)
- Recombination - True, False (**Done on Kaggle**)

--- 
## Network architecture
- Network architecture - ? 
- Hidden layer size - ? 
- LIF timesteps - 1, 2, 5, 10 (**1,2,5 Done on OMEN [Kaggle - 1davidwarutumo](https://www.kaggle.com/code/warutumod/lidargoalnavigation-lifsteps10?scriptVersionId=315239134)**) 
- - [This graph](https://claude.ai/share/c71960f9-4361-4ad2-a1fd-64245c60a109) shows that `lif_steps = 5` is the optimal one (since the end result is nearly indiscernible with 10). 

--- 
## Difficulty 
- Goal distance - 10, 20, 30, 40  (**Done on OMEN**)

--- 
## Other factors 

- Robot size - 1, 3, 5 (**David - Done on OMEN**)
- Cirriculum learning - on, off (**David**)
- Population - 100, 1_000, 10_000 (**Done on David PC**)

--- 
## Action Space 
- Discrete 
- Continuous 

--- 
## Uncertainty  
- Noise - on, off (There's lidar noise, etc)
- Determinism/Stochasticity - on, off 

## Fitness Function 
- Fitness function - none, health, energy (**David - Running on OMEN**)

This is what we are going to compare against. 

## Further Considerations 
https://gemini.google.com/share/16047d46abf0 - other maps, multiple maps; sensory impairments, sensor noise, 

# Dir Structure 
We want to have the following folder structure for experiments
```
experiments
    <variable>
        <variable value 1>
            <strategy 1>
            <strategy 2>
            <strategy 3>
            <strategy 4>
            <strategy 5>
        <variable value 2>
            <strategy 1>
            <strategy 2>
            <strategy 3>
            <strategy 4>
            <strategy 5>
        <variable value 3>
            <strategy 1>
            <strategy 2>
            <strategy 3>
            <strategy 4>
            <strategy 5>
```
eg 
```
experiments 
    steps
        100/
            random
            levy
            uniform 
            nn_random 
            nn_spiking 
        500
            random
            levy
            uniform 
            nn_random 
            nn_spiking 
        1000 
            random
            levy
            uniform 
            nn_random 
            nn_spiking 
        1000
            random
            levy
            uniform 
            nn_random 
            nn_spiking 
    mutation_rate
        0.05/
            random
            levy
            uniform 
            nn_random 
            nn_spiking 
        0.3
            random
            levy
            uniform 
            nn_random 
            nn_spiking 
        0.5 
            random
            levy
            uniform 
            nn_random 
            nn_spiking 
        0.75 
            random
            levy
            uniform 
            nn_random 
            nn_spiking 

```

# To-do 
Try do pending experiment for spiking_nn with a 10_000 population. It was taking too long on my (David's) computer. 

Let's complete: 
- ~~standard~~ 
- ~~mutation rate~~ 
- ~~mutation magnitude~~ 
- steps 
- ~~population~~ 

Then proceed to others 
