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

## Generations 
- We did for 1,000 generations and observed that there's little change after 200 generations, so we decided to use this as the base. 

## Time 
Time pressure (max steps) - 100, 500, 1_000, 10_000 (**Running on OMEN**) 

--- 
## Natural Selection 
- Mutation rate - 0.05, 0.1, 0.3, 0.5, 0.75 (**Done on OMEN**)
- Mutation magnitude - 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0 (**Done on OMEN**)
- Recombination - True, False 

--- 
## Network architecture
- Network architecture - ? 
- Hidden layer size - ? 
- LIF timesteps - 1, 2, 5, 10

--- 
## Difficulty 
- Goal difficulty - 10, 30, 50, 70 (**Running on Kaggle**)

--- 
## Other factors 

- Robot size - 1, 3, 5 (Peter)
- Cirriculum learning - on, off (Peter)
- Population - 100, 1_000, 10_000 (**Done on David PC**)

--- 
## Noise 
- Noise - on, off 
- Determinism/Stochasticity - on, off 

## Fitness Function 
- Fitness function - none, health, energy (Peter)

This is what we are going to compare against. 

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