# Test Configurations
## Time 
Time pressure (max steps) - 100, 500, 1_000, 10_000 **OMEN** 

--- 
## Natural Selection 
- Mutation rate - 0.05, 0.1, 0.3, 0.5, 0.75
- Mutation magnitude - 
- Recombination - True, False 

--- 
## Network architecture
- Network architecture 
- Hidden layer size - 
- __LIF timesteps - 1, 2, 5, 10__ 

--- 
## Difficulty 
- Goal difficulty - 10, 30, 50, 70 

--- 
## Other factors 

- Robot size 
- Cirriculum learning 
- Generations - 100, 500 
- Population - 100, 1_000, 10_000 

--- 
## Noise 
- Noise 
- Determinism/Stochasticity 

## Fitness Function 
- No explicit fitness function 
- Explicit fitness function (based on health, energy)

# Standard 
We want to have the following standard for the configuration tests 
- _trials_ 7 
- _generations_ 200 (arrived at by checking the generations at which the rate of success improves by less than a certain value in 10 generations)
- _population_ 1000 
- _max\_steps_ 1000 

# Dir Structure 
We want to have the following folder structure for experiments
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

eg 

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