I think that we want to have a problem like this. 

A survival object. It must be found for survival of the robot. 

- Robot: 2 wheels, 4 lidar rays; predefined energy. Every step costs 1 energy point; reward is getting to within 2 units of the goal. 

We have a bunch of neurons connected up: 
- 4 input 
- 2 output 

Our system must in a way be capable of search and find, but let's not explicitly program that 

We have a starting point. 

We also have an ending point. If we are successful with the preconfigured endpoint, we can change things up to see if we can create an even more flexible system. Otherwise we might just be ""mastering the environment". 

We want to implement a natural selection system for our system, to see whether it successfully implements our survival system. 

My theory is that we are creating a system that's capable of SEARCH. And depending on how we define our reward, it could be: 
- By proximity (where the robot might learn to explore spaces thoroughly - example if we define our survival elements by proximity... ie, when we say that we only get a reward if we are 2 units near the reward)  
- By sight (where the robot might learn to explore by sight - if we define our fitness object as something which can be seen from far, eg, when we have an object perceptible by lidar)

We don't have any feedback for fitness in real evolution. 


https://claude.ai/share/a312720d-1d92-4da6-ab05-bbe69a0d6717 / https://claude.ai/chat/1e48324f-d4e1-4b3b-bffc-e364551ba240 has a good comparison 
https://claude.ai/chat/9c4913cf-beb2-4b83-be3a-5da41d0487b1 it might be nice to have a good look at this 

## Survival Proxies 
We want to have the option for using a particular fitness proxy (health, energy) or not (as it is right now)
Please rewrite the code to do this
Show where we need to make changes and where not
We want to apply this to ns strategies and have these options as well: 
- reproduce normally (no fitness) 
- reproduce only using top n members for proxy - (whether or not they survived). n can 10% of population 
- reproduce only using top n members for proxy - (for survivors only)

## Action - Action Space, Action Distribution 
In the continuous stochastic case, we might be tempted to constraint the outputs. 

These constraints embed prior knowledge about the robot's physical limits: 
- Motors can't spin infinitely fast
- The simulation has max speeds defined

But these constraints are important. What happens without constraints: 
- The network could output linear_vel = 5000, which:
- Would teleport the robot across the map in one step
- Breaks collision detection (skips over walls)
- Makes the simulation unstable/nonsensical

## Check These 
- https://www.blackwellpublishing.com/ridley/classictexts/wright.pdf