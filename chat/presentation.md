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

