# Pessimistic Neigbourhood Aggregation for States in Reinforcement Learning

## Description
Reinforcement Learning (RL) is the task of maximising future reward by choosing the right actions in the right states. The agent typically starts out with limited knowledge about the environment, and learns from experience. When the number of states is finite and not too large, simple learning mechanisms can be devised relying on visiting the state many times. However, when the number of states is very large or inifinite, no state may be visited twice. In order to learn, the agent needs to extrapolate the value of action in one state from experience from similar states. The purpose of this project will be to explore a few novel ideas for how to do this extrapolation.

## Problem Setup
The task is to learn an MDP with a finite set *A* of actions, and inifinite metric space *S* of states. Sometimes we assume S has large dimensions, in which case states may be referred to as feature vectors. The value function V(s) supposed to be smooth in the state s.

## Solution Methods


## Environment

#### Mountain Car Domain
Mountain Car is a standard testing domain in Reinforcement Learning, in which an under-powered car must drive up a steep hill. Since gravity is stronger than the car's engine, even at full throttle, the car cannot simply accelerate up the steep slope. The car is situated in a valley and must learn to leverage potential energy by driving up the opposite hill before the car is able to make it to the goal at the top of the rightmost hill.

![Mountain Car](https://github.com/maleakhiw/Pessimistic-Neighbourhood-Aggregation-for-States-in-Reinforcement-Learning/blob/master/mountain-car.png)
The picture and description is taken from Wikipedia. For more information, you can visit <a href="https://en.wikipedia.org/wiki/Mountain_car_problem">here</a>


