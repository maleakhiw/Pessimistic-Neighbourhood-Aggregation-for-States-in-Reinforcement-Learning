# Pessimistic Neigbourhood Aggregation for States in Reinforcement Learning

## Description
Reinforcement Learning (RL) is the task of maximising future reward by choosing the right actions in the right states. The agent typically starts out with limited knowledge about the environment, and learns from experience. When the number of states is finite and not too large, simple learning mechanisms can be devised relying on visiting the state many times. However, when the number of states is very large or inifinite, no state may be visited twice. In order to learn, the agent needs to extrapolate the value of action in one state from experience from similar states. The purpose of this project will be to explore a few novel ideas for how to do this extrapolation.

## Problem Setup
The task is to learn an MDP with a finite set *A* of actions, and inifinite metric space *S* of states. Sometimes we assume S has large dimensions, in which case states may be referred to as feature vectors. The value function V(s) supposed to be smooth in the state s.

