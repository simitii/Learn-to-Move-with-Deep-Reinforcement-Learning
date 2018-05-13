# Cmpe462-Machine Learning Term Project 
# "Learn to Run with Deep Reinforcement Learning"

## NIPS 2017 Challenge: Learning to Run
- https://github.com/stanfordnmbl/osim-rl
- https://www.crowdai.org/challenges/nips-2017-learning-to-run

## Method
- SAC(Soft Actor Critics)[https://github.com/haarnoja/sac https://arxiv.org/abs/1801.01290] : Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor

## Why this method?
Because of the following claims of the related paper:
- Sample efficiency 
- Exploration capabilities
- Speedy than DDPG which is the most used method for this challenge. 

## Dealing with Environment Slowness Problem
* Since the envirenment is very slow or computationally expensive, we do environment farming(running multiple environments parallel) for training.
* [osim-rl-with-farming/farming_scripts](https://github.com/simitii/osim-rl/tree/ver1.5.5/farming_scripts) -> running multiple environments
* [rlkit-with-farming](https://github.com/simitii/rlkit) -> train models using multiple environments

## Getting All Source Code
Run following command:
```
git clone --recursive https://github.com/CMPE462-Spring2018-Bogazici/term-project-sak
```

## Plan
- Develop training module - **DONE**
- Train models(SAC, DDPG)
- Get a lot of visual material about the perfomance of models
- Write an IPython Notebook about the project

## Inside the IPython Notebook
- Tell about the challenge
- Tell what we did 
- Give code pieces from the training module
- Tell the result
- Compare two models/methods(SAC, DDPG)
