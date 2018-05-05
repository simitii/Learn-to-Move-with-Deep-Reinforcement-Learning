# Code for Parallel Training(Farming)

Codes are taken from an open-source code base https://github.com/ctmakro/stanford-osrl and modified according to the requirements of our project.

## Dependencies

  - Python 3.6 (For training with TF&Keras&Keral-RL)
  - Python 2.7 (For running osim-rl. There were some compatiblity issues with Python 3) 
  - Anaconda/Miniconda (for the harmony of the two above)
  - TF(Tensorflow)
  - numpy, gym
  - osim-rl (the simulation interface)
  - opensim (the simulation software)
  - Pyro4 (RPC. For communication)
  - IPython (for convenience)

## Parallelism / Farming

* Since the envirenment is very slow or computationally expensive, farming is crutial. 
- `farming_demo.py` will be used as a reference to develop keras-rl based training system.
- `ddpg2.py` file includes example code for complete training system.
- For example if training code is in `ddpg2.py` then following procedure can be done to start training:

Before starting `ddpg2.py`, you should first start one or more farm, with Python 2.7(working environment(conda environment) should have osim-rl and its requirements), by running `python farm.py` on each **SLAVE** machine you own. Then create a `farmlist.py` in the working directory (on the **HOST** machine) with the following content:

```py
farmlist_base = [('127.0.0.1', 4),('192.168.1.33',8)]
# a farm of 4 cores is available on localhost, while a farm of 8 available on another machine.
# expand the list if you have more machines.
# this file will be consumed by the host to find the slaves.
```
Then start `ipython -i ddpg2.py`, with python 3.6(working environment should have tensorflow&keras&keras-rl). The farmer should be able to reach the farm on your local machine. You can now type `r(100)` to train.


* Note: Please read <https://github.com/stanfordnmbl/osim-rl/issues/58> for more information about the slowness of the environment.

## Development Plan

- Delete unnecessary codes from the code base. (`farming_demo.py`, `ddpg2.py` files are just examples. They will be deleted after the related development is completed.)
- Develop Keras-RL based parallel DDPG training code, using farming example codes from `farming_demo.py`, `ddpg2.py` and keras-rl example codes from osim-rl repository (https://github.com/stanfordnmbl/osim-rl/blob/ver1.5.5/scripts/example.py)
- Develop SAC(Soft Actor Critics)[https://github.com/haarnoja/sac https://arxiv.org/abs/1801.01290] model compatible with Keras-RL interface so that features of keras-rl can be used and previously developed keras-rl based parallel DDPG training code can be used as a reference to develop keras-rl based SAC training code.
- Train both agents(DDPG, SAC)