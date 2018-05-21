# Cmpe462-Machine Learning Term Project 
# "Learn to Move with Deep Reinforcement Learning"

## NIPS 2017 Challenge: Learning to Run
- https://github.com/stanfordnmbl/osim-rl
- https://www.crowdai.org/challenges/nips-2017-learning-to-run

## Folders & Submodules inside this repository
* osim-rl -> submodule containing simulation environment
* rlkit -> submodule containing deep reinforcement learning framework
* notebook_files -> files necessary to run the notebook
* scripts -> scripts used to manage training & simulation
* trained_params -> files composed of trained parameters
* training_statistics -> files composed of training statistics

## Installation
Installation of the submodules are told inside the submodules.

## How to simulate with trained parameters?
Run **osim-rl-with-farming/sim_farm/farm.py** on the opensim-rl conda environment and run **scripts/simulate_policy.py** with parameter file argument on the rlkit conda environment:

On the first terminal:
```
source activate opensim-rl
python2 osim-rl-with-farming/sim_farm/farm.py
```
On the second terminal:
```
source activate rlkit
python3 scripts/simulate_policy --file <path to parameter file>
```

## How to train models?
Note: To change hyperparameter, see and change method script files.

Run **osim-rl-with-farming/farming_scripts/farm.py** on the opensim-rl conda environment and run **method(ddpg/sac/td3) script** on the rlkit conda environment:
On the first terminal:
```
source activate opensim-rl
python2 osim-rl-with-farming/farming_scripts/farm.py
```
On the second terminal:
```
source activate rlkit
python3 <path to method(ddpg/sac/td3) script>
```

## Components
* [osim-rl-with-farming/farming_scripts](https://github.com/simitii/osim-rl/tree/ver1.5.5/farming_scripts) -> running multiple environments
* [osim-rl-with-farming/sim_farm](https://github.com/simitii/osim-rl/tree/ver1.5.5/sim_farm) -> trained parameter simulation
* [rlkit-with-farming](https://github.com/simitii/rlkit) -> train models using multiple environments

## Downloading only this repository
Run following command:
```
git clone https://github.com/CMPE462-Spring2018-Bogazici/term-project-sak
```

## Downloading All Source Code
Run following command:
```
git clone --recursive https://github.com/CMPE462-Spring2018-Bogazici/term-project-sak
```

## Plan
- Develop training system - **DONE**
- Train models(DDPG, SAC, TD3) - **DONE**
- Get a lot of visual material - **DONE**
- Write an IPython Notebook about the project - **DONE**
