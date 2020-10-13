# Deep-RL-Torch
This library serves two purposes:
1. It allows me to understand reinforcement learning algorithms by building them.
2. It allows the easy combination of various reinforcement learning extensions, such as Prioritized Experience Replay, Eligibility Traces, Random Ensemble Mixture etc.

## Current features:

Currently all the following can be used and combined:

* All the basic non-mujoco discrete environments, including Atari. Additionally MineRL environments can be used. No default.
* Training for either a fixed number of steps, episodes or hours. No default.
* [Uniform Experience Replay](http://www.incompleteideas.net/lin-92.pdf) and [Prioritied Experience Replay](https://arxiv.org/abs/1511.05952). Defaults to uniform exp replay.
* Corrected Experience Replay, [CER](https://arxiv.org/abs/1712.01275). Can be combined either with uniform ode rprioritized experience replay.
* Frame Stacking as in DQN. The stacking dimension can be chosen, although dimension 0 performs bestaccording to some rudimentary experiments. Defaults to 4 frames and dim 0.
* Frame Skipping as in DQN. Defaults to 4.
* Optimizations per step - how many batches to sample on for optimization per step in the environment. 0.25 (1 optimization every 4 steps) is the default atm, as in the DQN paper.
* Use of a target net that is updated every N steps or of a Polyak-averaged target network, as seen in [DDPG](https://arxiv.org/abs/1509.02971). Defaults to Polyak-averaging.
* Pretraining on Expert Data - currently only for MineRL data.
* Bellman split - adds an additional head to a Q net that takes care of predicting only the immediate reward, whereas the other head is optimized to predict the value of the next state without the immediate reward. I could not yet show that this improves performance.
* [QV](https://www.researchgate.net/publication/224446250_The_QV_family_compared_to_other_reinforcement_learning_algorithms) and [QVMax](https://arxiv.org/abs/1909.01779v1) learning
* [Efficient Eligibility traces](https://arxiv.org/abs/1810.09967) - as described in v1 of the arXiv paper.
* Observation standardization. Turned on by default.
* Use of the [RAdam](https://arxiv.org/abs/1908.03265) optimizer.

## Upcoming features:

1. Two epsilon anneal schedules: DQN style linearly anneal until time T and then keep constant and exponential decay.
2. Improved replay buffer making use of the PyTorch dataloader.
3. Compatibility with Apex
4. Noisy Nets
5. Dueling Networks for Q function. Also an addition for it if it is combined with QV learning - the estimated state value in the Q network should be the output of the V network.

## Installation:
The following packages are necessary to run the code:

First we need swig and g++ compilers for box2d environments:
```
sudo apt-get update
sudo apt-get install swig gcc g++
```

Then, install all python dependencies via the requirements.txt file:
```
python3 -m pip install -r requirements.txt
```

## Usage:

```
python train.py --env [ENV_SHORTHAND] --n_[steps|episodes|hours] N
```
ENV_SHORTHANDs are defined in the train.py script. Please define your own shorthand for additional environments.

All additional options can be seen in parser.py

## Run in headless mode:
This is only necessary for MineRL environments:

```
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train.py --env [ENV_NAME] --n_[steps|episodes|hours] N
```
