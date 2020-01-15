# Deep-RL-Torch
This library serves two purposes:
1. It allows me to understand reinforcement learning algorithms.
2. It allows the easy combination of various reinforcement learning extensions, such as Prioritized Experience Replay, Eligibility Traces, Random Ensemble Mixture etc.

## Current features:

Currently all the following can be used and combined:

1. All the basic non-mujoco discrete environments, including Atari. Additionally MineRL environments can be used. No default.
2. Training for either a fixed number of steps, episodes or hours. No default.
3. [Uniform Experience Replay](http://www.incompleteideas.net/lin-92.pdf) and [Prioritied Experience Replay](https://arxiv.org/abs/1511.05952). Defaults to uniform exp replay.
4. Corrected Experience Replay, [CER](https://arxiv.org/abs/1712.01275). Can be combined either with uniform ode rprioritized experience replay.
5. Frame Stacking as in DQN. The stacking dimension can be chosen, although dimensions 0 probably performs best. Defaults to 6 frames and dim 0.
7. Frame Skipping as in DQN. Defaults to 4.
8. Use of a target net that is updated every N steps or of a Polyak-averaged target network, as seen in [DDPG](https://arxiv.org/abs/1509.02971). Defaults to Polyak-averaging.
9. Pretraining on Expert Data - currently only for MineRL data.
10. Bellman split - adds an additional head to a Q net that takes care of predicting only the immediate reward, whereas the other head is optimized to predict the value of the next state without the immediate reward. I could not yet show that this improves performance.
11. [QV](https://www.researchgate.net/publication/224446250_The_QV_family_compared_to_other_reinforcement_learning_algorithms) and [QVMax](https://arxiv.org/abs/1909.01779v1) learning
12. [Efficient Eligibility traces](https://arxiv.org/abs/1810.09967) - as described in v1 of the arXiv paper.
13. Observation normalization. Turned on by default.
14. Use of the [RAdam](https://arxiv.org/abs/1908.03265) optimizer.

## Upcoming features:

1. Two epsilon anneal schedules: DQN style linearly anneal until time T and then keep constant and exponential decay.
2. Improved replay buffer making use of the PyTorch dataloader.
3. Compatibility with Apex
4. Noisy Nets
5. Dueling Networks for Q function. Also an addition for it if it is combined with QV learning - the estimated state value in the Q network should be the output of the V network.

## Requirements:
The following packages are necessary to run the code:

```
atari-py==0.1.7
Box2D-kengz==2.3.3
bsuite==0.0.0
captum==0.1.0
numpy==1.16.4
ray==0.7.6
matplotlib==3.0.3
seaborn==0.9.0
tensorboard==1.13.1
torch==1.2.0
torchvision==0.4.0
tqdm==4.33.0
gym==0.15.4
```

## Run in headless mode:
This is only necessary for MineRL environments:

```
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train.py --env [ENV_NAME] --n_[steps|episodes|hours] N
```
