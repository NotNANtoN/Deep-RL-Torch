# Deep-RL-Torch

The following packages are necessary to run the code:
PyTorch version 1.0
MineRL > version .23
Gym version 0.10.5
Torchvision version 0.2.1

## Run in headless mode:

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train.py --env [ENV_NAME] --n_[steps|episodes|hours] N
