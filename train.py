import logging

import torch
import sys
import os

from deep_rl_torch import Trainer
from deep_rl_torch.parser import create_parser, create_arg_dict


def get_env(env_short):
    basic_dicrete = {"lunar": "LunarLander-v2",
                     "cart": "CartPole-v1",
                     "acro": "Acrobot-v1",
                     "mountain": "MountainCar-v0"}
    basic_cont = {"pendulum": "Pendulum-v0",
                  "mountain_cont": "MountainCarContinuous-v0"}
    box2d_cont = {"lunar_cont": "LunarLanderContinuous-v2",
                  "car_race": "CarRacing-v0",
                  "biped": "BipedalWalker-v2",
                  "biped_hard": "BipedalWalkerHardcore-v2"}
    mujoco = {"inv_double_pend": "InvertedDoublePendulum-v2",
              "hopper": "Hopper-v2",
              "ant": "Ant-v2",
              "cheetah": "HalfCheetah-v2",
              "human": "Humanoid-v2",
              "human_stand": "HumanoidStandup-v2"}
    minerl = {"tree": "MineRLTreechop-v0",
              "nav_dense": "MineRLNavigateDense-v0",
              "nav": "MineRLNavigate-v0",
              "nav_extreme_dense": "MineRLNavigateExtremeDense-v0",
              "nav_extreme": "MineRLNavigateExtreme-v0",
              "pickaxe": "MineRLObtainIronPickaxe-v0",
              "pickaxe_dense": "MineRLObtainIronPickaxeDense-v0",
              "diamond": "MineRLObtainDiamond-v0",
              "diamond_dense": "MineRLObtainDiamondDense-v0"}
    aigar = {"pellet": "AigarPellet-v0",
             "pellet_grid": "AigarPelletGrid-v0",
             "greedy": "AigarGreedy1-v0",
             "greedy_grid": "AigarGreedy1Grid-v0"}
    atari = {"pong": "Pong-v0",
             "pong_ram": "Pong-ram-v0",
             "atlantis": "Atlantis-v0"}
    env_dicts = [basic_dicrete, basic_cont, box2d_cont, mujoco, minerl, aigar, atari]
    all_envs = {shorthand: env_dict[shorthand] for env_dict in env_dicts for shorthand in env_dict}

    if env_short in all_envs:
        env = all_envs[env_short]
    else:
        env = env_short

    return env


if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = True
    
    params = create_arg_dict()
    
    # Decide on env here:
    env = get_env(params["env"])

    # Set up debugging:
    log_setup = params["debug"]
    if log_setup:
        logging.basicConfig(level=logging.DEBUG)
    params["verbose"] = True
    print("Env: ", env)
    # Set up trainer
    trainer = Trainer(env, **params)
    # Train
    try:
        trainer.run(total_steps=params["steps"], n_episodes=params["episodes"], n_hours=params["hours"],
                    render=params["render"], verbose=params["verbose"], disable_tqdm=False)
    except KeyboardInterrupt:
        print("KeyboardInterrupt - Goodbye!")
    finally:
        # Clean up the trainer in any case
        trainer.close()
        del trainer
