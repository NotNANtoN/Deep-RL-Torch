import logging

import torch
import sys
import os

from deep_rl_torch import Trainer
from deep_rl_torch.parser import create_parser


def get_env(env_short):
    # Basic Discrete:
    lunar = "LunarLander-v2"
    cart = "CartPole-v1"
    acro = "Acrobot-v1"
    mountain = "MountainCar-v0"
    # Continuous:
    pendulum = "Pendulum-v0"
    mountain_cont = "MountainCarContinuous-v0"
    # Box2d Continuous:
    lunar_cont = "LunarLanderContinuous-v2"
    car_race = "CarRacing-v0"
    biped = "BipedalWalker-v2"
    biped_hard = "BipedalWalkerHardcore-v2"
    # Mujoco:
    inv_double_pend = "InvertedDoublePendulum-v2"
    hopper = "Hopper-v2"
    ant = "Ant-v2"
    cheetah = "HalfCheetah-v2"
    human = "Humanoid-v2"
    human_stand = "HumanoidStandup-v2"
    # MineRL:
    tree = "MineRLTreechop-v0"
    nav_dense = "MineRLNavigateDense-v0"
    nav = "MineRLNavigate-v0"
    nav_extreme_dense = "MineRLNavigateExtremeDense-v0"
    nav_extreme = "MineRLNavigateExtreme-v0"
    pickaxe = "MineRLObtainIronPickaxe-v0"
    pickaxe_dense = "MineRLObtainIronPickaxeDense-v0"
    diamond = "MineRLObtainDiamond-v0"
    diamond_dense = "MineRLObtainDiamondDense-v0"
    # Aigar:
    pellet = "AigarPellet-v0"
    pellet_grid = "AigarPelletGrid-v0"
    greedy = "AigarGreedy1-v0"
    greedy_grid = "AigarGreedy1Grid-v0"

    if env_short == "cart":
        env = cart
    elif env_short == "lunar":
        env = lunar
    elif env_short == "acro":
        env = acro
    elif env_short == "tree":
        env = tree
    elif env_short == "diamond":
        env = diamond
    elif env_short == "pickaxe":
        env = pickaxe
    elif env_short == "nav":
        env = nav
    elif env_short == "nav_dense":
        env = nav_dense
    elif env_short == "pong":
        env = "Pong-v0"
    elif env_short == "pong_ram":
        env = "Pong-ram-v0"
    elif env_short == "atlantis":
        env = "Atlantis-v0"
    elif env_short == "pellet":
        env = pellet
    elif env_short == "pellet_grid":
        env = pellet_grid
    elif env_short == "greedy":
        env = greedy
    elif env_short == "greedy_grid":
        env = greedy_grid
    else:
        env = env_short
    return env


def create_arg_dict():
    # Parse arguments:
    parser = create_parser()
    args = parser.parse_args()
    parameters = vars(args)

    #parameters = get_default_hyperparameters()
    #parameters.update(arg_dict)
    #parameters = apply_parameter_changes(env, verbose)
        
    return parameters


if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = True
    
    parameters = create_arg_dict()
    
    # Decide on env here:
    env = get_env(parameters["env"])

    # Set up debugging:
    log_setup = parameters["debug"]
    if log_setup:
        logging.basicConfig(level=logging.DEBUG)
    parameters["verbose"] = True
    print("Env: ", env)
    # Set up trainer
    trainer = Trainer(env, **parameters)
    # Train
    try:
        trainer.run(total_steps=parameters["steps"], n_episodes=parameters["episodes"], n_hours=parameters["hours"],
                    render=parameters["render"], verbose=parameters["verbose"], disable_tqdm=False)
    except KeyboardInterrupt:
        print("KeyboardInterrupt - Goodbye!")
    finally:
        # Clean up the trainer in any case
        trainer.close()
        del trainer
