import logging

import torch
import argparse
import sys
import os

from RAdam import RAdam
from env_wrappers import SerialDiscreteActionWrapper, Convert2TorchWrapper
from trainer import Trainer

def create_parser():
    parser = argparse.ArgumentParser()
    # General:
    parser.add_argument("--gamma", type=float, help="Discount factor", default=0.99)
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true", default=1)
    parser.add_argument("--log", action="store_true", default=0)
    parser.add_argument("--frameskip", type=int, help="The number of times the env.step() is called per action",
                        default=1)
    parser.add_argument("--max_episode_steps", type=int, help="Limit the length of episodes", default=0)
    parser.add_argument("--reward_std", type=float, default=0.0)
    # Target net:
    parser.add_argument("--use_target_net", type=int, default=1)
    parser.add_argument("--use_polyak_averaging", type=int, default=1)
    parser.add_argument("--target_network_hard_steps", type=int, default=250)
    parser.add_argument("--polyak_averaging_tau", type=float, default=0.0025)
    # Experience replay:
    parser.add_argument("--use_exp_rep", type=int, default=1)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)
    parser.add_argument("--use_PER", type=int, default=1)
    parser.add_argument("--PER_alpha", type=float, default=0.6)
    parser.add_argument("--PER_beta", type=float, default=0.4)
    parser.add_argument("--use_CER", type=int, default=1)
    # Exploration:
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--action_sigma", type=float, default=0.0)
    parser.add_argument("--n_initial_random_actions", type=int, default=10000)
    # Split reward:
    parser.add_argument("--split_Bellman", type=int, default=0)
    # QV:
    group_QV = parser.add_mutually_exclusive_group()
    group_QV.add_argument("--use_QV", type=int, default=0)
    group_QV.add_argument("--use_QVMAX", type=int, default=0)
    # Input Normalization:
    parser.add_argument("--normalize_obs", type=int, default=1)
    parser.add_argument("--freeze_normalize_after_initial", type=int, default=1)
    parser.add_argument("--rgb_to_gray", type=int, default=1)
    parser.add_argument("--matrix_max_val", type=int, help="Maximum value an element in an input matrix can have",
                        default=255)
    # NN Architecture:
    parser.add_argument("--layers_conv", default="mnhi_later")
    # NN Training:
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", default="RAdam")
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--network_updates_per_step", type=float, default=1.0)
    parser.add_argument("--lr_Q", type=float, default=0.0002)
    parser.add_argument("--lr_V", type=float, default=0.0002)
    parser.add_argument("--lr_r", type=float, default=0.0001)
    parser.add_argument("--lr_actor", type=float, default=0.00005)
    # REM:
    parser.add_argument("--use_REM", type=int, default=0)
    parser.add_argument("--REM_num_heads", type=int, default=5)
    parser.add_argument("--REM_num_samples", type=int, default=2)
    # AC:
    parser.add_argument("--use_actor_critic", action="store_true", default=0)
    group_AC = parser.add_mutually_exclusive_group()
    group_AC.add_argument("--use_CACLA_V", action="store_true", default=0)
    group_AC.add_argument("--use_CACLA_Q", action="store_true", default=0)
    group_AC.add_argument("--use_DDPG", action="store_true", default=0)
    group_AC.add_argument("--use_SPG", action="store_true", default=0)
    group_AC.add_argument("--use_GISPG", action="store_true", default=0)
    return parser

if __name__ == "__main__":
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
    pickaxe_denes = "MineRLObtainIronPickaxeDense-v0"
    diamond = "MineRLObtainDiamond-v0"
    diamond_dense = "MineRLObtainDiamondDense-v0"

    # NN architectures:

    standard_feature_block = [{"name": "linear", "neurons": 256, "act_func": "relu"},
                              {"name": "linear", "neurons": 256, "act_func": "tanh"}]
    standard_hidden_block = [{"name": "linear", "neurons": 256, "act_func": "relu"},
                             {"name": "linear", "neurons": 256, "act_func": "relu"}]

    test_block = [{"name": "linear", "neurons": 64, "act_func": "relu"}]

    # standard_hidden_block = test_block
    # standard_feature_block = test_block

    layers_feature_vector = standard_hidden_block
    layers_feature_merge = standard_feature_block
    layers_action = standard_feature_block
    layers_state_action_merge = standard_feature_block
    layers_r = standard_hidden_block
    layers_Q = standard_hidden_block
    layers_V = standard_hidden_block
    layers_actor = standard_hidden_block

    mnhi_early = [{"name": "conv", "filters": 32, "kernel_size": 8, "stride": 4, "act_func": "relu"},
                       {"name": "conv", "filters": 64, "kernel_size": 4, "stride": 2, "act_func": "relu"},
                       {"name": "conv", "filters": 64, "kernel_size": 2, "stride": 1, "act_func": "relu"}
                       ]
    mnhi_later = [{"name": "conv", "filters": 32, "kernel_size": 8, "stride": 4, "act_func": "relu"},
                       {"name": "conv", "filters": 64, "kernel_size": 4, "stride": 2, "act_func": "relu"},
                       {"name": "conv", "filters": 64, "kernel_size": 3, "stride": 1, "act_func": "relu"}
                       ]

    vizdoom_winner = [{"name": "conv", "filters": 16, "kernel_size": 3, "stride": 2, "act_func": "relu"},
                           {"name": "conv", "filters": 32, "kernel_size": 3, "stride": 2, "act_func": "relu"},
                           {"name": "conv", "filters": 64, "kernel_size": 3, "stride": 2, "act_func": "relu"},
                           {"name": "conv", "filters": 128, "kernel_size": 3, "stride": 2, "act_func": "relu"},
                           {"name": "conv", "filters": 256, "kernel_size": 3, "stride": 2, "act_func": "relu"}
                           ]
    # TODO: define R2D2 architecture! (IMPALA uses the same)

    layers_conv = standard_hidden_block

    parameters = {
        # NN architecture setup:
        "layers_feature_vector": layers_feature_vector, "layers_state_action_merge": layers_state_action_merge,
        "layers_action": layers_action,
        "layers_feature_merge": layers_feature_merge, "layers_r": layers_r, "layers_Q": layers_Q,
        "layers_V": layers_V,
        "layers_actor": layers_actor,

        # Env specific:
        "convert_2_torch_wrapper": None,
        "action_wrapper": None,
        "always_keys": ["sprint"], "exclude_keys": ["sneak"], "reverse_keys": ["forward"],
        "use_MineRL_policy": False,
        "forward_when_jump": True,

        # TODO: The following still need to be implemented:
        "use_dueling_network": False,
        "use_double_Q": False,  # also implement for REM: sample a random other Q net that serves as target
        "use_clipped_double_Q": False, # also implement for REM. Either as above, or take overall min Q val over all networks that are sampled
        "epsilon_mid": 0.1, "boltzmann_temp": 0,
        "epsilon_decay": 0,
        "PER_anneal_beta": False,
        "normalize_reward_magnitude": False,
        "lambda": 0,


        "use_hrl": False,  # important

        "QV_NO_TARGET_Q": False,  # does it even make sense to do??
        "target_policy_smoothing_noise": 0.1,  # only for ac. can be delayed. can decay, make uniform or clip
        "delayed_policy_update_steps": 0,  # only for actor critic, can be delayed to implement


        "use_world_model": False,
        "TDEC_episodic": True,
        "TDEC_ENABLED": False, "TDEC_TRAIN_FUNC": "normal",
        "TDEC_ACT_FUNC": "abs",
        "TDEC_SCALE": 0.5, "TDEC_MID": 0, "TDEC_USE_TARGET_NET": True, "TDEC_GAMMA": 0.99,
    }

    # Parse arguments:
    parser = create_parser()
    args = parser.parse_args()
    arg_dict = vars(args)
    parameters.update(arg_dict)

    # Convert strings in hyperparams to objects:
    # optimizer:
    if parameters["optimizer"] == "RAdam":
        parameters["optimizer"] = RAdam
    elif parameters["optimizer"] == "Adam":
        parameters["optimizer"] = torch.optim.Adam
    # Conv layers:
    if parameters["layers_conv"] == "mnhi_early":
        parameters["layers_conv"] = mnhi_early
    elif parameters["layers_conv"] == "mnhi_later":
        parameters["layers_conv"] = mnhi_later
    elif parameters["layers_conv"] == "vizdoom_winner":
        parameters["layers_conv"] = vizdoom_winner

    # TODO: Introduce lr schedule - cosine anneal... but maybe don't. How does it work with ADAM to anneal lr?

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Decide on env here:
    tensorboard_comment = "" + "_".join([argument if idx % 2 == 1 else argument[2:] for idx, argument in enumerate(sys.argv[1:])])
    print("Tensorboard comment: ", tensorboard_comment)
    env = nav_dense
    print("Env: ", env)
    if "MineRL" in env:
        print("MineRL env!")
        parameters["convert_2_torch_wrapper"] = Convert2TorchWrapper
        parameters["action_wrapper"] = SerialDiscreteActionWrapper
        if "Pickaxe" in env or "Diamond" in env:
            parameters["use_MineRL_policy"] = True

    # Set up logging:
    if not os.path.exists("logs"):
        os.mkdir("logs")
    log_setup =  parameters["log"]
    if log_setup:
        logging.basicConfig(filename="logs/" + tensorboard_comment + ".log", filemode='w', level=logging.DEBUG)

    trainer = Trainer(env, parameters, log=True, log_NNs=False, tb_comment=tensorboard_comment)
    # TODO: (important) introduce the max number of steps parameter in the agent and policies, such that they can update their epsilon values, learn rates etc
    trainer.run(600000, render=False, verbose=True)
