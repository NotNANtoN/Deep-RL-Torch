import logging

import torch
import sys
import os

from optimizers.RAdam import RAdam
from deep_rl_torch.env_wrappers import SerialDiscreteActionWrapper, Convert2TorchWrapper, HierarchicalActionWrapper,\
    AtariObsWrapper, DefaultWrapper
from deep_rl_torch import Trainer
from parser import create_parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
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


    # Parse arguments:
    parser = create_parser()
    args = parser.parse_args()
    arg_dict = vars(args)

    # NN architectures:
    hidden_size = arg_dict["hidden_size"]
    # Save in the sense of enough non-linearities per block:
    save_feature_block = [{"name": "linear", "neurons": hidden_size, "act_func": "relu"},
                              {"name": "linear", "neurons": hidden_size}]
    save_hidden_block = [{"name": "linear", "neurons": hidden_size, "act_func": "relu"},
                             {"name": "linear", "neurons": hidden_size, "act_func": "relu"}]
    
    thin_block = [{"name": "linear", "neurons": hidden_size, "act_func": "relu"}]

    test_block = [{"name": "linear", "neurons": 64, "act_func": "relu"}]

    standard_hidden_block = thin_block
    standard_feature_block = thin_block

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
                           {"name": "conv", "filters": 128, "kernel_size": 3, "stride": 1, "act_func": "relu"},
                           {"name": "conv", "filters": 256, "kernel_size": 3, "stride": 1, "act_func": "relu"}
                           ]
    # TODO: define R2D2 conv architecture! (IMPALA uses the same)

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
        "always_keys": ["sprint"], "exclude_keys": ["sneak"],
        "use_MineRL_policy": False,
        "forward_when_jump": True,

        # TODO: The following still need to be implemented:
        "PER_anneal_beta": False,
        "normalize_reward_magnitude": False,

        "use_dueling_network": False, # could be used in QV especially
        "use_hrl": False,  # important
        "use_backwards_sampling": False, # new idea: sample batch of idxs, train on these, then get every idx-1 of idxs and train on these too for faster value propagation (kind of similar to eligibility traces, so maybe unnecessary)
        "use_double_Q": False,  # also implement for REM: sample a random other Q net that serves as target
        "use_clipped_double_Q": False, # also implement for REM. Either as above, or take overall min Q val over all networks that are sampled
        "epsilon_mid": 0.1, "boltzmann_temp": 0,
        "epsilon_decay": 0,

        "QV_NO_TARGET_Q": False,  # does it even make sense to do??
        "target_policy_smoothing_noise": 0.1,  # only for ac. can be delayed. can decay, make uniform or clip
        "delayed_policy_update_steps": 0,  # only for actor critic, can be delayed to implement


        "use_world_model": False,
        "TDEC_episodic": True,
        "TDEC_ENABLED": False, "TDEC_TRAIN_FUNC": "normal",
        "TDEC_ACT_FUNC": "abs",
        "TDEC_SCALE": 0.5, "TDEC_MID": 0, "TDEC_USE_TARGET_NET": True, "TDEC_GAMMA": 0.99,
    }

    # TODO: Introduce lr schedule - cosine anneal... but maybe don't. How does it work with ADAM to anneal lr? - apparently AdamW (in PyTorch) decouples weight decay properly from optimizer
#
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Decide on env here:
    atari_envs = ["pong", "atlantis"]
    env_short = parameters["env"]
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
    else:
        env = env_short
    print("Env: ", env)
    tensorboard_comment = parameters["tb_comment"] + "_" if parameters["tb_comment"] else ""
    unfiltered_arguments = iter(sys.argv[1:])
    arguments = []
    filter_single = ["debug", "render"]
    filter_double = ("log", "save", "load", "verbose", "tqdm")
    for arg in unfiltered_arguments:
        next_word = False
        for word in filter_single:
            if word in arg:
                next_word = True
                break
        for word in filter_double:
            if word in arg:
                next(unfiltered_arguments)
                next_word = True
                break
        if next_word:
            continue
        value = next(unfiltered_arguments)
        word = arg + str(value)
        arguments.append(word)

    arguments.sort()

    for arg in arguments:
        if arg[:2] == "--":
            arg = arg[2:]
        modified_arg = ""
        for char in arg:
            if char == ".":
                modified_arg += "_"
            else:
                modified_arg += char
        tensorboard_comment += modified_arg
    parameters["tb_comment"] = tensorboard_comment
    print("Tensorboard comment: ", tensorboard_comment)

    if "MineRL" in env:
        print("MineRL env!")
        use_hierarchical_action_wrapper = True
        parameters["convert_2_torch_wrapper"] = Convert2TorchWrapper
        if use_hierarchical_action_wrapper:
            parameters["action_wrapper"] = HierarchicalActionWrapper
        else:
            parameters["action_wrapper"] = SerialDiscreteActionWrapper

        parameters["use_MineRL_policy"] = True
        #if "Pickaxe" in env or "Diamond" in env:
        #    parameters["use_MineRL_policy"] = True
    elif env_short in atari_envs:
        parameters["convert_2_torch_wrapper"] = AtariObsWrapper
        print("Atari env!")
    else:
        parameters["convert_2_torch_wrapper"] = DefaultWrapper

    # Set up debugging:
    log_setup = parameters["debug"]
    if log_setup:
        logging.basicConfig(level=logging.DEBUG)

    trainer = Trainer(env, parameters, log=parameters["log"], tb_comment=tensorboard_comment)
    try:
        trainer.run(n_steps=parameters["n_steps"], n_episodes=parameters["n_episodes"], n_hours=parameters["n_hours"],
                render=parameters["render"], verbose=parameters["verbose"])
    except KeyboardInterrupt:
        print("KeyboardInterrupt - Goodbye!")
        trainer.close()
        del trainer
    except:
        print("Error while training, trying to close gracefully...")
        trainer.close()
        del trainer
        raise

    # TODO: log TDE of new incoming transitions and expected Q-vals

