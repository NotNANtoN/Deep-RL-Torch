import minerl
import torch

from RAdam import RAdam
from env_wrappers import SerialDiscreteActionWrapper, Convert2TorchWrapper
from trainer import Trainer

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
                              {"name": "linear", "neurons": 128}]
    standard_hidden_block = [{"name": "linear", "neurons": 64, "act_func": "relu"},
                             {"name": "linear", "neurons": 64, "act_func": "relu"}]

    test_block = [{"name": "linear", "neurons": 64, "act_func": "relu"}]

    # standard_hidden_block = test_block
    # standard_feature_block = test_block

    # TODO: add action-embedding/hidden layer for F_sa

    layers_feature_vector = standard_hidden_block
    layers_feature_merge = standard_feature_block
    layers_action = standard_feature_block
    layers_state_action_merge = standard_feature_block
    layers_r = standard_hidden_block
    layers_Q = standard_hidden_block
    layers_V = standard_hidden_block
    layers_actor = standard_hidden_block

    # TODO: define conv architectures
    conv_mnhi_early = [{"name": "conv", "filters": 32, "kernel_size": 8, "stride": 4, "act_func": "relu"},
                       {"name": "conv", "filters": 64, "kernel_size": 4, "stride": 2, "act_func": "relu"},
                       {"name": "conv", "filters": 64, "kernel_size": 2, "stride": 1, "act_func": "relu"}
                       ]
    conv_mnhi_later = [{"name": "conv", "filters": 32, "kernel_size": 8, "stride": 4, "act_func": "relu"},
                       {"name": "conv", "filters": 64, "kernel_size": 4, "stride": 2, "act_func": "relu"},
                       {"name": "conv", "filters": 64, "kernel_size": 3, "stride": 1, "act_func": "relu"}
                       ]

    conv_vizdoom_winner = [{"name": "conv", "filters": 16, "kernel_size": 3, "stride": 2, "act_func": "relu"},
                           {"name": "conv", "filters": 32, "kernel_size": 3, "stride": 2, "act_func": "relu"},
                           {"name": "conv", "filters": 64, "kernel_size": 3, "stride": 2, "act_func": "relu"},
                           {"name": "conv", "filters": 128, "kernel_size": 3, "stride": 2, "act_func": "relu"},
                           {"name": "conv", "filters": 256, "kernel_size": 3, "stride": 2, "act_func": "relu"}
                           ]

    layers_conv = standard_hidden_block

    parameters = {
        # General:
        "use_target_net": True, "max_episode_steps": 0, "gamma": 1, "frameskip": 0,
        "use_QV": False, "split_Bellman": True,
        "use_QVMAX": True,
        "normalize_obs": True, "freeze_normalize_after_initial": True,
        "rgb_to_gray": True, "matrix_max_val": 255,
        "reward_std": 0.0,
        # Actor-Critic:
        "use_actor_critic": False, "use_CACLA_V": False, "use_CACLA_Q": False, "use_DDPG": False,
        "use_SPG": False, "use_GISPG": False,
        # Target-net:
        "target_network_hard_steps": 250, "use_polyak_averaging": True, "polyak_averaging_tau": 0.005,
        # Replay buffer:
        "use_exp_rep": True, "replay_buffer_size": 50000, "use_PER": True, "PER_alpha": 0.6, "PER_beta": 0.4,
        "use_CER": True,
        # Exploration:
        "epsilon": 0.1, "action_sigma": 0,
        "n_initial_random_actions": 1000,
        # REM:
        "use_REM": True, "REM_num_heads": 5, "REM_num_samples": 2,
        # NN Training:
        "lr_Q": 0.001, "lr_r": 0.001, "lr_V": 0.001, "lr_actor": 0.0005, "batch_size": 64, "optimizer": RAdam,
        "max_norm": 1, "network_updates_per_step": 1,
        # NN architecture setup:
        "layers_feature_vector": layers_feature_vector, "layers_state_action_merge": layers_state_action_merge,
        "layers_action": layers_action,
        "layers_feature_merge": layers_feature_merge, "layers_r": layers_r, "layers_Q": layers_Q,
        "layers_V": layers_V,
        "layers_actor": layers_actor,
        "layers_feature_matrix": conv_mnhi_later,

        # Env specific:
        "convert_2_torch_wrapper": None,
        "action_wrapper": None,
        "always_keys": ["sprint"], "exclude_keys": ["sneak"],

        # TODO: The following still need to be implemented:
        "epsilon_mid": 0.1, "boltzmann_temp": 0,
        "epsilon_decay": 0,
        "PER_anneal_beta": False,

        "QV_NO_TARGET_Q": False,  # does it make sense to do??

        "use_hrl": False,  # important
        "use_double_Q": False,  # also implement for REM: sample a random other Q net that serves as target
        "use_clipped_double_Q": False,

        "target_policy_smoothing_noise": 0.1,  # only for ac. can be delayed. can decay, make uniform or clip
        "delayed_policy_update_steps": 0,  # only for actor critic, can be delayed to implement

        # also implement for REM. Either as above, or take overall min Q val over all networks that are sampled
        "use_world_model": False,
        "TDEC_episodic": True,
        "TDEC_ENABLED": False, "TDEC_TRAIN_FUNC": "normal",
        "TDEC_ACT_FUNC": "abs",
        "TDEC_SCALE": 0.5, "TDEC_MID": 0, "TDEC_USE_TARGET_NET": True, "TDEC_GAMMA": 0.99,
    }

    # TODO: Introduce lr schedule - cosine anneal... but maybe don't. How does it work with ADAM to anneal lr?

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Decide on env here:
    tensorboard_comment = ""
    env = nav_dense
    if "MineRL" in env:
        print("MineRL env!")
        parameters["convert_2_torch_wrapper"] = Convert2TorchWrapper
        parameters["action_wrapper"] = SerialDiscreteActionWrapper

    trainer = Trainer(env, parameters, log=False, log_NNs=False, tb_comment=tensorboard_comment)
    # TODO: (important) introduce the max number of steps parameter in the agent and policies, such that they can update their epsilon values, learn rates etc
    trainer.run(50000, render=False, verbose=True)
