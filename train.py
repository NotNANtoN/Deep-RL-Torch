import torch

from RAdam import RAdam
from trainer import Trainer

if __name__ == "__main__":
    # TODO: here we could declare functions for certain events that we pass as parameters. For MineRL we could define how the observation is split into matrix and vector and how to deal with the action space

    standard_feature_block = [{"name": "linear", "neurons": 256, "act_func": "relu"},
                              {"name": "linear", "neurons": 128}]
    standard_hidden_block =  [{"name":"linear", "neurons": 64, "act_func": "relu"},
                             {"name":"linear", "neurons": 64, "act_func": "relu"}]

    test_block = [{"name": "linear", "neurons": 64, "act_func": "relu"}]

    #standard_hidden_block = test_block
    #standard_feature_block = test_block

    # TODO: add action-embedding/hidden layer for F_sa

    layers_feature_vector = standard_hidden_block
    layers_feature_merge = standard_feature_block
    layers_action = standard_feature_block
    layers_state_action_merge = standard_feature_block
    layers_r = standard_hidden_block
    layers_Q = standard_hidden_block
    layers_V = standard_hidden_block
    layers_actor = standard_hidden_block

    parameters = {# General:
                  "use_QV": False, "split_Bellman": True, "gamma": 1,
        "use_QVMAX": True, "use_target_net": True, "max_episode_steps": 0,
                  "normalize_obs": False, # turning this on destroys training on cartpole
                  "reward_std": 0.0,
                  # Actor-Critic:
                  "use_actor_critic": False, "use_CACLA_V": False, "use_CACLA_Q": False, "use_DDPG": False,
                  "use_SPG": False, "use_GISPG": False,
                  # Target-net:
                  "target_network_hard_steps": 250, "use_polyak_averaging": True, "polyak_averaging_tau":0.005,
                  # Replay buffer:
                  "use_exp_rep": True, "replay_buffer_size": 50000, "use_PER": False, "PER_alpha": 0.6, "PER_beta": 0.4,
                  "use_CER": True,
                  # Exploration:
                  "epsilon": 0.1, "epsilon_decay": 0, "action_sigma": 0, "epsilon_mid": 0.1, "boltzmann_temp": 0,
                  "n_initial_random_actions": 3000,
                  # REM:
                  "use_REM": False, "REM_num_heads": 20, "REM_num_samples": 5,
                  # NN Training:
                  "lr_Q": 0.001, "lr_r": 0.001, "lr_V": 0.001, "lr_actor": 0.0005, "batch_size": 32, "optimizer": RAdam,
                  "max_norm": 1, "network_updates_per_step": 1,
                  # NN architecture setup:
                  "layers_feature_vector": layers_feature_vector, "layers_state_action_merge": layers_state_action_merge,
                  "layers_action": layers_action,
                  "layers_feature_merge": layers_feature_merge, "layers_r": layers_r, "layers_Q": layers_Q,
                  "layers_V": layers_V,
                  "layers_actor": layers_actor,

                  # TODO: The following still need to be implemented:
                  "SPLIT_BELL_NO_TARGET_r": True,
                  "QV_NO_TARGET_Q": False,


                  "use_hrl": False, # important
                  "target_policy_smoothing_noise": 0.1, #  only for ac. can be delayed. can decay, make uniform or clip
                  "delayed_policy_update_steps": 0, # only for actor critic, can be delayed to implement
                  "use_double_Q": False, # also implement for REM: sample a random other Q net that serves as target
                  "use_clipped_double_Q": False, # also implement for REM. Either as above, or take overall min Q val over all networks that are sampled
                  "use_world_model": False,
                  "TDEC_episodic": True,
                  "TDEC_ENABLED": False, "TDEC_TRAIN_FUNC": "normal",
                  "TDEC_ACT_FUNC": "abs",
                  "TDEC_SCALE": 0.5, "TDEC_MID": 0, "TDEC_USE_TARGET_NET": True, "TDEC_GAMMA": 0.99,
                  }
    tensorboard_comment = ""

    # TODO: why does normalize_obs destroy the whole training for cartpole????

    # TODO: investigate the hyperparameter 'eps' of Adam and RAdam. For Deep RL it is usually set at 0.01 instead of 1e-8 -- see https://medium.com/autonomous-learning-library/radam-a-new-state-of-the-art-optimizer-for-rl-442c1e830564

    # TODO: Introduce lr schedule - cosine anneal

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # print("Action space: ", env.action_space)
    # print("Observation space: ", env.observation_space)

    # trainer = Trainer(environment_name, device)

    trainer = Trainer(cart, parameters, log=False, log_NNs=True, tb_comment=tensorboard_comment)
    # TODO: (important) introduce the max number of steps parameter in the agent and policies, such that they can update their epsilon values, learn rates etc
    trainer.run(50000, render=False, verbose=True)
