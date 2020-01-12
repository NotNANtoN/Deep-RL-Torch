import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    # General:
    train_time_group = parser.add_mutually_exclusive_group()
    train_time_group.add_argument("--n_steps", type=int, default=0)
    train_time_group.add_argument("--n_episodes", type=int, default=0)
    train_time_group.add_argument("--n_hours", type=float, default=0.0)
    parser.add_argument("--env", help="Env name", default="cart")
    # User experience:
    parser.add_argument("--tb_comment", help="comment that is added to tensorboard", default="")
    parser.add_argument("--save_path", default="train/")
    parser.add_argument("--save_percentage", type=float, default=0.05)
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--load_path", default="")
    parser.add_argument("--tqdm", type=int, help="comment that is added to tensorboard", default=0)
    parser.add_argument("--render", help="render the env", action="store_true", default=0)
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true", default=1)
    parser.add_argument("--debug", action="store_true", default=0)
    parser.add_argument("--log", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=1000)
    # Train basics:
    parser.add_argument("--store_on_gpu", type=int, default=0)
    parser.add_argument("--pin_tensors", type=int,  default=0)
    parser.add_argument("--gamma", type=float, help="Discount factor", default=0.99)
    parser.add_argument("--frameskip", type=int, help="The number of times the env.step() is called per action",
                        default=4)
    parser.add_argument("--max_episode_steps", type=int, help="Limit the length of episodes", default=0)
    parser.add_argument("--reward_std", type=float, default=0.0)
    # Target net:
    parser.add_argument("--use_target_net", type=int, default=1)
    parser.add_argument("--use_polyak_averaging", type=int, default=1)
    parser.add_argument("--target_network_hard_steps", type=int, default=10000)
    parser.add_argument("--polyak_averaging_tau", type=float, default=0.0025)
    # Experience replay:
    parser.add_argument("--use_exp_rep", type=int, default=1)
    parser.add_argument("--replay_buffer_size", type=int, default=1000000)
    parser.add_argument("--use_PER", type=int, default=1)
    parser.add_argument("--PER_alpha", type=float, default=0.6)
    parser.add_argument("--PER_beta", type=float, default=0.4)
    parser.add_argument("--PER_anneal_beta", type=int, default=1)
    parser.add_argument("--use_CER", type=int, default=1)
    # Expert Data:
    parser.add_argument("--use_expert_data", type=int, default=0)
    parser.add_argument("--pretrain_percentage", type=float, default=0.1)
    parser.add_argument("--pretrain_weight_decay", type=float, default=0.0)
    # Exploration:
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--explore_until_reward", type=int, default=0)
    parser.add_argument("--action_sigma", type=float, default=0.0)
    parser.add_argument("--n_initial_random_actions", type=int, default=50000)
    # Split reward:
    parser.add_argument("--split_Bellman", type=int, default=0)
    # QV:
    group_QV = parser.add_mutually_exclusive_group()
    group_QV.add_argument("--use_QV", type=int, default=0)
    group_QV.add_argument("--use_QVMAX", type=int, default=0)
    # Eligibility traces:
    parser.add_argument("--use_efficient_traces", type=int, default=0)
    parser.add_argument("--elig_traces_lambda", type=float, default=0.8)
    parser.add_argument("--elig_traces_update_steps", type=int, default=10000)
    parser.add_argument("--elig_traces_anneal_lambda", type=int, default=0)
    # Input Normalization:
    parser.add_argument("--normalize_obs", type=int, default=1)
    parser.add_argument("--freeze_normalize_after_initial", type=int, default=1)
    parser.add_argument("--rgb_to_gray", type=int, default=1)
    parser.add_argument("--matrix_max_val", type=int, help="Maximum value an element in an input matrix can have",
                        default=255)
    # NN Architecture:
    parser.add_argument("--layers_conv", default="mnhi_later")
    parser.add_argument("--hidden_size", default=512)
    # NN Training:
    parser.add_argument("--optimize_centrally", type=int, default=1)
    parser.add_argument("--use_half", type=int, default=0)
    parser.add_argument("--general_lr", type=float, default=0.00025)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", default="RAdam")
    parser.add_argument("--max_norm", type=float, default=0)
    parser.add_argument("--network_updates_per_step", type=float, default=0.25)
    parser.add_argument("--lr_Q", type=float, default=0.0002)
    parser.add_argument("--lr_V", type=float, default=0.0002)
    parser.add_argument("--lr_r", type=float, default=0.0001)
    parser.add_argument("--lr_actor", type=float, default=0.00005)
    # REM:
    parser.add_argument("--use_REM", type=int, default=0)
    parser.add_argument("--REM_num_heads", type=int, default=5)
    parser.add_argument("--REM_num_samples", type=int, default=3)
    # AC:
    parser.add_argument("--use_actor_critic", action="store_true", default=0)
    group_AC = parser.add_mutually_exclusive_group()
    group_AC.add_argument("--use_CACLA_V", action="store_true", default=0)
    group_AC.add_argument("--use_CACLA_Q", action="store_true", default=0)
    group_AC.add_argument("--use_DDPG", action="store_true", default=0)
    group_AC.add_argument("--use_SPG", action="store_true", default=0)
    group_AC.add_argument("--use_GISPG", action="store_true", default=0)
    # Model evaluation:
    parser.add_argument("--eval_rounds", help="Number of rounds that the model should be evaluated in "
                                              "to calculate an average return  in model evaluation. "
                                              "Set to 0 to disable model evaluation.", type=int, default=5)
    parser.add_argument("--eval_percentage", help="Percentage of training time after which the model will be evaluated "
                                                  "in.", type=float, default=0.03)
    return parser
