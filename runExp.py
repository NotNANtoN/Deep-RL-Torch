from experimenter import runExp


lunar = "LunarLander-v2"
cart = "CartPole-v1"
acro = "Acrobot-v1"
mountain = "MountainCar-v0"
pong = "Pong-v0"
seaquest = "Seaquest-v0"

tree = "MineRLTreechop-v0"
diamond = "MineRLObtainDiamond-v0"

paramsQ = {"name": "Q"}

Q_s = [{"name": "Q" + str(i + 1)} for i in range(30)]

Q_no_exp_rep = {"name": "Q-Online", "USE_EXP_REP": False}


paramsQ_target_net_250 = {"name": "Q+Target_250", "TARGET_UPDATE": 250}
paramsQ_target_net_100 = {"name": "Q+Target_100", "TARGET_UPDATE": 100}
paramsQ_target_net_50 = {"name": "Q+Target_50", "TARGET_UPDATE": 50}

paramsQV = {"name": "QV", "use_QV": True}
paramsQVMAX = {"name": "QVMAX", "use_QVMAX": True}


paramsEps_2 = {"name": "EpsMid 0.2", "EPS_MID": 0.2}
paramsEps_1 = {"name": "EpsMid 0.1", "EPS_MID": 0.1}
paramsEps_05 = {"name": "EpsMid 0.05", "EPS_MID": 0.05}
paramsEps_01 = {"name": "EpsMid 0.01", "EPS_MID": 0.01}
paramsEps_005 = {"name": "EpsMid 0.005", "EPS_MID": 0.005}
paramsEps_001 = {"name": "EpsMid 0.001", "EPS_MID": 0.001}
paramsEps_0001 = {"name": "EpsMid 0.0001", "EPS_MID": 0.0001}


paramsSplit = {"name": "Q+Split", "split_Bellman": True}
 

params_Q_reward_noise_0_1 = {"name": "Q-RewardNoise0.1", "reward_added_noise_std": 0.1}
params_Q_reward_noise_1 = {"name": "Q-RewardNoise1", "reward_added_noise_std": 1.0}
params_Q_reward_noise_10 = {"name": "Q-RewardNoise10", "reward_added_noise_std": 10.0}
params_split_reward_noise_0_1 = {"name": "Q+Split-RewardNoise0.1", "split_Bellman": True, "reward_added_noise_std": 0.1}
params_split_reward_noise_1 = {"name": "Q+Split-RewardNoise1", "split_Bellman": True, "reward_added_noise_std": 1.0}
params_split_reward_noise_10 = {"name": "Q+Split-RewardNoise10", "split_Bellman": True, "reward_added_noise_std": 10.0}  
paramsNoTarget_r_split_reward_noise_0_1 = {"name": "Q+Split-NoTarget_r-RewardNoise0.1", "split_Bellman": True, "SPLIT_BELL_NO_TARGET_r": True, "reward_added_noise_std": 0.1}
paramsNoTarget_r_split_reward_noise_1 = {"name": "Q+Split-NoTarget_r-RewardNoise1", "split_Bellman": True, "SPLIT_BELL_NO_TARGET_r": True, "reward_added_noise_std": 1.0}
paramsNoTarget_r_split_reward_noise_10 = {"name": "Q+Split-NoTarget_r-RewardNoise10", "split_Bellman": True, "SPLIT_BELL_NO_TARGET_r": True, "reward_added_noise_std": 10.0}                         


TDEC_pure = {"name": "Q+TDEC", "TDEC_ENABLED": True}
TDEC_pure_offset_01 = {"name": "Q+TDEC-Offset:-0.1", "TDEC_ENABLED": True, "critic_output_offset": -0.1}
TDEC_pure_offset_1 = {"name": "Q+TDEC-Offset:-1", "TDEC_ENABLED": True, "critic_output_offset": -1}
TDEC_pure_offset_10 = {"name": "Q+TDEC-Offset:-10", "TDEC_ENABLED": True, "critic_output_offset": -10}

TDEC_gamma_0_8 = {"name": "Q+TDEC-Gamma0.8", "TDEC_ENABLED": True, "TDEC_GAMMA": 0.8}
TDEC_no_target = {"name": "Q+TDEC-NoTarget", "TDEC_ENABLED": True, "TDEC_USE_TARGET_NET": False}
TDEC_mid = {"name": "Q+TDEC-DecayEps", "TDEC_ENABLED": True, "TDEC_MID": 0.1}
TDEC_abs_act = {"name": "Q+TDEC-absAct", "TDEC_ENABLED": True, "TDEC_ACT_FUNC": "absolute"}

TDEC_abs_train = {"name": "Q+TDEC-absTrain", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute"}
TDEC_mse_train = {"name": "Q+TDEC-mseTrain", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "mse"}
TDEC_abs_train_no_target = {"name": "Q+TDEC-absTrain-NoTarget", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                            "TDEC_USE_TARGET_NET": False}
TDEC_abs_train_0_9 = {"name": "Q+TDEC-absTrain-Scale0.9", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                      "TDEC_SCALE": 0.9}
TDEC_abs_train_0_1 = {"name": "Q+TDEC-absTrain-Scale0.1", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                      "TDEC_SCALE": 0.1}
TDEC_abs_train_mid_0_9 = {"name": "Q+TDEC-absTrain-Mid0.9", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                          "TDEC_MID": 0.9}
TDEC_abs_train_mid_0_5 = {"name": "Q+TDEC-absTrain-Mid0.5", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                          "TDEC_MID": 0.5}
TDEC_abs_train_mid_0_2 = {"name": "Q+TDEC-absTrain-Mid0.2", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                          "TDEC_MID": 0.2}
TDEC_abs_train_mid_0_1 = {"name": "Q+TDEC-absTrain-Mid0.1", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                          "TDEC_MID": 0.1}
TDEC_abs_train_mid_0_01 = {"name": "Q+TDEC-absTrain-Mid0.01", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                           "TDEC_MID": 0.01}

TDEC_pos_act = {"name": "Q+TDEC-posAct", "TDEC_ENABLED": True, "TDEC_ACT_FUNC": "positive"}
TDEC_pos_train = {"name": "Q+TDEC-posTrain", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive"}
TDEC_pos_train_no_target = {"name": "Q+TDEC-posTrain-NoTarget", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive",
                            "TDEC_USE_TARGET_NET": False}
TDEC_pos_train_decay = {"name": "Q+TDEC-posTrain-DecayEps", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive",
                        "TDEC_MID": 0.1}
TDEC_pos_train_no_target_decay = {"name": "Q+TDEC-posTrain-NoTarget-DecayEps", "TDEC_ENABLED": True,
                                  "TDEC_TRAIN_FUNC": "positive", "TDEC_MID": 0.1, "TDEC_USE_TARGET_NET": False}
TDEC_pos_train_no_target = {"name": "Q+TDEC-posTrain-NoTarget", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive",
                            "TDEC_USE_TARGET_NET": False}
TDEC_pos_train_0_9 = {"name": "Q+TDEC-posTrain-Scale0.9", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive",
                      "TDEC_SCALE": 0.9}
TDEC_pos_train_1 = {"name": "Q+TDEC-posTrain-Scale1", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive",
                    "TDEC_SCALE": 1.0}


reliabilityTest_long = Q_s

epsilonList = [paramsEps_2, paramsEps_1, paramsEps_05, paramsEps_01, paramsEps_005, paramsEps_001, paramsEps_0001]

noisyRewardList = [params_Q_reward_noise_0_1, params_Q_reward_noise_1, params_Q_reward_noise_10, params_split_reward_noise_0_1, params_split_reward_noise_1, params_split_reward_noise_10, paramsNoTarget_r_split_reward_noise_0_1, paramsNoTarget_r_split_reward_noise_1, paramsNoTarget_r_split_reward_noise_10]

splitShortList = [paramsQ, paramsSplit]

Q_list = [paramsQ]

TDEC_basic = [paramsQ, TDEC_pure, TDEC_abs_train, TDEC_pos_train, TDEC_mse_train]
TDEC_scaling_abs = [paramsQ, TDEC_abs_train, TDEC_abs_train_0_9, TDEC_abs_train_0_1, TDEC_abs_train_mid_0_9,
                    TDEC_abs_train_mid_0_5, TDEC_abs_train_mid_0_2, TDEC_abs_train_mid_0_1,
                    TDEC_abs_train_mid_0_01]  # ....
TDEC_noTargets = [paramsQ, TDEC_pure, TDEC_no_target, TDEC_abs_train, TDEC_abs_train_no_target, TDEC_pos_train,
                  TDEC_pos_train_no_target]
TDEC_train_or_act = [paramsQ, TDEC_pure, TDEC_abs_train, TDEC_abs_act, TDEC_pos_train, TDEC_pos_act]

TDEC_list = [paramsQ, TDEC_pure, TDEC_no_target, TDEC_mid, TDEC_abs_act, TDEC_abs_train, TDEC_pos_act, TDEC_pos_train]

TDEC_smart_list = [paramsQ, TDEC_pure, TDEC_abs_train, TDEC_pos_train, TDEC_pos_train_no_target, TDEC_pos_train_0_9,
                   TDEC_pos_train_1]
                   
TDEC_mse = [TDEC_mse_train]

QV_list = [paramsQ, paramsQV]

TDEC_offset_list = [TDEC_pure, TDEC_pure_offset_01, TDEC_pure_offset_1, TDEC_pure_offset_10]

paramsTracesLambda0_8 = {"name": "EligTraces0.8", "use_efficient_traces": 1, "elig_traces_lambda":0.8}
paramsTracesLambda0_5 = {"name": "EligTraces0.5", "use_efficient_traces": 1, "elig_traces_lambda":0.5}
paramsTracesAnneal = {"name": "EligTracesAnneal", "use_efficient_traces": 1, "elig_traces_anneal_lambda":1}
paramsQuick = {"name": "Q", "n_initial_random_actions":1}

QV_quick = {"name": "QV", "n_initial_random_actions":1, "use_QV": 1}


traces = [paramsQ, paramsTracesAnneal, paramsTracesLambda0_8, paramsTracesLambda0_5]


PER = {"name": "Q+PER", "use_PER": 1}
PER2w = {"name": "Q+PER-2worker", "use_PER": 1, "worker": 2}
PER8w = {"name": "Q+PER-8worker", "use_PER": 1, "worker": 8}


large_batch = {"name": "Q+BS256", "batch_size": 256}




# test:
#runExp(pong, [paramsQuick, QV_quick], number_of_tests=5, length_of_tests=111, path="test")


# Test buffer:
#runExp(pong, [paramsQ, PER], number_of_tests=5, length_of_tests=8000000, path="base")

# Test PER:
runExp(seaquest, [paramsQ, PER], number_of_tests=5, length_of_tests=8000000, path="PER_seaquest")


# Run new Rainbow exps:
runExp(pong, [paramsQ, paramsQV, paramsSplit, PER, paramsQVMAX], number_of_tests=5, length_of_tests=8000000, path="Pong_Trials")

runExp(tree, [paramsQ, paramsQV, paramsSplit, PER, paramsQVMAX], number_of_tests=5, length_of_tests=8000000, path="Treechop_Trials")


# Test:
#runExp(cart, traces, number_of_tests=5, length_of_tests=20000, path="traces/", on_server=True, optimize="no", #run_metric_percentage=1, run_metric_final_percentage_weight=0)

#runExp(pong, traces, number_of_tests=5, length_of_tests=100000, path="traces_pong/", on_server=True, optimize="no", run_metric_percentage=1, run_metric_final_percentage_weight=0)

#runExp(seaquest, traces, number_of_tests=5, length_of_tests=10000000, path="traces_seaquest/", on_server=True, optimize="no", run_metric_percentage=1, run_metric_final_percentage_weight=0)



############ Exps:
#runExp(cart, reliabilityTest_long, number_of_tests=50, length_of_tests=50000, path="long_rel_test_opt", on_server=True, optimize="comet_best")
#runExp(cart, reliabilityTest_long, number_of_tests=25, length_of_tests=50000, path="long_rel_test_opt", on_server=True, optimize="no")




#runExp(cart, QV_list, number_of_tests=50, length_of_tests=50000, path="QV_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, QV_list, number_of_tests=50, length_of_tests=50000, path="QV_lunar/", on_server=True, optimize="comet_best")

#runExp(cart, TDEC_basic, number_of_tests=50, length_of_tests=50000, path="TDEC_basic_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, TDEC_basic, number_of_tests=50, length_of_tests=50000, path="TDEC_basic_lunar/", on_server=True, optimize="comet_best")

# Bellman Split basics:
#runExp(cart, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_lunar/", on_server=True, optimize="comet_best")
# Optimize for separate nets and for individual hidden layers
#runExp(cart, splitBellList, number_of_tests=50, length_of_tests=50000, path="split_netArch_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, splitBellList, number_of_tests=50, length_of_tests=50000, path="split_netArch_lunar/", on_server=True, optimize="comet_best")
# Check if added params for different architecture is the cause behind performance boost:
#runExp(cart, checkAddedParamsInfluenceList, number_of_tests=50, length_of_tests=50000, path="split_AddedParamsEffect_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, checkAddedParamsInfluenceList, number_of_tests=50, length_of_tests=50000, path="split_AddedParamsEffect_lunar/", on_server=True, optimize="comet_best")
# Check if spit approaches are more robuts to Gaussian noise (they should be in theory):
#runExp(cart, noisyRewardList, number_of_tests=50, length_of_tests=50000, path="split_GaussianNoise_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, noisyRewardList, number_of_tests=50, length_of_tests=50000, path="split_GaussianNoise_lunar/", on_server=True, optimize="comet_best")
# Optimize only lr for split
#runExp(cart, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_optOnlyLr_cart/", on_server=True, optimize="comet_best", optimize_only_lr=True)
#runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_optOnlyLr_lunar/", on_server=True, optimize="comet_best", optimize_only_lr=True)
# Optimize only Q params for split:
#runExp(cart, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_optOnlyQparams_cart/", on_server=True, optimize="comet_best", optimize_only_Q_params=True)
#runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_optOnlyQparams_lunar/", on_server=True, optimize="comet_best", optimize_only_Q_params=True)





#runExp(cart, Q_no_exp_rep_list, number_of_tests=50, length_of_tests=50000, path="no_exp_rep_cart/", on_server=True, optimize="comet_best")

#runExp(lunar, TDEC_offset_list, number_of_tests=50, length_of_tests=50000, path="Q_offsets_lunar/", on_server=True, optimize="comet_best")





############ How reliable given same hyperparameters?
#runExp(cart, reliabilityTest, number_of_tests=10, length_of_tests=50000, path="no_optimization_10/", on_server=True, optimize="no")

#runExp(cart, reliabilityTest, number_of_tests=15, length_of_tests=50000, path="no_optimization_15/", on_server=True, optimize="no")

#runExp(cart, reliabilityTest, number_of_tests=20, length_of_tests=50000, path="no_optimization_20/", on_server=True, optimize="no")

#runExp(cart, reliabilityTest, number_of_tests=30, length_of_tests=50000, path="no_optimization_30/", on_server=True, optimize="no")

#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="no_optimization_50/", on_server=True, optimize="no")

#runExp(cart, reliabilityTest, number_of_tests=100, length_of_tests=50000, path="no_optimization_100/", on_server=True, optimize="no")

# still needs to be tested
########### What is the influence of more runs?
#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="comet_75_sets/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=1, evals_per_optimization_step=2)

#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="comet_100_sets/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=1, evals_per_optimization_step=2)




########## How many optimizations?
#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="comet_1_optim/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=1, evals_per_optimization_step=2)

#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="comet_2_optim/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=2, evals_per_optimization_step=2)

#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="comet_3_optim/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=3, evals_per_optimization_step=2)

#runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_short_comet_lunar/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=1, evals_per_optimization_step=2)






# Reliability Experiments:
#Standard:
#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="reliability_new/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=3, evals_per_optimization_step=3
#       )
#More runs:

#runExp(cart, reliabilityTest, number_of_tests=100, length_of_tests=3000, path="reliability/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=1, evals_per_optimization_step=1
#       )


# More evals per optimization:
#runExp(cart, reliabilityTest, number_of_tests=30, length_of_tests=3000, path="reliability_more_evals/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=1, evals_per_optimization_step=2
#       )
# More optimizations:
#runExp(cart, reliabilityTest, number_of_tests=30, length_of_tests=3000, path="reliability_more_optimizations/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=2, evals_per_optimization_step=1
#       )
       


# Both of the above:
#runExp(cart, reliabilityTest, number_of_tests=15, length_of_tests=3000, path="reliability_more_evals_and_optimizations/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=2, evals_per_optimization_step=2
#       )

# Checking if running separate TPEs makes sense or if we can just keep going
#runExp(cart, reliabilityTest, number_of_tests=30, length_of_tests=3000, path="reliability_more_evals_and_more_checks/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=6, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=1, evals_per_optimization_step=2
#       )


#runExp(cart, splitShortList, number_of_tests=50, length_of_tests=50000, path="cart_split/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15
#       )

#runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=100000, path="lunar_split/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15
#       )

#runExp(cart, TDEC_basic, number_of_tests=50, length_of_tests=50000, path="cart_TDEC/", on_server=True, optimize="tpe_best", number_of_best_runs_to_check=3, number_of_checks_best_runs=5, #final_evaluation_runs=15
#       )

#runExp(lunar, TDEC_basic, number_of_tests=50, length_of_tests=100000, path="lunar_TDEC/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15
#       )










# runExp(cart, TDEC_smart_list, number_of_tests=10, length_of_tests=50000, on_server=True, path="TDEC_smart_cart_2nd_exp/")
# runExp(lunar, TDEC_smart_list, number_of_tests=5, length_of_tests=100000, on_server=True, path="TDEC_smart_lunar/")

# runExp(lunar, QVC_list, number_of_tests=50, length_of_tests=20000, on_server=True, path="QVC_lunar/")

# cartDict = {}
# cartDict.update(runExp(cart, QVC_abs, number_of_tests=50, length_of_tests=20000, on_server=True, path="QVC_abs_cart/"))
# cartDict.update(runExp(cart, QVC_scale, number_of_tests=50, length_of_tests=20000, on_server=True, path="QVC_scale_cart/"))


############################# TDEC Experiments #############################################
####### cart
# runExp(cart, TDEC_scaling_abs, number_of_tests=50, length_of_tests=100000, path="cart_TDEC_abs_scaling/", on_server=True)


# runExp(cart, TDEC_basic, number_of_tests=50, length_of_tests=100000, path="cart_TDEC_basic/", on_server=True)
# runExp(cart, TDEC_noTargets, number_of_tests=50, length_of_tests=100000, path="cart_TDEC_no_targets/",  on_server=True)
# runExp(cart, TDEC_train_or_act, number_of_tests=50, length_of_tests=100000, path="cart_TDEC_train_or_act/",  on_server=True)

####### acro
# runExp(acro, TDEC_scaling_abs, number_of_tests=50, length_of_tests=100000, path="acro_TDEC_abs_scaling/", on_server=True)


# runExp(acro, TDEC_basic, number_of_tests=50, length_of_tests=100000, path="acro_TDEC_basic/", on_server=True)
# runExp(acro, TDEC_noTargets, number_of_tests=50, length_of_tests=100000, path="acro_TDEC_no_targets/", on_server=True)
# runExp(acro, TDEC_train_or_act, number_of_tests=50, length_of_tests=100000, path="acro_TDEC_train_or_act/", on_server=True)

####### mountain
# runExp(mountain, TDEC_scaling_abs, number_of_tests=50, length_of_tests=100000, path="mountain_TDEC_abs_scaling/", on_server=True)

# runExp(mountain, TDEC_basic, number_of_tests=50, length_of_tests=100000, path="mountain_TDEC_basic/", on_server=True)
# runExp(mountain, TDEC_noTargets, number_of_tests=50, length_of_tests=100000, path="mountain_TDEC_no_targets/", on_server=True)
# runExp(mountain, TDEC_train_or_act, number_of_tests=50, length_of_tests=100000, path="mountain_TDEC_train_or_act/", on_server=True)

###### lunar
# runExp(lunar, TDEC_scaling_abs, number_of_tests=50, length_of_tests=100000, path="lunar_TDEC_abs_scaling/", on_server=True)


# runExp(lunar, TDEC_basic, number_of_tests=50, length_of_tests=150000, path="lunar_TDEC_basic/", on_server=True)
# runExp(lunar, TDEC_noTargets, number_of_tests=50, length_of_tests=100000, path="lunar_TDEC_no_targets/", on_server=True)
# runExp(lunar, TDEC_train_or_act, number_of_tests=50, length_of_tests=100000, path="lunar_TDEC_train_or_act/", on_server=True)


############################# Split Experiments #############################################
# cartDict = {}
# cartDict.update(runExp(cart, splitShortList, number_of_tests=50, length_of_tests=100000, path="QsplitExp_cart/", on_server=True, randomizeParams=True))
# cartDict.update(runExp(cart, QV_split, number_of_tests=50, length_of_tests=100000, path="QVsplitExp_cart/", on_server=True, randomizeParams=True))
# cartDict.update(runExp(cart, QV_no_target, number_of_tests=50, length_of_tests=100000, path="QVnoTargetExp_cart/", on_server=True, randomizeParams=True))

# acroDict = {}
# acroDict.update(runExp(acro, splitShortList, number_of_tests=50, length_of_tests=100000, path="QsplitExp_acro/",  on_server=True, randomizeParams=True))
# acroDict.update(runExp(acro, QV_split, number_of_tests=50, length_of_tests=100000, path="QVsplitExp_acro/",  on_server=True, randomizeParams=True))
# acroDict.update(runExp(acro, QV_no_target, number_of_tests=50, length_of_tests=100000, path="QVnoTargetExp_acro/", on_server=True, randomizeParams=True))

# mountainDict = {}
# mountainDict.update(runExp(mountain, splitShortList, number_of_tests=50, length_of_tests=100000,  path="QsplitExp_mountain/", on_server=True, randomizeParams=True))
# mountainDict.update(runExp(mountain, QV_split, number_of_tests=50, length_of_tests=100000,  path="QVsplitExp_mountain/",  on_server=True, randomizeParams=True))
# mountainDict.update(runExp(mountain, QV_no_target, number_of_tests=50, length_of_tests=100000,  path="QVnoTargetExp_mountain/", on_server=True, randomizeParams=True))

# lunarDict = {}
# lunarDict.update(runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=150000, path="QsplitExp_lunar/", on_server=True, randomizeParams=True))
# lunarDict.update(runExp(lunar, QV_split, number_of_tests=50, length_of_tests=150000,  path="QVsplitExp_lunar/",  on_server=True, randomizeParams=True))
# lunarDict.update(runExp(lunar, QV_no_target, number_of_tests=50, length_of_tests=150000,  path="QVnoTargetExp_lunar/", on_server=True, randomizeParams=True))
