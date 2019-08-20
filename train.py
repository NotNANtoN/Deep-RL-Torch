import gym

gym.logger.set_level(40)
import math
import itertools
import random
import numpy as np
import matplotlib

matplotlib.use('Pdf')
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import tracemalloc
import os
import linecache

from util import *
from networks import *
from policies import Agent

########## Setup #################
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'TDEC'))


########################################


def plot_rewards(rewards, name=None, xlabel="Step"):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel(xlabel)
    plt.ylabel('Cumulative Reward of current Episode')

    idxs = calculate_reduced_idxs(len(rewards), 1000)
    rewards = reducePoints(rewards, 1000)

    plt.plot(idxs, rewards)
    # Apply mean-smoothing and plot result
    window_size = len(rewards) // 10
    window_size += 1 if window_size % 2 == 0 else 0
    means = meanSmoothing(rewards, window_size)
    max_val = np.max(means)
    min_val = np.min(means)
    # plt.ylim(min_val, max_val * 1.1)
    plt.plot(idxs, means)
    if name is None:
        plt.savefig("current_test.pdf")
    else:
        plt.savefig(name + "_current.pdf")
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# "activation_function": hp.choice("activation_function", ["sigmoid", "relu", "elu"]),
#                "gamma_Q": hp.uniform("gamma_Q", 0.9, 0.9999),
#                "lr_Q": hp.loguniform("lr_Q", np.log(0.01), np.log(0.00001)),
#                "target_network_steps": hp.quniform("target_network_steps", 10, 20000, 1),
#                "hidden_neurons": hp.quniform("hidden_neurons", 32, 256, 1),
#                "hidden_layers": hp.quniform("hidden_layers", 1, 3, 1),
#                "batch_size": hp.quniform("batch_size", 16, 256, 1),
#                "replay_buffer_size": hp.quniform("replay_buffer_size", 1024, 100000, 1)],
#                "epsilon_mid": hp.uniform("epsilon_mid", 0.25, 0.0001) 





class Trainer:
    def __init__(self, env_name, hyperparameters, log=True):
        # Init logging:
        self.path = os.getcwd()
        self.log = Log(self.path + '/tb_log', log)
        self.steps_done = 0

        # Init env:
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        # Extract relevant hyperparameters:
        if hyperparameters["max_episode_steps"] > 0:
            self.max_steps_per_episode = hyperparameters["max_episode_steps"]
        elif self.env_name == "LunarLander-v2":
            self.max_steps_per_episode = 1000
        elif self.env_name == "CartPole-v1":
            self.max_steps_per_episode = 500
        elif self.env_name == "CartPole-v0":
            self.max_steps_per_episode = 200
        else:
            self.max_steps_per_episode = 0
        self.reward_std = hyperparameters["reward_std"]
        self.use_exp_rep = hyperparameters["use_exp_rep"]
        self.n_initial_random_actions = hyperparameters["n_initial_random_actions"]

        # copied from Old class:
        self.state_len = len(self.env.observation_space.high)
        self.normalize_observations = hyperparameters["normalize_obs"]
        if self.normalize_observations:
            self.normalizer = Normalizer(self.state_len)
        else:
            self.normalizer = None

        # Init Policy:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Agent(self.env, self.device, self.normalizer, self.log, hyperparameters)

    def reset(self):
        self.steps_done = 0
        self.episode_durations = []
        self.policy.reset()

    def optimize(self):
        self.policy.optimize()

    def modify_env_reward(self, reward):
        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        if self.reward_std:
            reward += torch.tensor(np.random.normal(0, self.reward_std))
        return reward

    # TODO: atm the replay buffer is filled with the agent acting with a normal policy, not fully random. This should probably be changed
    def fill_replay_buffer(self, n_actions):
        state = self.env.reset()
        state = torch.tensor([state], device=device).float()

        # Fill exp replay buffer so that we can start training immediately:
        for _ in range(n_actions):
            action, next_state, reward, done = self._act(self.env, state, store_in_exp_rep=True, render=False, explore=True)

            state = next_state
            if done:
                state = self.env.reset()
                state = torch.tensor([state], device=device).float()

    def _act(self, env, state, explore=True, render=False, store_in_exp_rep=True):
        # Select an action
        if explore:
            action = self.policy.explore(state)
        else:
            action = self.policy.exploit(state)
        # Apply the action:
        next_state, reward, done, _ = env.step(action)
        # Add possible noise to the reward:
        reward = self.modify_env_reward(reward)
        # Define next state in case it is terminal:
        if done:
            next_state = None
        else:
            next_state = torch.tensor([next_state], device=device).float()
            #next_state = torch.tensor([next_state], device=self.device).float()
            # Record state for normalization:
            if self.normalize_observations:
                self.normalizer.observe(next_state)
        # Store the transition in memory
        if self.use_exp_rep and store_in_exp_rep:
            self.policy.remember(state, torch.tensor([action], device=device).float(), next_state, reward, done)
        # Calculate TDE for debugging purposes:
        TDE = self.policy.calculate_TDE(state, action, next_state, reward, done)
        self.log.add("TDE", TDE.item())
        # Render:
        if render:
            self.env.render()

        return action, next_state, reward, done

    def _act_in_test_env(self, test_env, test_state, test_episode_rewards):
        _, next_state, reward, done = self._act(test_env, test_state, explore=False, store_in_exp_rep=False, render=False)

        test_episode_rewards.append(reward)
        self.log.add("Test_Env Reward", np.sum(test_episode_rewards))
        if done or (self.max_steps_per_episode > 0 and len(test_episode_rewards) >= self.max_steps_per_episode):
            next_state = test_env.reset()
            next_state = torch.tensor([next_state], device=self.device).float()
            test_episode_rewards.clear()

        return next_state

    def _display_debug_info(self, i_episode, episode_rewards, rewards):
        print("Episode ", i_episode)
        print("Steps taken so far: ", self.steps_done)
        print("Cumulative Reward this episode:", np.sum(episode_rewards))
        if i_episode % 10 == 0:
            pass
            # Not needed anymore, because he have tensorboard now
            #plot_rewards(rewards)
            #plot_rewards(self.log.storage["Test_Env Reward"], "Test_Env Reward")
            #plot_rewards(self.log.storage["Return"], "Return", xlabel="Episodes")

        self.policy.display_debug_info()
        print()

    def run(self, n_steps, verbose=False, render=False, on_server=True):
        # Fill replay buffer with random actions:
        self.fill_replay_buffer(n_actions=self.n_initial_random_actions)

        # Initialize test environment:
        test_env = gym.make(self.env_name).unwrapped
        test_state = test_env.reset()
        test_state = torch.tensor([test_state], device=device).float()

        test_episode_rewards = []

        # Do the actual training:
        rewards = []
        i_episode = 0
        run = True
        while run:
            i_episode += 1
            # Initialize the environment and state
            state = self.env.reset()
            state = torch.tensor([state], device=device).float()
            episode_rewards = []

            for t in count():
                self.steps_done += 1
                if not verbose and not on_server:
                    print("Episode loading:  " + str(round(self.steps_done / n_steps * 100, 2)) + "%", end='\r')

                # Act in exploratory env:
                action, next_state, reward, done = self._act(self.env, state, render=render, store_in_exp_rep=True,
                                                             explore=True)

                # Act in test env (no exploration in that env):
                test_state = self._act_in_test_env(test_env, test_state, test_episode_rewards)

                # Move to the next state
                state = next_state

                # Reduce epsilon and other exploratory values:
                self.policy.decay_exploration(self.steps_done)

                # Perform one step of the optimization (on the target network)
                self.policy.optimize()

                # Update the target network
                self.policy.update_targets(self.steps_done)

                # Log reward:
                self.log.step()
                episode_rewards.append(reward.item())
                rewards.append(np.sum(episode_rewards))

                if render:
                    self.env.render()

                if done or (self.max_steps_per_episode > 0 and t >= self.max_steps_per_episode) \
                        or self.steps_done >= n_steps:
                    if verbose:
                        self._display_debug_info(i_episode, episode_rewards, rewards)
                    self.log.add("Return", np.sum(episode_rewards), steps=i_episode)
                    if self.steps_done >= n_steps:
                        run = False
                    break
        print('Done.')
        self.env.close()
        return i_episode, rewards, self.log.storage


# TODO: just pass parameter object around and extract needed ones in the respective classes
class TrainerOld(object):
    def __init__(self, env_name, device, USE_QV=False, SPLIT_BELLMAN=False, gamma_Q=0.99,
                 batch_size=64, UPDATES_PER_STEP=1, target_network_steps=500, lr_Q=0.001, lr_r=0.001,
                 replay_buffer_size=10000, USE_EXP_REP=True, epsilon_mid=0.1, activation_function="elu",
                 SPLIT_BELL_additional_individual_hidden_layer=False, SPLIT_BELL_NO_TARGET_r=True, hidden_neurons=64,
                 hidden_layers=1, MAX_EPISODE_STEPS=0, initial_random_actions=1024,
                 QV_NO_TARGET_Q=False, QV_SPLIT_Q=False, QV_SPLIT_V=False, QVC_TRAIN_ABS_TDE=False,
                 TDEC_ENABLED=False, TDEC_TRAIN_FUNC="normal", TDEC_ACT_FUNC="abs", TDEC_SCALE=0.5, TDEC_MID=0,
                 TDEC_USE_TARGET_NET=True, TDEC_GAMMA=0.99, TDEC_episodic=True,
                 normalize_observations=True, critic_output_offset=0, reward_added_noise_std=0,
                 action_gaussian_noise_std=0.1, USE_CACLA=False, USE_OFFLINE_CACLA=False, actor_lr=0.001):

        self.steps_done = 0
        self.rewards = []
        self.log = Log()

        self.device = device
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        num_initial_states = 5
        self.initial_states = torch.tensor([self.env.reset() for _ in range(num_initial_states)], device=self.device,
                                           dtype=torch.float)

    def reset(self):
        self.steps_done = 0
        self.episode_durations = []

        self.policy.reset()

    def initialize_workers(self):
        self.workers = [{"env": gym.make(self.env_name).unwrapped, "episode length": 0} for i in range(self.batch_size)]
        for worker in self.workers:
            worker["state"] = worker["env"].reset()
            worker["state"] = torch.tensor([worker["state"]], device=self.device).float()

    def collect_experiences(self):
        # Do one step with each worker and return transition batch
        transition_list = []
        dones = 0
        for idx in range(len(self.workers)):
            worker = self.workers[idx]
            worker_env = worker["env"]
            worker_state = worker["state"]
            action = self.select_action(worker_state, self.epsilon)
            worker["episode length"] += 1
            next_state, reward, done, _ = worker_env.step(action.item())

            if done:
                next_state = None
            else:
                next_state = torch.tensor([next_state], device=self.device).float()
            reward = torch.tensor([reward], device=self.device, dtype=torch.float)
            if self.TDEC_ENABLED:
                TDE = self.calculateTDE(self.workers[idx]["state"], action, next_state, reward)
            else:
                TDE = None

            trans = Transition(self.workers[idx]["state"], action, next_state, reward, TDE)
            transition_list.append(trans)

            if done or worker["episode length"] > self.max_steps_per_episode > 0:
                dones += 1
                worker["episode length"] = 0
                worker["state"] = worker["env"].reset()
                worker["state"] = torch.tensor([worker["state"]], device=self.device).float()
            else:
                worker["state"] = next_state

        return transition_list

    def calculate_initial_state_val(self):
        with torch.no_grad():
            predictions_inital_states = self.target_net(self.initial_states).view(-1, self.num_actions,
                                                                                  self.num_Q_output_slots)
            initial_state_value = torch.mean(predictions_inital_states[:, :, 1]).item()
        return initial_state_value

    # should not be needed anymore: check if it is used anywhere before deleting
    # def getActionIdxs(self, action_batch):
    #    return torch.cat([action_batch * self.num_Q_output_slots + i for i in range(self.num_Q_output_slots)], dim=1)

    def optimize_model(self):
        self.policy.optimize()

    def run(self, num_steps=5000, verbose=True, render=False, on_server=False):
        eps_decay = self.epsilon_mid ** (1 / (num_steps / 2))
        self.epsilon = 1.0
        QVC_decay = self.QV_CURIOSITY_MID ** (1 / (num_steps / 2.0))
        TDEC_DECAY = self.TDEC_MID ** (1 / (num_steps / 2.0))

        self.testEnv = gym.make(self.env_name).unwrapped
        self.testState = self.testEnv.reset()
        self.testState = torch.tensor([self.testState], device=self.device).float()
        self.test_episode_rewards = []

        state = self.env.reset()
        state = torch.tensor([state], device=self.device).float()

        # Fill exp replay buffer so that we can start training immediately:
        for i in range(self.initial_random_actions):
            action = self.select_action(state, 1)
            next_state, reward, done, _ = self.env.step(action)
            if done:
                next_state = None
            else:
                next_state = torch.tensor([next_state], device=self.device).float()
            reward = torch.tensor([reward], device=self.device, dtype=torch.float)

            TDE = self.calculateTDE(state, action, next_state, reward, store_log=False)

            if self.reward_added_noise_std:
                reward += torch.tensor(np.random.normal(0, self.reward_added_noise_std))

            if self.USE_EXP_REP:
                self.memory.push(state, action, next_state, reward, TDE)

            state = next_state

            if done:
                state = self.env.reset()
                state = torch.tensor([state], device=self.device).float()

        # Do the actual training:
        i_episode = 0
        run = True
        while run:
            i_episode += 1
            # Initialize the environment and state
            state = self.env.reset()
            state = torch.tensor([state], device=self.device).float()
            episode_rewards = []

            for t in count():
                if not verbose and not on_server:
                    print("Current setting test: ", round(self.steps_done / num_steps * 100, 2), "%", end='\r')
                # Stop when max number of steps is reached
                if self.steps_done >= num_steps:
                    run = False
                    break

                self.act_in_test_env()

                # Select and perform an action
                action = self.policy.select_action(state)
                self.steps_done += 1
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor([next_state], device=self.device).float()

                reward = torch.tensor([reward], device=self.device, dtype=torch.float)
                episode_rewards.append(reward.item())
                self.rewards.append(np.sum(episode_rewards))

                if render:
                    self.env.render()

                TDE = self.calculateTDE(state, action, next_state, reward)

                # Store the transition in memory
                if self.USE_EXP_REP:
                    self.memory.push(state, action, next_state, reward, TDE)

                # Move to the next state
                state = next_state

                self.epsilon *= eps_decay
                self.log.add("Epsilon", self.epsilon)


            # Optimize the policy
            self.policy.optimize()

            if done or (self.max_steps_per_episode > 0 and t >= self.max_steps_per_episode):
                if verbose:
                    print("Episode ", i_episode)
                    print("Steps taken so far: ", self.steps_done)
                    print("Cumulative Reward this episode:", np.sum(episode_rewards))
                    if i_episode % 10 == 0:
                        plot_rewards(self.rewards)
                        plot_rewards(self.log.storage["Total Reward"], "Total Reward")
                    print("Epsilon: ", round(self.epsilon, 4))

                    print()
                break
            # Update the target network
            for net in self.networks:
                net.update_target_network(self.steps_done)

        # if self.steps_done % self.target_network_steps == 0:
        #    if __debug__ and verbose:
        #        print("Updating Target Network")
        #    if self.USE_QV:
        #        self.target_value_net.load_state_dict(self.V_net.state_dict())
        #    else:
        #        self.target_net.load_state_dict(self.Q_net.state_dict())

    # if verbose:
    #     plot_rewards(self.rewards)
    #     plot_rewards(self.log.storage["Total Reward"], "Total Reward")
    #     print('Complete')
    #     self.env.close()


def calculate_reduced_idxs(len_of_point_list, max_points):
    if max_points != 0:
        step_size = len_of_point_list // max_points
        step_size += 1 if len_of_point_list % max_points else 0
    else:
        return range(len_of_point_list)
    return range(0, len_of_point_list, step_size)


def reducePoints(list_of_points, max_points_per_line):
    if max_points_per_line != 0:
        step_size = len(list_of_points) // max_points_per_line
        step_size += 1 if len(list_of_points) % max_points_per_line else 0
    else:
        return range(len(list_of_points)), list_of_points
    steps = range(0, len(list_of_points), step_size)
    list_of_points = [np.mean(list_of_points[i:i + step_size]) for i in steps]
    return list_of_points


def mean_final_percent(result_list, percentage=0.1):
    final_percent_idx = int(len(result_list) * (1 - percentage))
    return np.mean(result_list[final_percent_idx:])


def run_metric(result_list, percentage=0.1, final_percentage_weight=1):
    return np.mean(result_list) * (1 - final_percentage_weight) + mean_final_percent(result_list,
                                                                                     percentage) * final_percentage_weight


def testSetup(env, device, number_of_tests, length_of_tests, trialParams, randomizeList=[], on_server=False,
              max_points=2000, hyperparamDict={}, verbose=True):
    results_len = []
    results = []
    logs = []

    for i in range(number_of_tests):
        if len(randomizeList) > 0:
            randomizedParams = randomizeList[i]
            print("Hyperparameters:")
            for key in randomizedParams:
                print(key + ":", randomizedParams[key], end=" ")
            print()
        else:
            randomizedParams = {}

        trainer = Trainer(env, device, **trialParams, **randomizedParams, **hyperparamDict)
        steps, rewards, log = trainer.run(verbose=False, n_steps=length_of_tests, on_server=on_server)

        rewards = reducePoints(rewards, max_points)
        for key in log:
            log[key] = reducePoints(log[key], max_points)

        results_len.append(len(rewards))
        results.append(rewards)
        logs.append(log)

        if number_of_tests > 1:
            print("Run ", str(i), "/", str(number_of_tests), end=" | ")
            print("Mean reward ", round(np.mean(rewards), 1), end=" ")
            print("Score: ", round(run_metric(log["Total Reward"]), 1))
    return results, logs


if __name__ == "__main__":
    # TODO: here we could declare functions for certain events that we pass as parameters. For MineRL we could define how the observation is split into matrix and vector and how to deal with the action space

    standard_feature_block = [{"name": "linear", "neurons": 256, "act_func": "relu"},
                              {"name": "linear", "neurons": 128}]
    standard_hidden_block =  [{"name":"linear", "neurons": 64, "act_func": "relu"},
                             {"name":"linear", "neurons": 64, "act_func": "relu"}]
    layers_feature_vector = standard_hidden_block
    layers_feature_merge = standard_feature_block
    layers_r = standard_hidden_block
    layers_Q = standard_hidden_block

    parameters = {"use_QV": False, "split_Bellman": True, "gamma": 1, "batch_size": 64, "UPDATES_PER_STEP": 1,
                  "use_QVMAX": False, "use_target_net": True,
                  "target_network_hard_steps": 250, "use_polyak_averaging":False, "polyak_averaging_tau":0.001,
                  "lr_Q": 0.001, "lr_r": 0.001,
                  "replay_buffer_size": 10000, "use_PER": False, "PER_alpha": 0.6, "PER_beta": 0.4,
                  "use_CER": True,
                  "use_exp_rep": True,
                  "epsilon": 0.1, "epsilon_decay": 0, "action_sigma": 0, "epsilon_mid": 0.1, "boltzmann_temp": 0,
                  "SPLIT_BELL_NO_TARGET_r": True,
                  "n_initial_random_actions": 100, "QV_NO_TARGET_Q": False,
                  "TDEC_ENABLED": False, "TDEC_TRAIN_FUNC": "normal",
                  "TDEC_ACT_FUNC": "abs",
                  "TDEC_SCALE": 0.5, "TDEC_MID": 0, "TDEC_USE_TARGET_NET": True, "TDEC_GAMMA": 0.99,
                  "TDEC_episodic": True,
                  "normalize_obs": False,
                  "reward_std": 0.0, "use_actor_critic":False, "use_CACLA_V": False, "use_CACLA_Q": False, "use_DDPG":False,
                  "use_SPG": False, "use_GISPG":False, "lr_actor": 0.001,
                  "max_episode_steps": 0, "use_hrl": False, "layers_feature_vector": layers_feature_vector,
                  "layers_feature_merge": layers_feature_merge, "layers_r": layers_r, "layers_Q": layers_Q,
                  "use_world_model": False, "max_norm":1,
                  "use_REM": True, "REM_num_heads": 50, "REM_num_samples": 5}
    # TODO: why does normalize_obs destroy the whole training for cartpole????
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lunar = "LunarLander-v2"
    cart = "CartPole-v1"
    acro = "Acrobot-v1"
    mountain = "MountainCar-v0"
    pendulum = "Pendulum-v0"
    mountain_cont = "MountainCarContinuous-v0"

    # print("Action space: ", env.action_space)
    # print("Observation space: ", env.observation_space)

    # trainer = Trainer(environment_name, device)

    trainer = Trainer(lunar, parameters, log=False)
    trainer.run(50000, render=False, verbose=True)
