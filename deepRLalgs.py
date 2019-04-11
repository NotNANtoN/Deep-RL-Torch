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


def display_top_memory_users(key_type='lineno', limit=3, censored=True):
    snapshot = tracemalloc.take_snapshot()
    if censored:
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        print("Top %s memory usage lines" % limit)
    else:
        limit = 0
    top_stats = snapshot.statistics(key_type)

    print(("Unc" if not censored else "C") + "ensored memory usage:")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)
    other = top_stats[limit:]
    if other and censored:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
    print()


########## Setup #################
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'TDEC'))

########################################
'''
# Not a good function, biases the end totally.
def meanSmoothing(input_list, window_size):
    assert window_size % 2 == 1 and window_size > 2 and len(input_list) > 0
    numbers_to_add = (window_size - 1) // 2
    # Pad list by repeating first and last element
    padded_list = [input_list[0] for i in range(numbers_to_add)]
    end_padding = [input_list[-1] for i in range(numbers_to_add)]
    padded_list.extend(input_list)
    padded_list.extend(end_padding)
    smoothed_list = []
    for number in range(len(padded_list) - 2 * numbers_to_add):
        window = padded_list[number:number+window_size]
        smoothed_list.append(np.mean(window))
    return np.array(smoothed_list)
'''

class Normalizer():
    def __init__(self, num_inputs):
        self.n = 0
        self.mean = torch.zeros(num_inputs)
        self.mean_diff = torch.zeros(num_inputs)
        self.var = torch.zeros(num_inputs)

    def observe(self, x):
        x = x.view(-1)
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean)/obs_std

def meanSmoothing(x, N):
    x = np.array(x)
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N % 2 == 0:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 2
        else:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 1
        # cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


def plot_rewards(rewards, env, name=None):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward of current Episode')
    # if env == "CartPole-v1":
    #    plt.ylim(0, 500)
    # elif env == "LunarLander-v2":
    #    plt.ylim(-250, 250)
    # elif env == "Acrobot-v1":
    #    plt.ylim(0, 2000)
    # elif env == "MountainCar-v0":
    #    plt.ylim(0, 3000)
    # else:
    #    print("No ylim defined for env ", env)
    #    quit()
    idxs = calculate_reduced_idxs(len(rewards), 1000)
    rewards = reducePoints(rewards, 1000)

    plt.plot(idxs, rewards)
    # Apply mean-smoothing and plot result
    window_size = len(rewards) // 10
    window_size += 1 if window_size % 2 == 0 else 0
    means = meanSmoothing(rewards, window_size)
    max_val = np.max(means)
    min_val = np.min(means)
    plt.ylim(min_val, max_val * 1.1)
    plt.plot(idxs, means)
    if name is None:
        plt.savefig("current_test.pdf")
    else:
        plt.savefig(name + "_current.pdf")
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Log(object):
    def __init__(self):
        self.storage = {}

    def add(self, name, value):
        try:
            self.storage[name].append(value)
        except KeyError:
            self.storage[name] = [value]

    def get(self):
        return self.storage


class Q(nn.Module):
    #  Also: (needs to be added to the networks)
    #    1. Add option to add one hidden layer in general
    #    2. Add option to make r and R prediction use separate nets
    #    3. Add option to let r and R prediction be in same net, but have their own hidden layer
    def __init__(self, state_len, num_actions, outputs_per_action, use_separate_nets=False,
                 additional_individual_hidden_layer=False, HIDDEN_NEURONS=64, HIDDEN_LAYERS=2,
                 activation_function=F.relu, normalizer=None, offset=-1):
        super(Q, self).__init__()
        self.use_separate_nets = use_separate_nets
        self.additional_individual_hidden_layer = additional_individual_hidden_layer
        self.num_actions = num_actions
        self.HIDDEN_LAYERS = HIDDEN_LAYERS - additional_individual_hidden_layer
        self.activation_function = activation_function
        self.normalizer = normalizer
        self.offset = offset

        if self.use_separate_nets:
            self.nets = nn.ModuleList()
            for i in range(outputs_per_action):
                layers = nn.ModuleList()
                layers.append(nn.Linear(state_len, HIDDEN_NEURONS))
                layers.append(nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS))
                layers.append(nn.Linear(HIDDEN_NEURONS, num_actions))
                self.nets.append(layers)

        else:
            self.hidden1 = nn.Linear(state_len, HIDDEN_NEURONS)
            if self.HIDDEN_LAYERS > 1:
                self.hidden2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
            if self.HIDDEN_LAYERS > 2:
                self.hidden3 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
            if self.HIDDEN_LAYERS > 3:
                print("ERROR in creating Q network: Only 3 hidden layers supported. Network was created with 2 layers")
            if self.additional_individual_hidden_layer:
                self.separate_parts = nn.ModuleList()
                for i in range(outputs_per_action):
                    layers = nn.ModuleList()
                    layers.append(nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS))
                    layers.append(nn.Linear(HIDDEN_NEURONS, num_actions))
                    self.separate_parts.append(layers)
            else:
                self.head = nn.Linear(HIDDEN_NEURONS, num_actions * outputs_per_action)

    def forward(self, x):
        if self.normalizer is not None:
            x = self.normalizer.normalize(x)

        if self.use_separate_nets:
            outputs = []
            for net in self.nets:
                y = x
                for layer in net:
                    y = self.activation_function(layer(y)) if layer is not net[-1] else layer(y)
                outputs.append(y)
            output = torch.cat(outputs, dim=1)
        else:
            x = self.activation_function(self.hidden1(x))
            if self.HIDDEN_LAYERS > 1:
                x = self.activation_function(self.hidden2(x))
            if self.HIDDEN_LAYERS > 2:
                x = self.activation_function(self.hidden3(x))
            if self.additional_individual_hidden_layer:
                outputs = []
                for part in self.separate_parts:
                    y = self.activation_function(part[0](x))
                    y = part[1](y)
                    outputs.append(y)
                output = torch.cat(outputs, dim=1)
            else:
                output = self.head(x)
        return output + self.offset


class V(nn.Module):
    def __init__(self, state_len, output_len, HIDDEN_NEURONS=64, HIDDEN_LAYERS=2, activation_function=F.relu, normalizer=None,offset=-1):
        super(V, self).__init__()
        self.HIDDEN_LAYERS = HIDDEN_LAYERS
        self.activation_function = activation_function
        self.normalizer = normalizer
        self.offset = offset

        self.hidden1 = nn.Linear(state_len, HIDDEN_NEURONS)
        if self.HIDDEN_LAYERS > 1:
            self.hidden2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        if self.HIDDEN_LAYERS > 2:
            self.hidden3 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        if self.HIDDEN_LAYERS > 3:
            print("ERROR in creating V network: Only 3 hidden layers supported. Network was created with 2 layers")
        self.head = nn.Linear(HIDDEN_NEURONS, output_len)

    def forward(self, x):
        if self.normalizer is not None:
            x = self.normalizer.normalize(x)

        x = self.activation_function(self.hidden1(x))
        if self.HIDDEN_LAYERS > 1:
            x = self.activation_function(self.hidden2(x))
        if self.HIDDEN_LAYERS > 2:
            x = self.activation_function(self.hidden3(x))
        return self.head(x) + self.offset


# "activation_function": hp.choice("activation_function", ["sigmoid", "relu", "elu"]),
#                "gamma_Q": hp.uniform("gamma_Q", 0.9, 0.9999),
#                "lr_Q": hp.loguniform("lr_Q", np.log(0.01), np.log(0.00001)),
#                "target_network_steps": hp.quniform("target_network_steps", 10, 20000, 1),
#                "hidden_neurons": hp.quniform("hidden_neurons", 32, 256, 1),
#                "hidden_layers": hp.quniform("hidden_layers", 1, 3, 1),
#                "batch_size": hp.quniform("batch_size", 16, 256, 1),
#                "replay_buffer_size": hp.quniform("replay_buffer_size", 1024, 100000, 1)],
#                "epsilon_mid": hp.uniform("epsilon_mid", 0.25, 0.0001) 

class Trainer(object):
    def __init__(self, env_name, device, USE_QV=False, SPLIT_BELLMAN=False, gamma_Q=0.99,
                 batch_size=64, UPDATES_PER_STEP=1, target_network_steps=500, lr_Q=0.001, lr_r = 0.001,
                 replay_buffer_size=10000,
                 QV_CURIOSITY_ENABLED=False, QV_CURIOSITY_SCALE=0.5, QV_CURIOSITY_TWO_HEADS=False,
                 QVC_USE_ABS_FOR_ACTION=False, QV_CURIOSITY_USE_TARGET_NET=True, USE_EXP_REP=True,
                 QV_CURIOSITY_MID=0.0, epsilon_mid=0.1, activation_function="elu",
                 SPLIT_BELL_use_separate_nets=False, SPLIT_BELL_additional_individual_hidden_layer=False,
                 SPLIT_BELL_AVG_r=False,
                 SPLIT_BELL_NO_TARGET_r=True, SPLIT_BELL_NO_TARGET_AT_ALL=False, hidden_neurons=64, hidden_layers=1,
                 MAX_EPISODE_STEPS=0, initial_random_actions=1024,
                 QV_NO_TARGET_Q=False, QV_SPLIT_Q=False, QV_SPLIT_V=False, QVC_TRAIN_ABS_TDE=False,
                 TDEC_ENABLED=False, TDEC_TRAIN_FUNC="normal", TDEC_ACT_FUNC="abs", TDEC_SCALE=0.5, TDEC_MID=0,
                 TDEC_USE_TARGET_NET=True, TDEC_GAMMA=0.99, TDEC_episodic=True,
                 normalize_observations=True, critic_output_offset=0):
                 
        self.steps_done = 0
        self.rewards = []
        self.log = Log()
        
        self.device = device
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        num_initial_states = 5
        self.initial_states = torch.tensor([self.env.reset() for _ in range(num_initial_states)], device=self.device,
                                           dtype=torch.float)
        self.MAX_EPISODE_STEPS = MAX_EPISODE_STEPS
        self.initial_random_actions = initial_random_actions

        self.num_actions = self.env.action_space.n
        self.state_len = len(self.env.observation_space.high)

        self.normalize_observations = normalize_observations
        if normalize_observations:
            self.normalizer = Normalizer(self.state_len)
        else:
            self.normalizer = None

        self.critic_output_offset = critic_output_offset
        if activation_function == "relu" or activation_function == 1:
            self.activation_function = F.relu
        elif activation_function == "sigmoid" or activation_function == 0:
            self.activation_function = torch.sigmoid
        elif activation_function == "elu" or activation_function == 2:
            self.activation_function = F.elu
        elif activation_function == "selu" or activation_function == 3:
            self.activation_function = F.selu
        self.gamma_Q = gamma_Q
        self.batch_size = batch_size
        self.UPDATES_PER_STEP = UPDATES_PER_STEP
        self.target_network_steps = target_network_steps
        self.EPS_START = 1
        self.epsilon_mid = epsilon_mid
        self.lr_Q = lr_Q
        self.hidden_neurons = hidden_neurons
        self.hidden_layers = hidden_layers
        self.replay_buffer_size = replay_buffer_size
        self.USE_EXP_REP = USE_EXP_REP

        self.SPLIT_BELLMAN = SPLIT_BELLMAN
        self.SPLIT_BELL_use_separate_nets = SPLIT_BELL_use_separate_nets
        self.SPLIT_BELL_additional_individual_hidden_layer = SPLIT_BELL_additional_individual_hidden_layer
        self.SPLIT_BELL_NO_TARGET_r = SPLIT_BELL_NO_TARGET_r
        self.SPLIT_BELL_AVG_r = SPLIT_BELL_AVG_r
        self.SPLIT_BELL_NO_TARGET_AT_ALL = SPLIT_BELL_NO_TARGET_AT_ALL
        self.lr_r = lr_r

        self.USE_QV = USE_QV
        self.QV_SPLIT_Q = QV_SPLIT_Q
        self.QV_SPLIT_V = QV_SPLIT_V
        self.QV_NO_TARGET_Q = QV_NO_TARGET_Q

        self.TDEC_USE_TARGET_NET = TDEC_USE_TARGET_NET
        self.TDEC_ENABLED = TDEC_ENABLED
        self.TDEC_SCALE = TDEC_SCALE
        self.TDEC_MID = TDEC_MID
        self.TDEC_ACT_FUNC = TDEC_ACT_FUNC
        self.TDEC_TRAIN_FUNC = TDEC_TRAIN_FUNC
        self.TDEC_GAMMA = TDEC_GAMMA
        self.TDEC_episodic = TDEC_episodic

        self.QVC_TRAIN_ABS_TDE = QVC_TRAIN_ABS_TDE
        self.QV_CURIOSITY_ENABLED = QV_CURIOSITY_ENABLED
        self.QV_CURIOSITY_SCALE = QV_CURIOSITY_SCALE
        self.QV_CURIOSITY_TWO_HEADS = QV_CURIOSITY_TWO_HEADS
        self.QV_CURIOSITY_SPLIT = False
        self.QVC_USE_ABS_FOR_ACTION = QVC_USE_ABS_FOR_ACTION
        self.QV_CURIOSITY_USE_TARGET_NET = QV_CURIOSITY_USE_TARGET_NET
        self.QV_CURIOSITY_MID = QV_CURIOSITY_MID
        if self.QV_CURIOSITY_MID:
            self.QVC_scale = 1
        else:
            self.QVC_scale = self.QV_CURIOSITY_SCALE
        self.num_Q_inputs = self.state_len
        self.num_Q_output_slots = (1 + SPLIT_BELLMAN + TDEC_ENABLED + (USE_QV and QV_SPLIT_Q) + (
                    USE_QV and QV_SPLIT_V and QV_SPLIT_Q)) + (
                                              USE_QV and QV_CURIOSITY_ENABLED and QV_CURIOSITY_TWO_HEADS)
        self.num_V_outputs = 1 + QV_SPLIT_V

        self.reset()

    def reset(self):
        self.steps_done = 0
        self.episode_durations = []
        self.Q_net = Q(self.num_Q_inputs, self.num_actions, self.num_Q_output_slots,
                       use_separate_nets=self.SPLIT_BELL_use_separate_nets,
                       additional_individual_hidden_layer=self.SPLIT_BELL_additional_individual_hidden_layer,
                       HIDDEN_NEURONS=self.hidden_neurons,
                       HIDDEN_LAYERS=self.hidden_layers, activation_function=self.activation_function,
                       normalizer=self.normalizer, offset=self.critic_output_offset).to(self.device)
        if self.USE_QV:
            self.V_net = V(self.state_len, self.num_V_outputs,
                           HIDDEN_NEURONS=self.hidden_neurons,
                           HIDDEN_LAYERS=self.hidden_layers, activation_function=self.activation_function,
                       normalizer=self.normalizer, offset=self.critic_output_offset).to(
                self.device)
            self.target_value_net = V(self.state_len, self.num_V_outputs,
                                      HIDDEN_NEURONS=self.hidden_neurons,
                                      HIDDEN_LAYERS=self.hidden_layers,
                                      activation_function=self.activation_function,
                       normalizer=self.normalizer, offset=self.critic_output_offset).to(self.device)
            self.target_value_net.load_state_dict(self.V_net.state_dict())
            self.target_value_net.eval()
            self.value_optimizer = optim.Adam(self.V_net.parameters(), lr=self.lr_Q)

        self.target_net = Q(self.num_Q_inputs, self.num_actions, self.num_Q_output_slots,
                            use_separate_nets=self.SPLIT_BELL_use_separate_nets,
                            additional_individual_hidden_layer=self.SPLIT_BELL_additional_individual_hidden_layer,
                            HIDDEN_NEURONS=self.hidden_neurons,
                            HIDDEN_LAYERS=self.hidden_layers, activation_function=self.activation_function,
                       normalizer=self.normalizer, offset=self.critic_output_offset).to(
            self.device)
        self.target_net.load_state_dict(self.Q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=self.lr_Q)
        self.optimizer_r = optim.Adam(self.Q_net.parameters(), lr=self.lr_r)
        if self.USE_EXP_REP:
            self.memory = ReplayMemory(self.replay_buffer_size)
        else:
            self.initialize_workers()

        if not self.TDEC_MID:
            self.TDEC_FACTOR = self.TDEC_SCALE
        else:
            self.TDEC_FACTOR = 1

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

    def select_action(self, state, eps):
        if self.normalize_observations:
            self.normalizer.observe(state)

        sample = random.random()
        if sample > eps:
            with torch.no_grad():
                predictions = self.Q_net(state).view(self.num_actions, self.num_Q_output_slots)

                if self.TDEC_ENABLED:
                    predicted_state_action_values = predictions[:, [0]]
                    predicted_TDEs = predictions[:, [1]]
                    if self.TDEC_ACT_FUNC == "absolute":
                        predicted_TDEs = abs(predicted_TDEs)
                    elif self.TDEC_ACT_FUNC == "positive":
                        predicted_TDEs = (predicted_TDEs + abs(predicted_TDEs)) / 2
                    elif self.TDEC_ACT_FUNC == "mse":
                        predicted_TDEs = predicted_TDEs ** 2
                    predictions = self.TDEC_FACTOR * predicted_TDEs + (
                                1 - self.TDEC_FACTOR) * predicted_state_action_values

                if self.SPLIT_BELLMAN:
                    if __debug__:
                        predicted_rewards = predictions[:, [0]]
                        predicted_values = predictions[:, [1]]
                        print("Predicted rewards: ", predicted_rewards)
                        print("Predicted values: ", predicted_values)
                    predictions = predictions.sum(dim=1)
                    predictions = predictions.view(1, self.num_actions)

                    # predictions = predicted_rewards + predicted_values
                if self.USE_QV:
                    if self.QV_SPLIT_Q:
                        if self.QV_SPLIT_V:
                            predicted_r = predictions[:, [0]]
                            predicted_r_V = predictions[:, [1]]
                            predicted_R_V = predictions[:, [2]]
                            predictions = predicted_r + predicted_r_V + predicted_R_V
                        else:
                            predicted_r = predictions[:, [0]]
                            predicted_V_s_prime = predictions[:, [1]]
                            predictions = predicted_r + predicted_V_s_prime

                    if self.QV_CURIOSITY_ENABLED:
                        if self.QV_CURIOSITY_TWO_HEADS:
                            predicted_state_action_values = predictions[:, [0]]
                            predicted_TD_errors = predictions[:, [1]]

                            if self.QVC_USE_ABS_FOR_ACTION == "absolute":
                                predicted_TD_errors = predicted_TD_errors.abs()
                            predictions = predicted_state_action_values * (
                                        1 - self.QVC_scale) + predicted_TD_errors * self.QVC_scale

                predictions = predictions.view(1, self.num_actions)
                return predictions.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)

    # Only for QV learning:
    def expectedVals(self, net, non_final_next_states, non_final_mask, reward_batch):
        with torch.no_grad():
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            next_state_values[non_final_mask] = net(non_final_next_states).view(-1)
            # Compute the expected V values of the next state
            expected_state_action_values = (next_state_values * self.gamma_Q) + reward_batch
        return expected_state_action_values.unsqueeze(1)

    def add_to_prediction_dict(name, prediction_dict, predictions, idx, next_state):
        if next_state:
            prediction_dict[name] = predictions[:, :, 0]
        else:
            prediction_dict[name] = predictions[:, [0]]
        idx += 1
        return idx

    def transform_Q_prediction_to_dict(self, predictions, action_batch=None):
        prediction_dict = {}
        next_state = action_batch is None
        if not next_state:
            correct_indices = self.getActionIdxs(action_batch)
            predictions = predictions.gather(1, correct_indices)
        else:
            predictions = predictions.view(-1, self.num_actions, self.num_Q_output_slots)
        return predictions

        # TODO: enable the lower part to woork
        idx = 0
        if self.USE_QV:
            if self.QV_SPLIT_Q:
                idx = add_to_prediction_dict("r_Q", prediction_dict, predictions, idx, next_state)
                if self.QV_SPLIT_V:
                    idx = add_to_prediction_dict("r_V", prediction_dict, predictions, idx, next_state)
                    idx = add_to_prediction_dict("R_V", prediction_dict, predictions, idx, next_state)
                else:
                    idx = add_to_prediction_dict("R_Q", prediction_dict, predictions, idx, next_state)
            else:
                idx = add_to_prediction_dict("Q", prediction_dict, predictions, idx, next_state)
                if self.TDEC_ENABLED and not self.TDEC_ON_V:
                    idx = add_to_prediction_dict("TDEC", prediction_dict, predictions, idx, next_state)

        else:
            if self.SPLIT_BELLMAN:
                idx = add_to_prediction_dict("r_Q", prediction_dict, predictions, idx, next_state)
                idx = add_to_prediction_dict("R_Q", prediction_dict, predictions, idx, next_state)

                # if self.TDEC_ENABLED:
                #    idx = add_to_prediction_dict("TDEC", prediction_dict, predictions, idx, next_state)
                #   # ooor:
                #    idx = add_to_prediction_dict("r_TDE", prediction_dict, predictions, idx, next_state)
            else:
                idx = add_to_prediction_dict("Q", prediction_dict, predictions, idx, next_state)

                if self.TDEC_ENABLED:
                    idx = add_to_prediction_dict("TDEC", prediction_dict, predictions, idx, next_state)

        # (1 + SPLIT_BELLMAN + TDEC_ENABLED +  (USE_QV and QV_SPLIT_Q) + (USE_QV and QV_SPLIT_V and QV_SPLIT_Q))

        # 

        # TODO: later return dict: return prediction_dict

        # current state: .gather(1, correct_indices), predictions[:, [0]]
        # next_state : .view(-1, self.num_actions, 2), next_state_predictions[:, :, 0], take max later

    def getActionIdxs(self, action_batch):
        return torch.cat([action_batch * self.num_Q_output_slots + i for i in range(self.num_Q_output_slots)], dim=1)

    def calculate_initial_state_val(self):
        with torch.no_grad():
            predictions_inital_states = self.target_net(self.initial_states).view(-1, self.num_actions,
                                                                                  self.num_Q_output_slots)
            initial_state_value = torch.mean(predictions_inital_states[:, :, 1]).item()
        return initial_state_value

    def optimize_model(self):
        if self.USE_EXP_REP:
            if len(self.memory) < self.batch_size:
                return
            transitions = self.memory.sample(self.batch_size)
        else:
            transitions = self.collect_experiences()

        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        if self.TDEC_ENABLED:
            TDEC_reward_batch = torch.cat(batch.TDEC)

        predictions = self.Q_net(state_batch)
        predictions = self.transform_Q_prediction_to_dict(predictions, action_batch=action_batch)

        # QV-Learning:
        if self.USE_QV:
            if self.QV_SPLIT_V:
                with torch.no_grad():
                    next_state_target_net_predictions = torch.zeros(self.batch_size, self.num_V_outputs,
                                                                    device=self.device)
                    next_state_target_net_predictions[non_final_mask] = self.target_value_net(
                        non_final_next_states).view(-1, self.num_V_outputs)
                    next_state_R = next_state_target_net_predictions[:, [1]]

                    # Do not use target net to get prediction for reward of next state
                    next_state_V_net_predictions = torch.zeros(self.batch_size, self.num_V_outputs, device=self.device)
                    next_state_V_net_predictions[non_final_mask] = self.V_net(non_final_next_states).view(-1,
                                                                                                          self.num_V_outputs)
                    next_state_r = next_state_V_net_predictions[:, [0]]

                    expected_return = (next_state_r + next_state_R) * self.gamma_Q

                state_values = self.V_net(state_batch)
                state_reward_pred = state_values[:, [0]]
                state_return_pred = state_values[:, [1]]

                loss_reward_V = F.smooth_l1_loss(state_reward_pred, reward_batch.unsqueeze(1))
                loss_return_V = F.smooth_l1_loss(state_return_pred, expected_return)
                loss_V = loss_reward_V + loss_return_V

                self.value_optimizer.zero_grad()
                loss_V.backward()
                self.value_optimizer.step()

                self.log.add("Loss_V", loss_V.detach())
                self.log.add("Loss_r_V", loss_reward_V.detach())
                self.log.add("Loss_R_V", loss_return_V.detach())

                # For Q-net if it is not split:
                expected_state_values = (expected_return.view(-1) + reward_batch).unsqueeze(1)

                # If Q-net is split:
                expected_r_V = next_state_r * self.gamma_Q
                expected_R_V = next_state_R * self.gamma_Q

            else:
                expected_state_values = self.expectedVals(self.target_value_net, non_final_next_states, non_final_mask,
                                                          reward_batch)
                if self.QV_SPLIT_Q:
                    expected_return = expected_state_values.squeeze(1) - reward_batch
                    expected_return = expected_return.unsqueeze(1)
                state_values = self.V_net(state_batch)
                loss_V = F.smooth_l1_loss(state_values, expected_state_values)

                self.value_optimizer.zero_grad()
                loss_V.backward()
                self.value_optimizer.step()

                self.log.add("Loss_V", loss_V.detach())

            if self.QV_NO_TARGET_Q:
                if self.QV_SPLIT_Q:
                    if self.QV_SPLIT_V:
                        with torch.no_grad():
                            predictions_V_next_state = torch.zeros(self.batch_size, self.num_V_outputs,
                                                                   device=self.device)
                            predictions_V_next_state[non_final_mask] = self.V_net(non_final_next_states).view(-1,
                                                                                                              self.num_V_outputs)
                            predicted_rewards_next_state = predictions_V_next_state[:, [0]]
                            predicted_returns_next_state = predictions_V_next_state[:, [1]]

                            expected_r_V = predicted_rewards_next_state * self.gamma_Q
                            expected_R_V = predicted_returns_next_state * self.gamma_Q

                    else:
                        with torch.no_grad():
                            predictions_V_next_state = torch.zeros(self.batch_size, device=self.device)
                            predictions_V_next_state[non_final_mask] = self.V_net(non_final_next_states).view(-1)
                            expected_return = predictions_V_next_state.unsqueeze(1) * self.gamma_Q
                else:
                    expected_state_values = self.expectedVals(self.V_net, non_final_next_states, non_final_mask,
                                                              reward_batch)

            # QV-Learning:
            if self.QV_SPLIT_Q:
                if self.QV_SPLIT_V:
                    predicted_r = predictions[:, [0]]
                    predicted_r_V = predictions[:, [1]]
                    predicted_R_V = predictions[:, [2]]

                    loss_rewards = F.smooth_l1_loss(predicted_r, reward_batch.unsqueeze(1))
                    loss_r_V = F.smooth_l1_loss(predicted_r_V, expected_r_V)
                    loss_R_V = F.smooth_l1_loss(predicted_R_V, expected_R_V)
                    loss_R_Q = loss_r_V + loss_R_V
                    loss_Q = loss_rewards + loss_R_Q

                    self.optimizer.zero_grad()
                    loss_Q.backward()
                    self.optimizer.step()

                    self.log.add("Loss_Q", loss_Q.detach())
                    self.log.add("Loss_r_Q", loss_rewards.detach())
                    self.log.add("Loss_r_V_Q", loss_r_V.detach())
                    self.log.add("Loss_R_V_Q", loss_R_V.detach())
                    self.log.add("Loss_R_Q", loss_R_Q.detach())

                else:
                    predicted_r = predictions[:, [0]]
                    predicted_V_s_prime = predictions[:, [1]]

                    loss_rewards = F.smooth_l1_loss(predicted_r, reward_batch.unsqueeze(1))
                    loss_return = F.smooth_l1_loss(predicted_V_s_prime, expected_return)
                    loss = loss_rewards + loss_return

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.log.add("Loss_r_Q", loss_rewards.detach())
                    self.log.add("Loss_R_Q", loss_return.detach())
                    self.log.add("Loss_Q", loss.detach())
            else:
                state_action_values = predictions
                # Loss for Q-net:
                loss = F.smooth_l1_loss(state_action_values, expected_state_values)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.log.add("Loss_Q", loss.detach())

        # Predict Reward and Expectation of future reward separately:
        elif self.SPLIT_BELLMAN:
            # Options for calculation of the target for the return R(s,a):
            #    1. No Target network for r prediction.
            #    2. Use Target network and Real network combined for r prediction
            #    3. Or keep using target net

            predicted_rewards = predictions[:, [0]]
            predicted_values = predictions[:, [1]]

            # Calculate the target for the return of the action a in state s: R(s,a)
            with torch.no_grad():
                next_state_values = torch.zeros(self.batch_size, device=self.device)
                next_state_predictions = self.target_net(non_final_next_states).view(-1, self.num_actions, 2)
                next_state_target_rewards = next_state_predictions[:, :, 0]
                next_state_target_returns = next_state_predictions[:, :, 1]
                next_state_returns = next_state_target_returns
                next_state_rewards = next_state_target_rewards
                if self.SPLIT_BELL_NO_TARGET_r or self.SPLIT_BELL_AVG_r or self.SPLIT_BELL_NO_TARGET_AT_ALL:
                    next_state_predictions_no_target = self.Q_net(non_final_next_states).view(-1, self.num_actions, 2)
                    # Only use r(s',a) as part of sum for target of R(s,a)
                    next_state_rewards = next_state_predictions_no_target[:, :, 0]
                    # Use average of r'(s',a) and r(s',a):
                    if self.SPLIT_BELL_AVG_r:
                        next_state_rewards = (next_state_rewards + next_state_target_rewards) / 2
                    if self.SPLIT_BELL_NO_TARGET_AT_ALL:
                        next_state_returns = next_state_predictions_no_target[:, :, 1]

                next_state_predictions = next_state_returns + next_state_rewards
                next_state_values[non_final_mask] = next_state_predictions.max(1)[0] * self.gamma_Q

            loss_rewards = F.smooth_l1_loss(predicted_rewards, reward_batch.unsqueeze(1))
            loss_values = F.smooth_l1_loss(predicted_values, next_state_values.unsqueeze(1))
            loss = loss_rewards + loss_values

            self.optimizer.zero_grad()
            loss_values.backward(retain_graph=True)
            self.optimizer.step()
            
            self.optimizer_r.zero_grad()
            loss_rewards.backward()
            self.optimizer_r.step()

            self.log.add("Loss_Q", loss.detach())
            self.log.add("Loss_r", loss_rewards.detach())
            self.log.add("Loss_R", loss_values.detach())

        elif self.TDEC_ENABLED:
            values_current_state = predictions[:, [0]]
            TDECs_current_state = predictions[:, [1]]

            next_state_values = torch.zeros(self.batch_size, device=self.device)
            # In the episodic curiosity case upon death the value of the next state is 0, but in the non-episodic case
            # the value is equal to the value of the first state of the simulation.
            if self.TDEC_episodic:
                next_state_TDECs = torch.zeros(self.batch_size, device=self.device)
            else:
                next_state_TDECs = torch.ones(self.batch_size, device=self.device)
                initial_state_value = self.calculate_initial_state_val()
                next_state_TDECs *= initial_state_value

            with torch.no_grad():
                predictions_next_state_target = self.target_net(non_final_next_states).view(-1, self.num_actions,
                                                                                            self.num_Q_output_slots)
                predicted_values_next_state = predictions_next_state_target[:, :, 0]
                predicted_TDECs_next_state = predictions_next_state_target[:, :, 1]

                next_state_values[non_final_mask] = predicted_values_next_state.max(1)[0]
                next_state_TDECs[non_final_mask] = predicted_TDECs_next_state.max(1)[0]
                next_state_TDECs = next_state_TDECs.unsqueeze(1)

                expected_state_action_values = (next_state_values * self.gamma_Q) + reward_batch

                expected_TDECs = (next_state_TDECs * self.TDEC_GAMMA) + TDEC_reward_batch

            loss_values = F.smooth_l1_loss(values_current_state, expected_state_action_values.unsqueeze(1))
            loss_TDEC = F.smooth_l1_loss(TDECs_current_state, expected_TDECs)
            loss = loss_values + loss_TDEC

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.log.add("Loss_Q", loss_values.detach())
            self.log.add("Loss_TDEC", loss_TDEC.detach())

        # Standard Q-Learning:
        else:
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            # use .gather() to select the value of the action that we actually took
            state_action_values = predictions

            # Compute V(s_{t+1}) for all next states.
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma_Q) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            self.log.add("Loss_Q", loss.detach())

            # Optimize the Q model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        #### TODO: AHHH do this somehow
        if self.TDEC_ENABLED and 1 == 0:
            values_current_state = predictions[:, [0]]
            TDECs_current_state = predictions[:, [1]]

            next_state_TDECs = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                predictions_next_state_target = self.target_net(non_final_next_states).view(-1, self.num_actions, 2)
                predicted_TDECs_next_state = predictions_next_state_target[:, :, 1]

                next_state_TDECs[non_final_mask] = predicted_TDECs_next_state.max(1)[0]

                expected_TDECs = (next_state_TDECs * self.TDEC_GAMMA) + TDEC_reward_batch

            loss_TDEC = F.smooth_l1_loss(TDECs_current_state, expected_TDECs.unsqueeze(1))

            self.optimizer.zero_grad()
            loss_TDEC.backward()
            self.optimizer.step()

            self.log.add("Loss_TDEC", loss_TDEC.detach())

    def calculateTDE(self, state, action, next_state, reward, store_log=True):
        with torch.no_grad():
            if self.TDEC_USE_TARGET_NET:
                usedNet = self.target_net
            else:
                usedNet = self.Q_net
            # Use only the Q val predictions
            state_value = usedNet(state).view(self.num_actions, self.num_Q_output_slots)[:, 0][action]

            if next_state is not None:
                next_state_value = usedNet(next_state).view(self.num_actions, self.num_Q_output_slots)[:, 0].max()
            else:
                next_state_value = 0
            expected_state_action_value = (next_state_value * self.gamma_Q) + reward

            TDE = expected_state_action_value - state_value

            if self.TDEC_TRAIN_FUNC == "absolute":
                TDE = abs(TDE)
            elif self.TDEC_TRAIN_FUNC == "positive":
                TDE = (abs(TDE) + TDE) / 2
            elif self.TDEC_TRAIN_FUNC == "mse":
                TDE = TDE ** 2

            if store_log:
                self.log.add("TDE", TDE.item())

            return TDE

    def act_in_test_env(self):
        # Act without noise
        old_TDEC_Factor = self.TDEC_FACTOR
        self.TDEC_FACTOR = 0
        action = self.select_action(self.testState, 0)
        self.TDEC_FACTOR = old_TDEC_Factor

        self.testState, reward, done, _ = self.testEnv.step(action.item())
        self.test_episode_rewards.append(reward)
        self.log.add("Total Reward", np.sum(self.test_episode_rewards))

        if done or (self.max_steps_per_episode > 0 and len(self.test_episode_rewards) >= self.max_steps_per_episode):
            self.testState = self.testEnv.reset()
            self.test_episode_rewards = []
        self.testState = torch.tensor([self.testState], device=self.device).float()

    def run(self, num_steps=5000, verbose=True, render=False, on_server=False):
        if self.MAX_EPISODE_STEPS > 0:
            self.max_steps_per_episode = self.MAX_EPISODE_STEPS
        elif self.env_name == "LunarLander-v2":
            self.max_steps_per_episode = 1000
        elif self.env_name == "CartPole-v1":
            self.max_steps_per_episode = 500
        else:
            self.max_steps_per_episode = 0

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
            next_state, reward, done, _ = self.env.step(action.item())
            if done:
                next_state = None
            else:
                next_state = torch.tensor([next_state], device=self.device).float()
            reward = torch.tensor([reward], device=self.device, dtype=torch.float)

            TDE = self.calculateTDE(state, action, next_state, reward, store_log=False)

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
                    print("Current setting test: ", round(self.steps_done / num_steps * 100, 2), "%", end="\r")
                # Stop when max number of steps is reached
                if self.steps_done >= num_steps:
                    run = False
                    break

                self.act_in_test_env()

                # Select and perform an action
                action = self.select_action(state, self.epsilon)
                self.steps_done += 1
                next_state, reward, done, _ = self.env.step(action.item())
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
                if self.QV_CURIOSITY_ENABLED and self.QV_CURIOSITY_MID:
                    self.QVC_scale *= QVC_decay
                elif self.TDEC_ENABLED and self.TDEC_MID:
                    self.TDEC_FACTOR = self.TDEC_FACTOR * TDEC_DECAY

                # Perform one step of the optimization (on the target network)
                for i in range(self.UPDATES_PER_STEP):
                    self.optimize_model()

                if done or (self.max_steps_per_episode > 0 and t >= self.max_steps_per_episode):
                    if verbose:
                        print("Episode ", i_episode)
                        print("Steps taken so far: ", self.steps_done)
                        print("Cumulative Reward this episode:", np.sum(episode_rewards))
                        if i_episode % 10 == 0:
                            plot_rewards(self.rewards, self.env_name)
                            plot_rewards(self.log.storage["Total Reward"], self.env_name, "Total Reward")
                        print("Epsilon: ", round(self.epsilon, 4))
                        if self.QV_CURIOSITY_ENABLED and self.QV_CURIOSITY_MID:
                            print("QVC scale: ", round(self.QVC_scale, 4))
                        print()
                    break
                # Update the target network
                if self.steps_done % self.target_network_steps == 0:
                    if __debug__ and verbose:
                        print("Updating Target Network")
                    if self.USE_QV:
                        self.target_value_net.load_state_dict(self.V_net.state_dict())
                    else:
                        self.target_net.load_state_dict(self.Q_net.state_dict())

        if verbose:
            plot_rewards(self.rewards, self.env_name)
            plot_rewards(self.log.storage["Total Reward"], self.env_name, "Total Reward")
            print('Complete')
        self.env.close()
        return i_episode, self.rewards, self.log.storage


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
    return np.mean(result_list) * (1 - final_percentage_weight) + mean_final_percent(result_list, percentage) * final_percentage_weight

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
                print(key + ":", randomizedParams[key], end="  ")
            print()
        else:
            randomizedParams = {}

        trainer = Trainer(env, device, **trialParams, **randomizedParams, **hyperparamDict)
        steps, rewards, log = trainer.run(verbose=False, num_steps=length_of_tests, on_server=on_server)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lunar = "LunarLander-v2"
    cart = "CartPole-v1"
    acro = "Acrobot-v1"
    mountain = "MountainCar-v0"

    # print("Action space: ", env.action_space)
    # print("Observation space: ", env.observation_space)

    # trainer = Trainer(environment_name, device)

    trainer = Trainer(lunar, device, TDEC_ENABLED=False, TDEC_episodic=True, TDEC_SCALE=0.5)
    trainer.run(render=False, num_steps=50000)
