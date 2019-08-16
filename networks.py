import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
from util import *

import math
import copy
import numpy as np


def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1

def act_funct_string2function(name):
    name = name.lower()
    if name == "relu":
        return F.relu
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "tanh":
        return torch.tanh

def query_act_funct(layer_dict):
    try:
        activation_function = act_funct_string2function(layer_dict["act_func"])
    except KeyError:
        def activation_function(x):
            return x
    return activation_function

def string2layer(name, input_size, neurons):
    name = name.lower()
    if name == "linear":
        return nn.Linear(input_size, neurons)
    elif name == "lstm":
        return nn.LSTM(input_size, neurons)
    elif name == "gru":
        return nn.GRU(input_size, neurons)


# TODO: possibly put activation functions per layer into some other list...
def create_ff_layers(input_size, layer_dict, output_size):
    layers = nn.ModuleList()
    act_functs = []
    for layer in layer_dict:
        this_layer_neurons = layer["neurons"]
        layers.append(string2layer(layer["name"], input_size, this_layer_neurons))
        act_functs.append(query_act_funct(layer))
        input_size = this_layer_neurons
    if output_size is not None:
        layers.append(nn.Linear(input_size, output_size))
        act_functs.append(lambda x: x)
    return layers, act_functs


# Create a module list of conv layers specified in layer_dict
def create_conv_layers(input_matrix_shape, layer_dict):
    # format for entry in matrix_layers: ("conv", channels_in, channels_out, kernel_size, stride) if conv or
    #  ("batchnorm") for batchnorm
    matrix_width = input_matrix_shape[0]
    matrix_height = input_matrix_shape[1]
    channel_last_layer = input_matrix_shape[2]

    act_functs = []
    layers = nn.ModuleList()
    for layer in layer_dict:
        # Layer:
        if layer["name"] == "batchnorm":
            layers.append(nn.BatchNorm2d(channel_last_layer))
        elif layer["name"] == "conv":
            this_layer_channels = layer["channels_out"]
            layers.append(nn.Conv2d(channel_last_layer, this_layer_channels, layer["kernel_size"],
                                    layer["stride"]))
            matrix_width = conv2d_size_out(matrix_width, layer["kernel_size"], layer["stride"])
            matrix_height = conv2d_size_out(matrix_height, layer["kernel_size"], layer["stride"])
            channel_last_layer = this_layer_channels

        act_functs.append(query_act_funct(layer))

    conv_output_size = matrix_width * matrix_height * channel_last_layer

    return layers, conv_output_size, act_functs

def apply_layers(x, layers, act_functs):
    for idx in range(len(layers)):
        x = act_functs[idx](layers[idx](x))
    return x


class OptimizableNet(nn.Module):
    def __repr__(self):
        # TODO: return summary using pytorch
        return self.type

    def __init__(self, device, log, hyperparameters):
        super(OptimizableNet, self).__init__()
        self.log = log
        self.device = device
        self.hyperparameters = hyperparameters

    def compute_loss(self, output, target):
        return F.smooth_l1_loss(output, target)

    def optimize_net(self, output, target, optimizer, name=""):
        loss = self.compute_loss(output, target.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #TODO: log gradients and weight sizes/stds! Have one log per NN (whether V, Q, or actor)
        name = "loss_" + self.name + (("_" + name) if name != "" else "")
        self.log.add(name, loss.detach())


class ProcessState(OptimizableNet):
    def __init__(self, vector_len, matrix_shape, log, device, hyperparameters, matrix_max_val=255):
        super(ProcessState, self).__init__(log, device, hyperparameters)

        vector_output_size = 0
        if vector_len is not None:
            self.vector_normalizer = Normalizer(vector_len)
            vector_layers = hyperparameters["layers_feature_vector"]
            self.layers_vector, self.act_functs_vector = create_ff_layers(vector_len, vector_layers, None)
            vector_output_size = self.layers_vector[-1].out_features

        # matrix size has format (x_len, y_len, n_channels)
        matrix_output_size = 0
        if matrix_shape is not None:
            self.matrix_normalizer = Normalizer(matrix_shape, matrix_max_val)
            matrix_layers = hyperparameters["layers_feature_matrix"]
            self.layers_matrix, matrix_output_size, self.act_functs_matrix = create_conv_layers(matrix_shape, matrix_layers)

        # format for parameters: ["linear": (input, output neurons), "lstm": (input, output neurons)]
        merge_layers = hyperparameters["layers_feature_merge"]
        merge_input_size = vector_output_size + matrix_output_size
        self.layers_merge, self.act_functs_merge = create_ff_layers(merge_input_size, merge_layers, None)

        # Set feature extractor to GPU if possible:
        self.to(device)

    def forward(self, vector, matrix):
        # TODO: instead of having two inputs, only have one state to make the function more general.
        # TODO: to separate the state into matrix and vector, check the number of dimensions of the state
        merged = torch.tensor([])
        if matrix is not None:
            batch_size = matrix.size(0)
            matrix = apply_layers(matrix, self.layers_matrix, self.act_functs_matrix)
            matrix = matrix.view(batch_size, -1)
            merged = torch.cat((merged, matrix), 0)

        if vector is not None:
            vector = apply_layers(vector, self.layers_vector, self.act_functs_vector)
            merged = torch.cat((merged, vector), 0)

        merged = apply_layers(merged, self.layers_merge, self.act_functs_merge)

        return merged


class ProcessStateAction(OptimizableNet):
    def __init__(self, state_features_len, action_len, log, device, hyperparameters):
        super(ProcessStateAction, self).__init__(device, log, hyperparameters)

        input_size = state_features_len + action_len

        layers = hyperparameters["layers_state_action_features"]
        self.layers, self.act_functs = create_ff_layers(input_size, layers, device, None)

        # Put feature extractor on GPU if possible:
        self.to(device)

    def forward(self, state_features, actions):
        x = torch.cat((state_features, actions), 0)
        x = apply_layers(x, self.layers, self.act_functs)
        return x

class TempDiffNet(OptimizableNet):
    def __init__(self, input_size, updateable_parameters, device, log, hyperparameters):
        super(TempDiffNet, self).__init__(device, log, hyperparameters)

        self.target_network_polyak = hyperparameters["use_polyak_averaging"]
        if self.target_network_polyak:
            self.tau = hyperparameters["polyak_averaging_tau"]
        self.target_network_hard_steps = hyperparameters["target_network_hard_steps"]
        self.split = hyperparameters["split_Bellman"]
        self.use_target_net = hyperparameters["use_target_net"]

        self.current_reward_prediction = None

        # Initiate reward prediction network:
        if self.split:
            self.lr_r = hyperparameters["lr_r"]
            reward_layers = hyperparameters["layers_r"]
            self.layers_r, self.act_functs_r = create_ff_layers(input_size, reward_layers, self.output_neurons)
            self.to(device)
            self.optimizer_r = optim.Adam(itertools.chain.from_iterable(self.layers_r + updateable_parameters),
                                        lr=self.lr_r)

    def create_target_net(self):
        target_net = None
        if self.use_target_net:
            target_net = copy.deepcopy(self)
            # TODO(small): check if the following line makes sense - do we want different initial weights for the target network if we use polyak averaging?
            #target_net.apply(self.weights_init)
            target_net.use_target_net = False
            target_net.eval()
        return target_net

    def forward(self, x):
        predicted_reward = 0
        if self.split:
            predicted_reward = apply_layers(x, self.layers_r, self.act_functs_r)
            #self.last_r_prediction = predicted_reward
            # TODO: What was the upper line used for?
        predicted_state_value = apply_layers(x, self.layers_TD, self.act_functs_TD)
        return predicted_state_value + predicted_reward

    def forward_r(self, x):
        return apply_layers(x, self.layers_r, self.act_functs_r)

    def forward_R(self, x):
        return apply_layers(x, self.layers_TD, self.act_functs_TD)

    def weights_init(self, m):
        # if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)

    # TODO: the following functions is just a model to modify the two functions below it appropriately
    def take_mean_weights_of_two_models(self, model1, model2):
        beta = 0.5  # The interpolation parameter
        params1 = model1.named_parameters()
        params2 = model2.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(beta * param1.data + (1 - beta) * dict_params2[name1].data)

        model.load_state_dict(dict_params2)

    def multiply_state_dict(self, state_dict, number):
        for i in state_dict:
            state_dict[i] *= number
        return state_dict

    def add_state_dicts(self, state_dict_1, state_dict_2):
        for i in state_dict_1:
            state_dict_1[i] += state_dict_2[i]
        return state_dict_1

    def update_target_network(self, steps):
        if self.target_network_polyak:
            # TODO: maybe find a better way than these awkward modify dict functions? (if they even work)
            self.target_net.load_state_dict(self.add_state_dicts(
                self.multiply_state_dict(self.target_net.state_dict(), (1 - self.tau)),
                self.multiply_state_dict(self.state_dict(), self.tau)))
        else:
            if steps % self.target_network_hard_steps == 0:
                self.target_net.load_state_dict(self.state_dict())

    def predict_current_state(self, state_features, state_action_features, actions):
        raise NotImplementedError

    def calculate_next_state_values(self, non_final_next_state_features, non_final_mask):
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_predictions = self.target_net.predict_state_value(non_final_next_state_features,
                                                                     actor=self.actor).detach()
        next_state_values[non_final_mask] = next_state_predictions  # [0] #TODO: why [0]?
        return next_state_values

    def optimize(self, state_features, state_action_features, action_batch, reward_batch,
                        non_final_next_state_features, non_final_mask):
        # Compute V(s_t) or Q(s_t, a_t)
        predictions_current, reward_prediction = self.predict_current_state(state_features, state_action_features,
                                                                                 action_batch)
        # Train reward net if it exists:
        if self.split:
            self.optimize_net(reward_prediction, reward_batch, self.optimizer_r, "r")
        # Compute V(s_t+1) or max_aQ(s_t+1, a) for all next states.
        predictions_next_state = self.predict_next_state(non_final_next_state_features, non_final_mask)

        # Compute the expected values. Do not add the reward, if the critic is split
        self.expected_value_next_state = (predictions_next_state * self.gamma) + (reward_batch if self.split else 0)

        # TD must be stored for actor-critic updates:
        self.TDE = self.optimize_net(predictions_current, self.expected_value_next_state, self.optimizer_TD, "R")

    def calculate_TDE(self, state, action_batch, next_state, reward_batch):
        return torch.tensor([0])
        # TODO fix


        # TODO: replace the None for matrix as soon as we have the function in policies.py of state2parts
        state_features = self.F_s(state, None)
        if next_state is not None:
            non_final_next_state_features = self.F_s(next_state, None)
            non_final_mask = [0]
        else:
            non_final_mask = []
        state_action_features = None
        if not self.discrete:
            state_action_features = self.F_sa(state_features, action_batch)
        # Compute V(s_t) or Q(s_t, a_t)
        predictions_current, reward_prediction = self.predict_current_state(state_features, state_action_features,
                                                                            action_batch).detach()
        # Compute current prediction for reward plus state value:
        current_prediction = reward_prediction + self.gamma * predictions_current
        # Compute V(s_t+1) or max_aQ(s_t+1, a) for all next states.
        predictions_next_state = self.predict_next_state(non_final_next_state_features, non_final_mask)
        # Compute the expected values. Do not add the reward, if the critic is split
        expected_value_next_state = (predictions_next_state * self.gamma) + (reward_batch if self.split else 0)
        return expected_value_next_state - current_prediction



class Q(TempDiffNet):
    def __init__(self, input_size, num_actions, discrete_action_space, F_s, F_s_a, device, log, hyperparameters):
        updateable_parameters = list(F_s.parameters()) +(list(F_s_a.parameters()) if not discrete_action_space else [])
        super(Q, self).__init__(input_size, updateable_parameters, device, log, hyperparameters)

        # can either have many outputs or one
        self.num_actions = num_actions
        self.discrete = discrete_action_space
        self.output_neurons = num_actions if self.discrete else 1

        # Set up params:
        self.use_QV = hyperparameters["use_QV"]
        self.use_QVMAX = hyperparameters["use_QVMAX"]

        # Create layers
        layers = hyperparameters["layers_Q"]
        self.layers_TD, self.act_functs_TD = create_ff_layers(input_size, layers, self.output_neurons)
        # Put feature extractor on GPU if possible:
        self.to(device)

        # Define optimizer and previous networks
        # TODO: only optimize F_sa depedning on self.multi_output
        self.lr_TD = hyperparameters["lr_Q"]
        self.F_s = F_s
        self.F_s_a = F_s_a

        self.optimizer_TD = optim.Adam(list(self.layers_TD.parameters()) + updateable_parameters, lr=self.lr_TD)
        # Create target net
        self.target_net = self.create_target_net()

    def predict_next_state(self, non_final_next_state_features, non_final_mask):
        if self.use_QVMAX:
            return self.V.calculate_next_state_values(non_final_next_state_features, non_final_mask)
        elif self.use_QV:
            # This assumes that V is always trained directly before Q
            return self.V_net.expected_value_next_state
        else:
            return self.calculate_next_state_values(non_final_next_state_features, non_final_mask)

    def predict_current_state(self, state_features, state_action_features, actions):
        reward_prediction = 0
        if self.discrete:
            if self.split:
                reward_prediction = self.forward_r(state_features).gather(1, actions)
            return self.forward_R(state_features).gather(1, actions), reward_prediction  # .gather action that is taken
        else:
            if self.split:
                reward_prediction = self.forward_r(state_action_features)
            return self.forward_R(state_action_features), reward_prediction  # self.F_s_A(state_features, actions))

    def predict_state_value(self, state_features):
        if self.discrete:
            return self.forward(state_features).max(1)[0]
        else:
            with torch.no_grad():
                state_action_features = self.F_s_a(state_features, self.actor(state_features))
            # TODO: make sure whether these state-action_features are required somewhere else and store it if that is the case
            return self.predict_state_action_value(None, state_action_features, None)

    def predict_state_action_value(self, state_features, state_action_features, actions):
        if self.discrete:
            return self.forward(state_features).gather(1, actions)  # .gather action that is taken
        else:
            return self.forward(state_action_features)  # self.F_s_A(state_features, actions))


class V(TempDiffNet):
    def __init__(self, input_size, F_s, log, device, hyperparameters):
        updateable_parameters = F_s.parameters()
        super(V, self).__init__(input_size, updateable_parameters, log, device, hyperparameters)

        self.output_neurons = 1

        self.use_QVMAX = hyperparameters["use_QVMAX"]

        # Create layers
        layers = hyperparameters["layers_V"]
        self.layers_TD, self.act_functs_TD = create_ff_layers(input_size, layers, self.output_neurons)
        # Put feature extractor on GPU if possible:
        self.to(device)

        # Define optimizer and previous networks
        self.lr_TD = hyperparameters["lr_V"]
        self.F_s = F_s
        self.optimizer_TD = optim.Adam(itertools.chain(self.layers, updateable_parameters), lr=self.lr_TD)
        # Create target net
        self.target_net = self.create_target_net()

    def predict_next_state(self, non_final_next_state_features, non_final_mask):
        if self.use_QVMAX:
            return self.Q.calculate_next_state_values(non_final_next_state_features, non_final_mask)
        else:
            return self.calculate_next_state_values(non_final_next_state_features, non_final_mask)

    def predict_state_value(self, state_features):
        with torch.no_grad():
            return self(state_features)

    def predict_current_state(self, state_features, state_action_features, actions):
        reward_prediction = 0
        if self.split:
            reward_prediction = self.forward_r(state_features)
        return self.forward_R(state_features), reward_prediction


class Actor(OptimizableNet):
    def __init__(self, input_size, num_actions, action_lows, action_highs,
                 hyperparameters, device):
        super(Actor, self).__init__()

        self.discrete_action_space = num_actions > 1
        output_size = num_actions if self.discrete_action_space else len(action_lows)

        # Initiate arrays for output function:
        self.relu_idxs = []
        self.sigmoid_idxs = []
        self.tanh_idxs = []
        self.scaling = np.ones(len(action_lows))
        self.offset = np.ones(len(action_lows))

        # Create layers
        layers = hyperparameters["layers_actor"]
        self.layers, self.act_functs = create_ff_layers(input_size, layers, output_size)
        self.act_func_output_layer = self.create_output_act_func(self.discrete_action_space, action_lows,
                                                                 action_highs)
        # Put feature extractor on GPU if possible:
        self.to(device)

        # Define optimizer and previous networks
        self.lr = hyperparameters["lr_actor"]
        self.F_s = F_s
        updateable_parameters = [self.F_s.parameters()]
        self.optimizer = optim.Adam(itertools.chain.from_iterable(self.layers + updateable_parameters), lr=self.lr)


    def forward(self, x):
        for idx in range(len(self.layers)):
            x = self.act_functs[idx](self.layers[idx](x))

        outputs = []
        for i in range(len(self.output_layers)):
            outputs.append(self.act_funcs_output_layer[i](self.output_layers[i](x)))

        return torch.cat(outputs, dim=1)

    def compute_loss(self, output, target):
        if self.discrete_action_space:
            loss_func =  torch.nn.CrossEntropyLoss()
            return loss_func(output, target)
        else:
            # TODO: this loss does not help combat the vanishing gradient problem that we have because of the use of sigmoid activations to squash our actions into the correct range
            return F.smooth_l1_loss(output, target)

    def output_function_continuous(self, x):
        if self.relu_idxs:
            x[self.relu_idxs] = F.relu(x[self.relu_idxs])
        if self.sigmoid_idxs:
            x[self.sigmoid_idxs] = F.sigmoid(x[self.sigmoid_idxs])
        if self.tanh_idxs:
            x[self.tanh_idxs] = F.tanh(x[self.tanh_idxs])
        return (x * self.scaling) + self.offset

    def create_output_act_func(self, discrete_action_space, action_lows, action_highs):
        if discrete_action_space:
            return F.sigmoid
        else:
            for i in range(len(action_lows)):
                low = action_lows[i]
                high = action_highs[i]
                if not (low and high):
                    if low == -math.inf or high == math.inf:
                        self.relu_idxs.append(i)
                    else:
                        self.sigmoid_idxs.append(i)
                        self.scaling[i] = high + low
                elif low == high * -1:
                    if low != -math.inf:
                        self.tanh_idxs.append(i)
                        self.scaling[i] = high
                else:
                    self.offset[i] = (high - low) / 2
                    self.scaling[i] = high - offset[i]
                    self.tanh_idxs.append(i)
        return self.output_function_continuous

    def optimize(self, state_features, reward_batch, action_batch, non_final_next_states, non_final_mask):
        # Calculate current actions for state_batch:
        actions_current_state = self.actor(state_features)
        better_actions_current_state = actions_current_state.detach().copy()
        if self.use_CACLA_V:
            # Requires TDE_V
            # Check which actions have a pos TDE
            pos_TDE_mask = self.V_net.TDE > 0
            better_actions_current_state[pos_TDE_mask] = action_batch[pos_TDE_mask]
        if self.use_CACLA_Q:
            # Calculate mask of pos expected Q minus Q(s, mu(s))
            action_TDE = self.Q_net.expectations_next_state - self.Q_net(state_features, actions_current_state).detach()
            pos_TDE_mask = action_TDE > 0
            better_actions_current_state[pos_TDE_mask] = action_batch[pos_TDE_mask]
        # TODO: implement CACLA+Var

        # TODO - Idea: When using QV, possibly reinforce actions only if Q and V net agree (first check how often they disagree and in which cases)
        if self.use_DDPG:
            # 1. calculate derivative of Q towards actions 2. Reinforce towards actions plus gradients
            q_vals = self.Q_net(state_features, actions_current_state)
            q_vals.backward()  # retain necessary?
            gradients = actions_current_state.grad
            # TODO: multiply gradients * -1 to ensure Q values increase? probably...
            # Normalize gradients:
            gradients = self.normalize_gradients(gradients)
            # TODO: maybe normalize within the actor optimizer...?
            # TODO Normalize over batch, then scale by inverse TDE (risky thing:what about very small TDEs?
            better_actions_current_state = actions_current_state + gradients

        if self.use_SPG:
            # Calculate mask of Q(s,a) minus Q(s, mu(s))
            action_TDE = Q_pred_batch_state_action.detach()["Q"] - self.Q_net(state_batch, actions_current_state)
            pos_TDE_mask = action_TDE > 0
            better_actions_current_state[pos_TDE_mask] = action_batch[pos_TDE_mask]
            # 1. Get batch_actions and batch_best_actions (implement best_actions everywhere)
            # 2. Calculate eval of current action
            # 3. Compare batch_action and batch_best_actions to evals of current actions
            # 4. Sample around best action with Gaussian noise until better action is found, then sample around this
            # 5. Reinforce towards best actions
        if self.use_GISPG:
            # Gradient Informed SPG
            # Option one:
            # Do SPG, but for every action apply DDPG to get the DDPG action and check if it is better than the non-
            # DDPG action.
            # Option two:
            # Do SPG, but do not sample with Gaussian noise. Instead always walk towards gradient of best action,
            #  with magnitude that decreases over one sampling period
            #
            #
            pass

        self.optimize_net(actions_current_state, better_actions_current_state, self.optimizer, "actor")
        # Train actor towards better actions (loss = better - current)
