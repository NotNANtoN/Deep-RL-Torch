import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
from util import *

from torch.utils.tensorboard import SummaryWriter

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
        act_functs.append(None)
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
        if act_functs[idx] is None:
            x = layers[idx](x)
        else:
            x = act_functs[idx](layers[idx](x))
    return x


def one_hot_encode(x, num_actions):
    y = torch.zeros(x.shape[0], num_actions).float()
    return y.scatter(1, x, 1)


class OptimizableNet(nn.Module):
    # def __repr__(self):
    # TODO: return summary using pytorch
    #    return str(self.type)

    def __init__(self, env, device, log, hyperparameters, is_target_net=False):
        super(OptimizableNet, self).__init__()
        self.env = env
        self.log = log
        self.device = device
        self.hyperparameters = hyperparameters

        # Env Action Space:
        self.discrete_env = True if "Discrete" in str(env.action_space) else False
        if self.discrete_env:
            self.num_actions = env.action_space.n
        else:
            self.num_actions = len(env.action_space.low)
            self.action_low = torch.tensor(env.action_space.high)
            self.action_high = torch.tensor(env.action_space.low)
        # Env Obs Space:
        self.vector_len = len(self.env.observation_space.high)
        self.matrix_shape = None

        # Load hyperparameters:
        if is_target_net:
            self.use_target_net = False
        else:
            self.use_target_net = hyperparameters["use_target_net"]
        self.retain_graph = False
        self.max_norm = hyperparameters["max_norm"]
        self.batch_size = hyperparameters["batch_size"]
        self.optimizer = hyperparameters["optimizer"]
        # Actor:
        self.use_actor_critic = hyperparameters["use_actor_critic"]
        self.use_CACLA_V = hyperparameters["use_CACLA_V"]
        self.use_CACLA_Q = hyperparameters["use_CACLA_Q"]
        self.use_DDPG = hyperparameters["use_DDPG"]
        self.use_SPG = hyperparameters["use_SPG"]
        self.use_GISPG = hyperparameters["use_GISPG"]

        # Target net:
        self.target_network_polyak = hyperparameters["use_polyak_averaging"]
        if self.target_network_polyak:
            self.tau = hyperparameters["polyak_averaging_tau"]
        self.target_network_hard_steps = hyperparameters["target_network_hard_steps"]

    def compute_loss(self, output, target, sample_weights):
        loss = F.smooth_l1_loss(output, target, reduction='none')
        if sample_weights is None:
            return loss.mean()
        else:
            return (loss * sample_weights).mean()

    def optimize_net(self, output, target, optimizer, name="", sample_weights=None, retain_graph=False):
        loss = self.compute_loss(output, target, sample_weights)

        optimizer.zero_grad()
        loss.backward(retain_graph=self.retain_graph + retain_graph)
        if self.max_norm:
            torch.nn.utils.clip_grad.clip_grad_norm_(self.parameters(), self.max_norm)
        optimizer.step()

        # TODO: log gradients and weight sizes/stds! Have one log per NN (whether V, Q, or actor)
        name = "loss_" + self.name + (("_" + name) if name != "" else "")
        detached_loss = loss.detach().clone().item()
        self.log.add(name, detached_loss)

        return detached_loss

    # TODO: the following functions is just a model to modify the two functions below it appropriately
    def take_mean_weights_of_two_models(self, target_net, real_net):
        # beta = 0.5  # The interpolation parameter. 0.5 for mean

        params1 = target_net.named_parameters()
        params2 = real_net.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            # print(name1)
            if name1 in dict_params2:
                # print("is drin")
                dict_params2[name1].data.copy_((1 - self.tau) * param1.data + self.tau * dict_params2[name1].data)

        return dict_params2

    def update_targets(self, steps):
        if self.target_network_polyak:
            # TODO: maybe find a better way than these awkward modify dict functions? (if they even work)
            self.target_net.load_state_dict(self.take_mean_weights_of_two_models(self.target_net, self))
        else:
            if steps % self.target_network_hard_steps == 0:
                current_weight_dict = self.state_dict()
                # print("Q weight dict items: ", current_weight_dict.items())
                # print("Target weight dict items: ", self.target_net.state_dict().items())
                # print("Current dict: ", self.state_dict().items())
                current_weight_dict_filtered = {k: v for k, v in current_weight_dict.items()
                                                if "target_net" not in k and
                                                "F_s" not in k and
                                                "Q" not in k and
                                                "V" not in k}
                # print("Filtered: ", current_weight_dict_filtered)
                # print("Target dict: ", self.target_net.state_dict().items())
                # print()
                self.target_net.load_state_dict(current_weight_dict_filtered)

    def create_target_net(self):
        target_net = None
        if self.use_target_net:
            target_net = self.recreate_self()
            # TODO(small): check if the following line makes sense - do we want different initial weights for the target network if we use polyak averaging?
            # target_net.apply(self.weights_init)
            for param in target_net.parameters():
                param.requires_grad = False
            target_net.use_target_net = False
            target_net.eval()
        return target_net


class ProcessState(OptimizableNet):
    def __init__(self, env, log, device, hyperparameters, matrix_max_val=255, is_target_net=False):
        super(ProcessState, self).__init__(env, device, log, hyperparameters)

        vector_len = self.vector_len
        matrix_shape = self.matrix_shape

        if is_target_net:
            self.use_target_net = False
        else:
            self.use_target_net = hyperparameters["use_target_net"]

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
            self.layers_matrix, matrix_output_size, self.act_functs_matrix = create_conv_layers(matrix_shape,
                                                                                                matrix_layers)

        # format for parameters: ["linear": (input, output neurons), "lstm": (input, output neurons)]
        merge_layers = hyperparameters["layers_feature_merge"]
        merge_input_size = vector_output_size + matrix_output_size
        self.layers_merge, self.act_functs_merge = create_ff_layers(merge_input_size, merge_layers, None)

        # TODO: the following does not work because we still have the weird (vector, matrix) input and matrix cannot be none here
        # self.log.writer.add_graph(self, input_to_model=[torch.rand(vector_len), None], verbose=True)

        # Set feature extractor to GPU if possible:
        self.to(device)

        self.target_net = self.create_target_net()

    def forward(self, state):
        # TODO: instead of having two inputs, only have one state to make the function more general.
        # TODO: to separate the state into matrix and vector, check the number of dimensions of the state

        vector, matrix = self.state2parts(state)

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

    def forward_next_state(self, states):
        if self.use_target_net:
            return self.target_net(states)
        else:
            return self(states)

    def state2parts(self, state):
        # TODO: adjust this function to be able to deal with vector envs, rgb envs, and mineRL
        return state, None

    def recreate_self(self):
        return self.__class__(self.env, self.log, self.device, self.hyperparameters, is_target_net=True)


# TODO for all nets: enable options to not update target nets and to predict next state value based on current net instead of using the target net

class ProcessStateAction(OptimizableNet):
    def __init__(self, state_features_len, env, log, device, hyperparameters, is_target_net=False):
        super(ProcessStateAction, self).__init__(env, device, log, hyperparameters, is_target_net)

        self.state_features_len = state_features_len

        # Create layers:
        input_size = state_features_len + self.num_actions
        layers = hyperparameters["layers_state_action_features"]
        self.layers, self.act_functs = create_ff_layers(input_size, layers, None)

        # self.log.writer.add_graph(self, input_to_model=torch.rand(state_features_len), verbose=True)

        # Put feature extractor on GPU if possible:
        self.to(device)


        self.target_net = self.create_target_net()

    def forward(self, state_features, actions):
        if self.discrete_env:
            actions = one_hot_encode(actions, self.num_actions)
        x = torch.cat((state_features, actions), 1)
        x = apply_layers(x, self.layers, self.act_functs)
        return x

    def forward_next_state(self, state_features, action):
        if self.use_target_net:
            return self.target_net(state_features, action)
        else:
            return self(state_features, action)

    def recreate_self(self):
        return self.__class__(self.state_features_len, self.env, self.log, self.device, self.hyperparameters,
                              is_target_net=True)


class TempDiffNet(OptimizableNet):
    def __init__(self, env, device, log, hyperparameters, is_target_net=False):
        super(TempDiffNet, self).__init__(env, device, log, hyperparameters)

        self.use_actor_critic = hyperparameters["use_actor_critic"]
        self.split = hyperparameters["split_Bellman"]
        self.use_target_net = hyperparameters["use_target_net"] if not is_target_net else False
        self.gamma = hyperparameters["gamma"]

        self.current_reward_prediction = None

    def create_split_net(self, input_size, updateable_parameters, device, hyperparameters):
        if self.split:
            self.lr_r = hyperparameters["lr_r"]
            reward_layers = hyperparameters["layers_r"]
            self.layers_r, self.act_functs_r = create_ff_layers(input_size, reward_layers, self.output_neurons)
            self.to(device)
            self.optimizer_r = self.optimizer(list(self.layers_r.parameters()) + updateable_parameters, lr=self.lr_r)

    def recreate_self(self):
        return self.__class__(self.input_size, self.env, None, None, self.device, self.log,
                              self.hyperparameters, is_target_net=True)

    def forward(self, x):
        predicted_reward = 0
        if self.split:
            predicted_reward = apply_layers(x, self.layers_r, self.act_functs_r)
            # self.last_r_prediction = predicted_reward
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

    def predict_current_state(self, state_features, state_action_features, actions):
        raise NotImplementedError

    def calculate_next_state_values(self, non_final_next_state_features, non_final_mask, actor=None):
        next_state_values = torch.zeros(self.batch_size, 1, device=self.device)
        with torch.no_grad():
            next_state_predictions = self.target_net.predict_state_value(non_final_next_state_features, self.F_sa,
                                                                         actor)
        # print(next_state_values[non_final_mask])
        # print(next_state_predictions)
        next_state_values[non_final_mask] = next_state_predictions  # [0] #TODO: why [0]?
        return next_state_values

    def calculate_updated_value_next_state(self, reward_batch, non_final_next_state_features, non_final_mask,
                                           actor=None, Q=None, V=None):
        # Compute V(s_t+1) or max_aQ(s_t+1, a) for all next states.
        predictions_next_state = self.predict_next_state(non_final_next_state_features, non_final_mask, actor, Q, V)
        # print("Next state features: ", non_final_next_state_features)
        # print("Prediction next state: ", predictions_next_state)

        # Compute the updated expected values. Do not add the reward, if the critic is split
        return (predictions_next_state * self.gamma) + (reward_batch if not self.split else 0)

    def optimize(self, transitions, importance_weights, actor=None, Q=None, V=None):

        state_features = transitions["state_features"]
        state_action_features = transitions["state_action_features"]
        action_batch = transitions["action"]
        reward_batch = transitions["reward"]
        non_final_next_state_features = transitions["non_final_next_state_features"]
        non_final_mask = transitions["non_final_mask"]

        # Compute V(s_t) or Q(s_t, a_t)
        predictions_current, reward_prediction = self.predict_current_state(state_features, state_action_features,
                                                                            action_batch)

        # Store for SPG Actor update:--- probably not necessary, as we need to use target network predictions for SPG
        #self.predictions_current_state = (predictions_current + reward_prediction).detach().clone()

        # print("Current state features: ", state_features)
        # print("Prediction current state: ", predictions_current.detach())

        # Train reward net if it exists:
        if self.split:
            self.optimize_net(reward_prediction, reward_batch, self.optimizer_r, "r", retain_graph=True)
            TDE_r = reward_batch - reward_prediction
        else:
            TDE_r = 0

        # Compute the expected values. Do not add the reward, if the critic is split
        self.expected_value_next_state = self.calculate_updated_value_next_state(reward_batch,
                                                                                 non_final_next_state_features,
                                                                                 non_final_mask, actor, Q, V)
        # print("Expected value next state: ", self.expected_value_next_state)
        # print()

        # TD must be stored for actor-critic updates:
        self.optimize_net(predictions_current, self.expected_value_next_state, self.optimizer_TD, "TD",
                          sample_weights=importance_weights)
        TDE_TD = self.expected_value_next_state - predictions_current
        self.TDE = (TDE_r + TDE_TD).detach()
        return self.TDE

    def calculate_TDE(self, state, action_batch, next_state, reward_batch, done):
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
        predictions_next_state = self.predict_next_state(non_final_next_states, non_final_mask)
        # Compute the expected values. Do not add the reward, if the critic is split
        expected_value_next_state = (predictions_next_state * self.gamma) + (reward_batch if self.split else 0)
        return expected_value_next_state - current_prediction


class Q(TempDiffNet):
    def __init__(self, input_size, env, F_s, F_sa, device, log, hyperparameters, is_target_net=False):
        super(Q, self).__init__(env, device, log, hyperparameters, is_target_net)

        self.input_size = input_size
        self.hyperparameters = hyperparameters

        # can either have many outputs or one
        self.name = "Q"
        self.output_neurons = self.num_actions if not self.use_actor_critic else 1

        # Set up params:
        self.use_QV = hyperparameters["use_QV"]
        self.use_QVMAX = hyperparameters["use_QVMAX"]

        # Define params of previous net
        if is_target_net:
            updateable_parameters = []
        else:
            updateable_parameters = list(F_s.parameters()) + (list(F_sa.parameters()) if self.use_actor_critic else [])

        # Create split net:
        self.create_split_net(self.input_size, updateable_parameters, device, hyperparameters)

        # Create layers
        layers = hyperparameters["layers_Q"]
        self.layers_TD, self.act_functs_TD = create_ff_layers(self.input_size, layers, self.output_neurons)
        # Put feature extractor on GPU if possible:
        self.to(device)

        # Define optimizer and previous networks
        self.lr_TD = hyperparameters["lr_Q"]
        self.F_s = F_s
        self.F_sa = F_sa

        # TODO: the following does not work yet
        # with SummaryWriter(comment='Q') as w:
        #    random_input = torch.rand(self.input_size)
        #    w.add_graph(self, input_to_model=random_input, verbose=True)

        self.optimizer_TD = self.optimizer(list(self.layers_TD.parameters()) + updateable_parameters, lr=self.lr_TD)
        # Create target net
        self.target_net = self.create_target_net()

    def predict_next_state(self, non_final_next_state_features, non_final_mask, actor=None, Q=None, V=None):
        if self.use_QVMAX:
            return V.calculate_next_state_values(non_final_next_state_features, non_final_mask, actor=actor)
        elif self.use_QV:
            # This assumes that V is always trained directly before Q
            return V.expected_value_next_state
        else:
            return self.calculate_next_state_values(non_final_next_state_features, non_final_mask, actor=actor)

    def predict_current_state(self, state_features, state_action_features, actions):
        if not self.use_actor_critic:
            input_features = state_features
            if self.split:
                reward_prediction = self.forward_r(input_features).gather(1, actions)
            else:
                reward_prediction = 0
            value_prediction = self.forward_R(input_features).gather(1, actions)
            return value_prediction, reward_prediction
        else:
            input_features = state_action_features
            if self.split:
                reward_prediction = self.forward_r(input_features)
            else:
                reward_prediction = 0
            value_prediction = self.forward_R(input_features)
            return value_prediction, reward_prediction

    def predict_state_value(self, state_features, F_sa, actor):
        if not self.use_actor_critic:
            return self.forward(state_features).max(1)[0].unsqueeze(1)
        else:
            with torch.no_grad():
                action = actor(state_features)
                if self.discrete_env:
                    action = action.max(1)[1].unsqueeze(1)
                state_action_features = F_sa(state_features, action)
            # TODO: make sure whether these state-action_features are required somewhere else and store it if that is the case
            return self.predict_state_action_value(None, state_action_features, None)

    def predict_state_action_value(self, state_features, state_action_features, actions):
        if not self.use_actor_critic:
            return self.forward(state_features).gather(1, actions)  # .gather action that is taken
        else:
            return self.forward(state_action_features)  # self.F_s_A(state_features, actions))


# TODO: At the moment if we use Bellman split V and Q have separate reward networks (layers_r)... why?

class V(TempDiffNet):
    def __init__(self, input_size, env, F_s, F_sa, device, log, hyperparameters, is_target_net=False):
        super(V, self).__init__(env, device, log, hyperparameters, is_target_net=is_target_net)

        self.input_size = input_size
        self.F_sa = F_sa

        self.name = "V"
        self.output_neurons = 1

        self.use_QVMAX = hyperparameters["use_QVMAX"]

        # Create layers
        layers = hyperparameters["layers_V"]
        self.layers_TD, self.act_functs_TD = create_ff_layers(input_size, layers, self.output_neurons)
        # Put feature extractor on GPU if possible:
        self.to(device)

        # Define params of previous net
        if is_target_net:
            updateable_parameters = []
        else:
            updateable_parameters = list(F_s.parameters())

        # Create split net:
        self.create_split_net(input_size, updateable_parameters, device, hyperparameters)

        # Define optimizer and previous networks
        self.lr_TD = hyperparameters["lr_V"]
        self.F_s = F_s
        self.optimizer_TD = self.optimizer(list(self.layers_TD.parameters()) + updateable_parameters, lr=self.lr_TD)

        # Create target net
        self.target_net = self.create_target_net()

    def predict_next_state(self, non_final_next_states, non_final_mask, actor=None, Q=None, V=None):
        if self.use_QVMAX:
            return Q.calculate_next_state_values(non_final_next_states, non_final_mask, actor=actor)
        else:
            return self.calculate_next_state_values(non_final_next_states, non_final_mask, actor=actor)

    def predict_state_value(self, state_features, F_sa, actor):
        with torch.no_grad():
            return self(state_features)

    def predict_current_state(self, state_features, state_action_features, actions):
        reward_prediction = 0
        if self.split:
            reward_prediction = self.forward_r(state_features)
        return self.forward_R(state_features), reward_prediction



class Actor(OptimizableNet):
    def __init__(self, F_s, env, log, device, hyperparameters, is_target_net=False):
        super(Actor, self).__init__(env, device, log, hyperparameters, is_target_net=is_target_net)

        self.name = "Actor"

        # Initiate arrays for output function:
        self.relu_idxs = []
        self.sigmoid_idxs = []
        self.tanh_idxs = []
        self.scaling = None
        self.offset = None

        # Create layers
        input_size = F_s.layers_merge[-1].out_features
        output_size = self.num_actions if self.discrete_env else len(self.action_low)
        layers = hyperparameters["layers_actor"]
        self.layers, self.act_functs = create_ff_layers(input_size, layers, output_size)
        self.act_func_output_layer = self.create_output_act_func()
        # Put feature extractor on GPU if possible:
        self.to(device)

        # Define optimizer and previous networks
        self.lr = hyperparameters["lr_actor"]
        if not is_target_net:
            self.F_s = F_s
            updateable_parameters = list(self.F_s.parameters())
        else:
            updateable_parameters = []
        self.optimizer = self.optimizer(list(self.layers.parameters()) + updateable_parameters, lr=self.lr)

        if self.use_target_net:
            self.target_net = self.create_target_net()

    def forward(self, x):
        x = apply_layers(x, self.layers, self.act_functs)
        x = self.act_func_output_layer(x)
        # print(x)
        return x

    def compute_loss(self, output, target, sample_weights=None):
        # TODO: implement cross entropy loss: we need to have targets of type long
        if self.discrete_env:
            loss_func = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(output, target)
        else:
            # TODO: this loss does not help combat the vanishing gradient problem that we have because of the use of sigmoid activations to squash our actions into the correct range
            loss = F.smooth_l1_loss(output, target, reduction='none')

        if sample_weights is None:
            return loss.mean()
        else:
            return (loss * sample_weights).mean()

    def output_function_continuous(self, x):
        if self.relu_idxs:
            x[self.relu_idxs] = F.relu(x[self.relu_idxs])
        if self.sigmoid_idxs:
            x[self.sigmoid_idxs] = torch.sigmoid(x[self.sigmoid_idxs])
        if self.tanh_idxs:
            print("first: ", x)
            print(self.tanh_idxs)
            print(x[:, self.tanh_idxs])
            x[:, self.tanh_idxs] = torch.tanh(x[:, self.tanh_idxs])
            print("after: ", x)
        return (x * self.scaling) + self.offset

    def create_output_act_func(self):
        print("Action_space: ", self.env.action_space)
        if self.discrete_env:
            print("Actor has only sigmoidal activation function")
            print()
            return torch.sigmoid
        else:
            self.scaling = torch.ones(len(self.action_low))
            self.offset = torch.zeros(len(self.action_low))
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
            num_linear_actions = len(self.scaling) - len(self.tanh_idxs) - len(self.relu_idxs) - len(self.sigmoid_idxs)
            print("Actor has ", len(self.relu_idxs), " ReLU, ", len(self.tanh_idxs), " tanh, ", len(self.sigmoid_idxs),
                  " sigmoid, and ", num_linear_actions, " linear actions.")
            print("Action Scaling: ", self.scaling)
            print("Action Offset: ", self.offset)
            print()
        return self.output_function_continuous

    def optimize(self, transitions):
        state_batch = transitions["state"]
        state_features = transitions["state_features"]
        action_batch = transitions["action"]

        # Calculate current actions for state_batch:
        # torch.autograd.set_detect_anomaly(True)
        actions_current_state = self(state_features)
        better_actions_current_state = actions_current_state.detach().clone()
        # if self.discrete_env:
        #    action_batch = one_hot_encode(action_batch, self.num_actions)
        sample_weights = None

        if self.use_CACLA_V:
            # Requires TDE_V
            # Check which actions have a pos TDE
            pos_TDE_mask = (self.V.TDE < 0).squeeze()
            #better_actions_current_state[pos_TDE_mask] = action_batch[pos_TDE_mask]

            output = actions_current_state[pos_TDE_mask]
            target = action_batch[pos_TDE_mask].view(output.shape[0])
            sample_weights = self.V.TDE[pos_TDE_mask]
        if self.use_CACLA_Q:
            # Calculate mask of pos expected Q minus Q(s, mu(s))
            # action_TDE = self.Q.expectations_next_state - self.Q(state_features, actions_current_state).detach()
            pos_TDE_mask = (self.Q.TDE < 0).squeeze()
            # print(self.Q.TDE)
            # print("Actions predicted for current state: ", actions_current_state)
            # print("Action batch when updating: ", action_batch)
            # print("Better action initial : ", better_actions_current_state)
            # print("action batch of pos TDE mask: ", action_batch[pos_TDE_mask])

            # better_actions_current_state[pos_TDE_mask] = action_batch[pos_TDE_mask]

            output = actions_current_state[pos_TDE_mask]
            target = action_batch[pos_TDE_mask].view(output.shape[0])
            sample_weights = self.Q.TDE[pos_TDE_mask]
            # print("output: ", output)
            # print("target: ", target)
            # print("target shape: ", target.shape)

        # TODO: implement CACLA+Var

        # TODO - Idea: When using QV, possibly reinforce actions only if Q and V net agree (first check how often they disagree and in which cases)
        if self.use_DDPG:
            # 1. calculate derivative of Q towards actions 2. Reinforce towards actions plus gradients
            q_vals = self.Q(state_features, actions_current_state)
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
            with torch.no_grad():
                # TODO: either convert to max policy using the following line or pass raw output to F_sa and don't one-hot encode
                state_features_target = self.F_s.target_net(state_batch)
                actions_target_net = self.target_net(state_features_target)
                # print("Actions target net: ", actions_target_net)
                if self.discrete_env:
                    actions_current_policy = actions_target_net.argmax(1).unsqueeze(1)
                state_action_features_sampled_actions = self.Q.F_sa.target_net(state_features_target, action_batch)
                state_action_features_current_policy = self.Q.F_sa.target_net(state_features_target, actions_current_policy)
                Q_val_sampled_actions = self.Q.target_net(state_action_features_sampled_actions)
                Q_val_current_policy = self.Q.target_net(state_action_features_current_policy)
                action_TDE = Q_val_sampled_actions - Q_val_current_policy
                # print("action TDE: ", action_TDE)
            pos_TDE_mask = (action_TDE > 0).squeeze()

            # better_actions_current_state[pos_TDE_mask] = action_batch[pos_TDE_mask]

            output = actions_current_state[pos_TDE_mask]
            # print("Output: ", output)
            target = action_batch[pos_TDE_mask].view(output.shape[0])
            # print("Target: ", target)
            # sample_weights = action_TDE[pos_TDE_mask]

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

        # TODO: filter out actions where actions_current_state is equal to better_actions
        # self.optimize_net(actions_current_state, better_actions_current_state, self.optimizer, "actor")
        error = 0
        if len(output) > 0:
            error = self.optimize_net(output, target, self.optimizer, "actor", sample_weights=sample_weights)
        # Train actor towards better actions (loss = better - current)

        return error

    def recreate_self(self):
        return self.__class__(self.F_s, self.env, self.log, self.device, self.hyperparameters, is_target_net=True)
