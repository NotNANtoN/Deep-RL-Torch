import copy
import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from util import *


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
    channel_last_layer = input_matrix_shape[0]
    matrix_width = input_matrix_shape[1]
    matrix_height = input_matrix_shape[2]

    act_functs = []
    layers = nn.ModuleList()
    for layer in layer_dict:
        # Layer:
        if layer["name"] == "batchnorm":
            layers.append(nn.BatchNorm2d(channel_last_layer))
        elif layer["name"] == "conv":
            this_layer_channels = layer["filters"]
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

def calc_gradient_norm(layers):
    total_norm = 0
    for p in layers.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)

def calc_norm(layers):
    total_norm = torch.tensor(0.)
    for param in layers.parameters():
        total_norm += torch.norm(param)
    return total_norm

def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.get_updateable_params(), net.get_updateable_params()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def hard_update(net, net_target):
    for param_target, param in zip(net_target.get_updateable_params(), net.get_updateable_params()):
        param_target.data.copy_(param.data)

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
            self.action_low = torch.zeros(self.num_actions)
            self.action_high = torch.ones(self.num_actions)
        else:
            self.num_actions = len(env.action_space.low)
            self.action_low = torch.tensor(env.action_space.low)
            self.action_high = torch.tensor(env.action_space.high)

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
            reduced_loss = loss.mean()
        else:
            reduced_loss = (loss * sample_weights).mean()
        return loss, reduced_loss

    def optimize_net(self, output, target, optimizer, name="", sample_weights=None, retain_graph=False):
        #print("output: ", output)
        #print("target: ", target)
        loss, reduced_loss = self.compute_loss(output, target, sample_weights)

        optimizer.zero_grad()
        reduced_loss.backward(retain_graph=self.retain_graph + retain_graph)
        #if self.max_norm:
            #torch.nn.utils.clip_grad.clip_grad_norm_(self.parameters(), self.max_norm)
        optimizer.step()
        # TODO: we should test the option of not stepping the optimizer here, but instead returning the loss and only stepping once per policy for all networks at the same time (averaging the losses of all networks)

        name = "loss_" + self.name + (("_" + name) if name != "" else "")
        detached_loss = reduced_loss.detach().clone().item()
        self.log.add(name, detached_loss, skip_steps=100)
        #print("detached loss: ", detached_loss)
        #print("loss: ", loss)

        PER_weights = loss.detach().clone().cpu()

        # Log weight and gradient norms:
        if self.log.do_logging and self.log.log_NNs:
            self.log_nn_data()

        return PER_weights

    def get_updateable_params(self):
        return self.parameters()

    def update_targets(self, steps):
        if self.target_network_polyak:
            soft_update(self, self.target_net, self.tau)
        else:
            if steps % self.target_network_hard_steps == 0:
                hard_update(self, self.target_net)

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

    def log_layer_data(self, layers, name, extra_name=""):
        #weight_norm = calc_norm(layers)
        #grad_norm = calc_gradient_norm(layers)
        #self.log.add(name + " Weight Norm", weight_norm)
        #self.log.add(name + "_" + extra_name + " Grad Norm", grad_norm)
        name += "_" + extra_name + "_" if extra_name else ""
        weights = torch.cat([torch.flatten(layer).detach() for layer in layers.parameters()])
        #for layer in layers.parameters():
        #    print(layer.grad)
        gradients = torch.cat([torch.flatten(layer.grad.data).detach() for layer in layers.parameters()])
        self.log.add(name + "Weights", weights, distribution=True, skip_steps=10000)
        self.log.add(name + "Gradients", gradients, distribution=True, skip_steps=10000)

class ProcessState(OptimizableNet):
    def __init__(self, env, log, device, hyperparameters, is_target_net=False):
        super(ProcessState, self).__init__(env, device, log, hyperparameters)

        self.vector_layers = hyperparameters["layers_feature_vector"]
        self.matrix_layers = hyperparameters["layers_conv"]
        self.normalize_obs = hyperparameters["normalize_obs"]
        self.pin_tensors = hyperparameters["pin_tensors"]

        self.freeze_normalizer = False

        if is_target_net:
            self.use_target_net = False
        else:
            self.use_target_net = hyperparameters["use_target_net"]

        self.processing_list, merge_input_size = self.create_input_layers(env)

        # format for parameters: ["linear": (input, output neurons), "lstm": (input, output neurons)]
        merge_layers = hyperparameters["layers_feature_merge"]
        self.layers_merge, self.act_functs_merge = create_ff_layers(merge_input_size, merge_layers, None)

        # TODO: the following does not work yet, make it work at some point
        # self.log.writer.add_graph(self, input_to_model=[torch.rand(vector_len), None], verbose=True)

        # Set feature extractor to GPU if possible:
        self.to(device)

        self.target_net = self.create_target_net()
        if self.target_net is not None:
            for proc_dict, proc_dict_target in zip(self.processing_list, self.target_net.processing_list):
                proc_dict_target["Normalizer"] = proc_dict["Normalizer"]

    def apply_processing_dict(self, x, proc_dict):
        normalizer = proc_dict["Normalizer"]
        layers = proc_dict["Layers"]
        act_functs = proc_dict["Act_Functs"]
        batch_size = x.shape[0]
        if self.normalize_obs:
            if not self.freeze_normalizer:
                normalizer.observe(x)
            x = normalizer.normalize(x)
        x = apply_layers(x, layers, act_functs)
        return x.view(batch_size, -1)

    def apply_processing_list(self, x, proc_list):
        outputs = []
        if isinstance(x, dict):
            for key, proc_dict in zip(x, proc_list):
                # For e.g. MineRL we need to extract the obs from the key in-depth:
                obs = x[key]
                obs = self.apply_processing_dict(obs, proc_dict)
                outputs.append(obs)
        # If the obs is simply a torch tensor:
        else:
            x = self.apply_processing_dict(x, proc_list[0])
            outputs.append(x)
        return outputs

    def create_layer_dict(self, obs, name=""):
        """
        Creates a list of layers that operates on an observation in a dict of observations
        :param obs: numpy or PyTorch tensor
        :return: processing list
        """

        normalizer_device = self.device #None if self.pin_tensors else self.device

        # Create feedforward layers:
        if len(obs.shape) <= 1:
            layers_vector, act_functs_vector = create_ff_layers(len(obs), self.vector_layers, None)
            layers_vector.to(self.device)
            output_size = layers_vector[-1].out_features
            vector_normalizer = Normalizer(obs.shape, normalizer_device)

            # Add to lists:
            layer_dict = {"Layers": layers_vector, "Act_Functs": act_functs_vector, "Normalizer": vector_normalizer}
        # Create conv layers:
        elif 1 < len(obs.shape) <= 4:
            layers_matrix, output_size, act_functs_matrix = create_conv_layers(obs.shape, self.matrix_layers)
            layers_matrix.to(self.device)
            matrix_normalizer = Normalizer(obs.shape, normalizer_device)
            # Add to lists:
            layer_dict = {"Layers": layers_matrix, "Act_Functs": act_functs_matrix, "Normalizer": matrix_normalizer}
        else:
            raise NotImplementedError("Four dimensional input data not yet supported.")
        layer_dict["Name"] = name
        return layer_dict, output_size

    def create_input_layers(self, env):
        # Get a sample to assess the shape of the observations easily:
        obs_space = env.observation_space
        sample = obs_space.sample()
        if isinstance(sample, dict):
            sample = self.env.observation(sample)

        processing_list = []
        merge_input_size = 0
        # If the obs is a dict:
        if isinstance(sample, dict):
            for key in sample:
                obs = sample[key][0]
                layer_dict, output_size = self.create_layer_dict(obs, name=key)
                processing_list.append(layer_dict)
                merge_input_size += output_size
        # If the obs is simply a np array:
        elif isinstance(sample, np.ndarray):
            layer_dict, output_size = self.create_layer_dict(sample, name="input_array")
            processing_list.append(layer_dict)
            merge_input_size += output_size

        return processing_list, merge_input_size

    def forward(self, state):
        x = self.apply_processing_list(state, self.processing_list)
        x = torch.cat(x, dim=1)
        x = apply_layers(x, self.layers_merge, self.act_functs_merge)
        return x

    def forward_next_state(self, states):
        if self.use_target_net:
            return self.target_net(states)
        else:
            return self(states)

    def freeze_normalizers(self):
        self.freeze_normalizer = True
        self.target_net.freeze_normalizer = True
        for proc_dict in self.processing_list:
            proc_dict["Normalizer"].to(self.device)

    # observe a state to update the state normalizers:
    def observe(self, state):
        if isinstance(state, dict):
            apply_rec_to_dict(lambda x: x.to(self.device), state)
            for key, proc_dict in zip(state, self.processing_list):
                # For e.g. MineRL we need to extract the obs from the key in-depth:
                obs = state[key]
                normalizer = proc_dict["Normalizer"]
                if self.normalize_obs and not self.freeze_normalizer:
                    normalizer.observe(obs)
        # If the obs is simply a torch tensor:
        else:
            state = state.to(self.device)
            proc_dict = self.processing_list[0]
            normalizer = proc_dict["Normalizer"]
            if self.normalize_obs and not self.freeze_normalizer:
                normalizer.observe(state)

    def log_nn_data(self, name):
        for layers in self.processing_list:
            self.log_layer_data(layers["Layers"], "F_s-" + layers["Name"], extra_name=name)
        self.log_layer_data(self.layers_merge, "F_s Merge", extra_name=name)

    def recreate_self(self):
        return self.__class__(self.env, self.log, self.device, self.hyperparameters, is_target_net=True)

    def get_updateable_params(self):
        params = list(self.layers_merge.parameters())
        for layer in self.processing_list:
            params += list(layer["Layers"].parameters())
        return params

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.layers_merge, path + "merge.pth")
        for idx, layer in enumerate(self.processing_list):
            torch.save(layer["Layers"], path + layer["Name"] + ".pth")

    def load(self, path):
        loaded_merge = torch.load(path + "merge.pth")
        self.layers_merge = loaded_merge
        for layer in self.processing_list:
            loaded_model = torch.load(path + layer["Name"] + ".pth")
            layer["Layers"] = loaded_model

class ProcessStateAction(OptimizableNet):
    def __init__(self, state_features_len, env, log, device, hyperparameters, is_target_net=False):
        super(ProcessStateAction, self).__init__(env, device, log, hyperparameters, is_target_net)

        self.state_features_len = state_features_len
        self.freeze_normalizer = False

        # Create layers:
        # Action Embedding
        layers_action = hyperparameters["layers_action"]
        self.layers_action, self.act_functs_action = create_ff_layers(self.num_actions, layers_action, None)

        # State features and Action features concat:
        input_size = state_features_len + self.layers_action[-1].out_features
        layers = hyperparameters["layers_state_action_merge"]
        self.layers_merge, self.act_functs_merge = create_ff_layers(input_size, layers, None)

        # self.log.writer.add_graph(self, input_to_model=torch.rand(state_features_len), verbose=True)

        # Put feature extractor on GPU if possible:
        self.to(device)

        self.target_net = self.create_target_net()

    def forward(self, state_features, actions, apply_one_hot_encoding=True):
        # if not self.use_actor_critic and apply_one_hot_encoding:
        #    actions = one_hot_encode(actions, self.num_actions)
        actions = apply_layers(actions, self.layers_action, self.act_functs_action)
        x = torch.cat((state_features, actions), 1)
        x = apply_layers(x, self.layers_merge, self.act_functs_merge)
        return x

    def forward_next_state(self, state_features, action):
        if self.use_target_net:
            return self.target_net(state_features, action)
        else:
            return self(state_features, action)

    def log_nn_data(self, name=""):
        self.log_layer_data(self.layers_action, "F_sa_Action", extra_name=name)
        self.log_layer_data(self.layers_merge, "F_sa_Merge", extra_name=name)

    def recreate_self(self):
        return self.__class__(self.state_features_len, self.env, self.log, self.device, self.hyperparameters,
                              is_target_net=True)

    def get_updateable_params(self):
        params = list(self.layers_merge.parameters())
        params += list(self.layers_action.parameters())
        return params

    def freeze_normalizers(self):
        self.freeze_normalizer = True
        self.target_net.freeze_normalizer = True


    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.layers_merge, path + "merge.pth")
        torch.save(self.layers_action, path + "action.pth")

    def load(self, path):
        loaded_merge = torch.load(path + "merge.pth")
        self.layers_merge = loaded_merge
        loaded_action = torch.load(path + "action.pth")
        self.layers_action = loaded_action


class TempDiffNet(OptimizableNet):
    def __init__(self, env, device, log, hyperparameters, is_target_net=False):
        super(TempDiffNet, self).__init__(env, device, log, hyperparameters)

        self.use_actor_critic = hyperparameters["use_actor_critic"]
        self.split = hyperparameters["split_Bellman"]
        self.use_target_net = hyperparameters["use_target_net"] if not is_target_net else False
        self.gamma = hyperparameters["gamma"]

        self.current_reward_prediction = None

        # Eligibility traces:
        self.use_efficient_traces = hyperparameters["use_efficient_traces"]
        if self.use_efficient_traces:
            self.traces = torch.empty(hyperparameters["replay_buffer_size"] + hyperparameters["num_expert_samples"],
                                      device=device)

    def create_split_net(self, input_size, updateable_parameters, device, hyperparameters):
        lr_r = hyperparameters["lr_r"]
        reward_layers = hyperparameters["layers_r"]
        layers_r, act_functs_r = create_ff_layers(input_size, reward_layers, self.output_neurons)
        layers_r.to(device)
        optimizer_r = self.optimizer(list(layers_r.parameters()) + updateable_parameters, lr=lr_r)
        return layers_r, act_functs_r, optimizer_r


    def recreate_self(self):
        new_self = self.__class__(self.input_size, self.env, None, None, self.device, self.log,
                                  self.hyperparameters, is_target_net=True)
        if self.split:
            new_self.layers_r = self.layers_r
        return new_self

    def forward(self, x):
        predicted_reward = 0
        if self.split:
            predicted_reward = apply_layers(x, self.layers_r, self.act_functs_r)
        predicted_state_value = apply_layers(x, self.layers_TD, self.act_functs_TD)
        return predicted_state_value + predicted_reward

    def forward_r(self, x):
        return apply_layers(x, self.layers_r, self.act_functs_r)

    def forward_R(self, x):
        return apply_layers(x, self.layers_TD, self.act_functs_TD)

    def calculate_next_state_values(self, non_final_next_state_features, non_final_mask, actor=None, use_target_net=True):
        next_state_values = torch.zeros(len(non_final_mask), 1, device=self.device)
        if non_final_next_state_features is None:
            return next_state_values
        with torch.no_grad():
            predict_net = self.target_net if use_target_net else self
            next_state_predictions = predict_net.predict_state_value(non_final_next_state_features, self.F_sa,
                                                                         actor)
        next_state_values[non_final_mask] = next_state_predictions
        return next_state_values

    def calculate_updated_value_next_state(self, reward_batch, non_final_next_state_features, non_final_mask,
                                           actor=None, Q=None, V=None):
        # Compute V(s_t+1) or max_aQ(s_t+1, a) for all next states.
        predictions_next_state = self.predict_next_state(non_final_next_state_features, non_final_mask, actor, Q,
                                                              V)
        # self.log.add(self.name + " Prediction_next_state", self.predictions_next_state[0].item())
        # print("Next state features: ", non_final_next_state_features)
        # print("Prediction next state: ", predictions_next_state)

        # Compute the updated expected values. Do not add the reward, if the critic is split
        return (reward_batch if not self.split else 0) + (self.predictions_next_state * self.gamma)

    def update_traces(self, episode_transitions, lambda_val, actor=None, V=None, Q=None):
        num_steps_in_episode = len(episode_transitions["state"])
        non_final_next_state_features = episode_transitions["non_final_next_state_features"]
        non_final_mask = episode_transitions["non_final_mask"]
        rewards = episode_transitions["reward"]
        idxs = episode_transitions["idxs"]

        # Pre-calculate next-state-values in a large batch:
        next_state_vals = None
        if non_final_next_state_features is not None:
            next_state_vals = self.predict_next_state(non_final_next_state_features, non_final_mask, actor=actor, Q=Q,
                                                      V=V, use_target_net=False)

        traces = torch.empty(num_steps_in_episode)
        last_trace_value = 0
        # Iterate backwards through transitions in the episode:
        for step_idx in range(0, num_steps_in_episode):
            reversed_idx = num_steps_in_episode - 1 - step_idx
            current_trace_val = rewards[reversed_idx].clone()
            if non_final_mask[reversed_idx]:
                current_trace_val += self.gamma * (lambda_val * last_trace_value +
                                                  (1 - lambda_val) * next_state_vals[reversed_idx][0])
            traces[reversed_idx] = current_trace_val
            last_trace_value = current_trace_val

        # If split the direct reward prediction is taken care of another network
        if self.split:
            traces -= rewards.squeeze()
            #traces = [trace - rewards[idx] for idx, trace in enumerate(traces)]

        self.traces[idxs] = traces

    def optimize(self, transitions, importance_weights, actor=None, Q=None, V=None, policy_name=""):
        state_features = transitions["state_features"]
        state_action_features = transitions["state_action_features"]
        action_batch = transitions["action_argmax"]
        reward_batch = transitions["reward"]
        non_final_next_state_features = transitions["non_final_next_state_features"]
        non_final_mask = transitions["non_final_mask"]
        idxs = transitions["idxs"]

        # Compute V(s_t) or Q(s_t, a_t)
        predictions_current, reward_prediction = self.predict_current_state(state_features, state_action_features,
                                                                            action_batch)

        # Train reward net if it exists:
        if self.split:
            print("r:")
            print(self.optimizer_r.state_dict()["state"].keys())
            TDE_r = self.optimize_net(reward_prediction, reward_batch, self.optimizer_r, "r", retain_graph=True)
            #self.log_nn_data(policy_name + "_r-net_", r_net=True)
        else:
            TDE_r = 0

        # Compute the expected values. Do not add the reward, if the critic is split
        if self.use_efficient_traces:
            expected_value_next_state = self.traces[idxs].unsqueeze(1)
            # TODO: A possible extension could be to update the traces of the sampled transitions in this method
        else:
            expected_value_next_state = self.calculate_updated_value_next_state(reward_batch,
                                                                                non_final_next_state_features,
                                                                                non_final_mask, actor, Q, V)
        #print(predictions_current)
        #print("expected_val: ", expected_value_next_state)
        #print()
        # TD must be stored for actor-critic updates:
        #print("TD:")
        #print(self.optimizer_TD.state_dict()["state"].keys())

        TDE_TD = self.optimize_net(predictions_current, expected_value_next_state, self.optimizer_TD, "TD",
                          sample_weights=importance_weights)
        self.log_nn_data(policy_name + "_TD-net", r_net=False)
        self.TDE = (abs(TDE_r) + abs(TDE_TD))
        return self.TDE

    def calculate_Q_and_TDE(self, state, action, next_state, reward, done, actor=None, Q=None, V=None):
        tde = 0
        state_features = self.F_s(state.unsqueeze(0))
        if self.F_sa is None:
            state_action_features = None
        else:
            state_action_features = self.F_sa(state_features, action.unsqueeze(0))
        predictions_current, reward_prediction = self.predict_current_state(state_features, state_action_features,
                                                                            action)
        q_val = predictions_current + reward_prediction
        # TDE for split:
        if self.split:
            tde += reward - reward_prediction
        # TDE for normal:
        non_final_mask = [False if done else True]
        non_final_next_state_features = self.F_s(state.unsqueeze(0)) if not done else None
        expected_value_next_state = self.calculate_updated_value_next_state(reward, non_final_next_state_features,
                                                                            non_final_mask, actor, Q, V)
        tde += expected_value_next_state - predictions_current

        return q_val, tde


    def log_nn_data(self, name="", r_net=False):
        if not r_net:
            self.log_layer_data(self.layers_TD, self.name + "_TD", extra_name=name)
        if r_net and self.split:
            self.log_layer_data(self.layers_r, self.name + "_r", extra_name=name)
        if self.F_s is not None:
            self.F_s.log_nn_data(self.name + name)
        if self.F_sa is not None:
            self.F_sa.log_nn_data(self.name + name)

    def get_updateable_params(self):
        params = list(self.layers_TD.parameters())
        if self.split:
            params += list(self.layers_r.parameters())
        return params

    def weights_init(self, m):
        # if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)

    def predict_current_state(self, state_features, state_action_features, actions):
        raise NotImplementedError

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.layers_TD, path + "TD.pth")
        if self.split:
            torch.save(self.layers_r, path + "r.pth")

    def load(self, path):
        loaded_TD = torch.load(path + "TD.pth")
        self.layers_TD = loaded_TD
        loaded_r = torch.load(path + "r.pth")
        self.layers_r = loaded_r


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
            updateable_parameters = list(F_s.get_updateable_params()) + (list(F_sa.get_updateable_params()) if self.use_actor_critic else [])

        # Create split net:
        if self.split and not is_target_net:
            self.layers_r, self.act_functs_r, self.optimizer_r = self.create_split_net(self.input_size,
                                                                                       updateable_parameters,
                                                                                       device, hyperparameters)
            #print("RRRRRR:")
            #print(list(self.layers_r.state_dict().keys()) + list(F_s.get_updateable_params()))
            #print(type(next(self.layers_r.parameters())))
        else:
            self.layers_r, self.act_functs_r, self.optimizer_r = None, None, None


            # Create layers
        layers = hyperparameters["layers_Q"]
        self.layers_TD, self.act_functs_TD = create_ff_layers(self.input_size, layers, self.output_neurons)
        # Put feature extractor on GPU if possible:
        self.layers_TD.to(device)

        # Define optimizer and previous networks
        self.lr_TD = hyperparameters["lr_Q"]
        self.F_s = F_s
        self.F_sa = F_sa

        # TODO: the following does not work yet
        # with SummaryWriter(comment='Q') as w:
        #    random_input = torch.rand(self.input_size)
        #    w.add_graph(self, input_to_model=random_input, verbose=True)
        #if not is_target_net:
        #    print("TD::::")
        #    print(list(self.layers_TD.state_dict().keys()) + list(F_s.state_dict().keys()))

        self.optimizer_TD = self.optimizer(list(self.layers_TD.parameters()) + updateable_parameters, lr=self.lr_TD)
        # Create target net
        self.target_net = self.create_target_net()
        if self.target_net and self.split:
            self.target_net.layers_r = self.layers_r
            self.target_net.act_functs_r = self.act_functs_r


    def predict_next_state(self, non_final_next_state_features, non_final_mask, actor=None, Q=None, V=None, use_target_net=True):
        if self.use_QVMAX:
            self.predictions_next_state = V.calculate_next_state_values(non_final_next_state_features, non_final_mask, actor=actor, use_target_net=use_target_net)
        elif self.use_QV:
            # This assumes that V is always trained directly before Q
            self.predictions_next_state = V.predictions_next_state
        else:
            self.predictions_next_state = self.calculate_next_state_values(non_final_next_state_features,
                                                                           non_final_mask, actor=actor,
                                                                           use_target_net=use_target_net)
        return self.predictions_next_state

    def predict_current_state(self, state_features, state_action_features, actions):
        if not self.use_actor_critic:
            input_features = state_features
            if self.split:
                reward_prediction = self.forward_r(input_features).gather(1, actions)
            else:
                reward_prediction = 0
            #print("input feature shape: ", input_features.shape)
            #print("forward _R shape: ", self.forward_R(input_features).shape)
            #print("num actions: ", self.num_actions)
            #print("actions: ", actions)
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
                # if self.discrete_env:
                #    action = action.max(1)[1].unsqueeze(1)
                state_action_features = F_sa(state_features, action)
            # TODO: make sure whether these state-action_features are required somewhere else and store it if that is the case
            return self.predict_state_action_value(None, state_action_features, None)

    def predict_state_action_value(self, state_features, state_action_features, actions):
        if not self.use_actor_critic:
            return self.forward(state_features).gather(1, actions)  # .gather action that is taken
        else:
            return self.forward(state_action_features)  # self.F_s_A(state_features, actions))


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
            updateable_parameters = list(F_s.get_updateable_params())

        # Create split net:
        if self.split and not is_target_net:
            self.layers_r, self.act_functs_r, self.optimizer_r = self.create_split_net(self.input_size,
                                                                                       updateable_parameters,
                                                                                       device, hyperparameters)
        else:
            self.layers_r, self.act_functs_r, self.optimizer_r = None, None, None

        # Define optimizer and previous networks
        self.lr_TD = hyperparameters["lr_V"]
        self.F_s = F_s
        self.optimizer_TD = self.optimizer(list(self.get_updateable_params()) + updateable_parameters, lr=self.lr_TD)

        # Create target net
        self.target_net = self.create_target_net()
        if self.target_net and self.split:
            self.target_net.layers_r = self.layers_r
            self.target_net.act_functs_r = self.act_functs_r


    def predict_next_state(self, non_final_next_states, non_final_mask, actor=None, Q=None, V=None, use_target_net=True):
        if self.use_QVMAX:
            self.predictions_next_state = Q.calculate_next_state_values(non_final_next_states, non_final_mask, actor=actor, use_target_net=use_target_net)
        else:
            self.predictions_next_state = self.calculate_next_state_values(non_final_next_states, non_final_mask, actor=actor, use_target_net=use_target_net)
        return self.predictions_next_state

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
        self.relu_mask = None
        self.sigmoid_mask = None
        self.tanh_mask = None
        self.relu_idxs = None
        self.tanh_idxs = None
        self.sigmoid_idxs = None
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
            updateable_parameters = list(self.F_s.get_updateable_params())
        else:
            updateable_parameters = []
        self.optimizer = self.optimizer(list(self.get_updateable_params()) + updateable_parameters, lr=self.lr)

        if self.use_target_net:
            self.target_net = self.create_target_net()

    def forward(self, x):
        x = apply_layers(x, self.layers, self.act_functs)
        x = self.act_func_output_layer(x)
        # print(x)
        return x

    def compute_loss(self, output, target, sample_weights=None):
        # TODO: test if actor training might be better without CrossEntropyLoss. It might be, because we do not need to convert to long!
        if self.use_DDPG:
            loss = abs(target - output)
        elif self.discrete_env:
            loss_func = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(output, target)
        else:
            # TODO: this loss does not help combat the vanishing gradient problem that we have because of the use of sigmoid activations to squash our actions into the correct range
            loss = F.smooth_l1_loss(output, target, reduction='none')

        if sample_weights is not None:
            loss *= sample_weights.squeeze()
        return loss.mean()

    def output_function_continuous(self, x):
        if self.relu_idxs:
            # x[self.relu_idxs] = F.relu(x[self.relu_idxs])
            y = F.relu(x[:, self.relu_idxs])
            with torch.no_grad():
                x[:, self.relu_idxs] = y
        if self.sigmoid_idxs:
            # x[self.sigmoid_idxs] = torch.sigmoid(x[self.sigmoid_idxs])
            y = x[:, self.sigmoid_idxs].sigmoid()
            with torch.no_grad():
                x[:, self.sigmoid_idxs] = y
        if self.tanh_idxs:
            # print("first: ", x)
            # print(self.tanh_idxs)
            # print(x[:, self.tanh_idxs])
            y = x[:, self.tanh_idxs].tanh()
            # x[self.tanh_mask] = torch.tanh(x[self.tanh_mask])
            # print("after: ", y)
            with torch.no_grad():
                x[:, self.tanh_idxs] = y
            # print("inserted: ", x)
        #print("Action: ", x)
        return (x * self.scaling) + self.offset

    def create_output_act_func(self):
        print("Action_space: ", self.env.action_space)
        relu_idxs = []
        tanh_idxs = []
        sigmoid_idxs = []

        # Init masks:
        # self.relu_mask = torch.zeros(self.batch_size, len(self.action_low))
        # self.relu_mask.scatter_(1, torch.tensor(relu_idxs).long(), 1.)
        # self.tanh_mask = torch.zeros(self.batch_size, len(self.action_low))
        # self.tanh_mask.scatter_(1, torch.tensor(tanh_idxs).long(), 1.)
        # self.sigmoid_mask = torch.zeros(self.batch_size, len(self.action_low))
        # self.sigmoid_mask.scatter_(1, torch.tensor(sigmoid_idxs).long(), 1.)

        if self.discrete_env:
            print("Actor has only sigmoidal activation function")
            print()
            return torch.sigmoid
        else:
            self.scaling = torch.ones(len(self.action_low))
            self.offset = torch.zeros(len(self.action_low))
            for i in range(len(self.action_low)):
                low = self.action_low[i]
                high = self.action_high[i]
                if not (low and high):
                    if low == -math.inf or high == math.inf:
                        relu_idxs.append(i)
                        # self.relu_mask[i] = 1.0
                    else:
                        sigmoid_idxs.append(i)
                        # self.sigmoid_mask[i] = 1.0
                        self.scaling[i] = high + low
                elif low == high * -1:
                    if low != -math.inf:
                        tanh_idxs.append(i)
                        # self.tanh_mask[i] = 1.0
                        self.scaling[i] = high
                else:
                    self.offset[i] = (high - low) / 2
                    self.scaling[i] = high - offset[i]
                    tanh_idxs.append(i)
            num_linear_actions = len(self.scaling) - len(tanh_idxs) - len(relu_idxs) - len(sigmoid_idxs)
            print("Actor has ", len(relu_idxs), " ReLU, ", len(tanh_idxs), " tanh, ", len(sigmoid_idxs),
                  " sigmoid, and ", num_linear_actions, " linear actions.")
            print("Action Scaling: ", self.scaling)
            print("Action Offset: ", self.offset)
            print()

        self.tanh_idxs = tanh_idxs
        self.relu_idxs = relu_idxs
        self.sigmoid_idxs = sigmoid_idxs

        return self.output_function_continuous

    def optimize(self, transitions, policy_name=""):
        # Only for debugging:
        # torch.autograd.set_detect_anomaly(True)

        state_batch = transitions["state"]
        state_features = transitions["state_features"]
        action_batch = transitions["action"]

        # TODO: also do it for SPG?
        if self.discrete_env and self.use_CACLA_V or self.use_CACLA_Q:
            transformed_action_batch = torch.argmax(action_batch, dim=1)

        # Calculate current actions for state_batch:
        actions_current_state = self(state_features)
        better_actions_current_state = actions_current_state.detach().clone()
        # if self.discrete_env:
        #    action_batch = one_hot_encode(action_batch, self.num_actions)
        sample_weights = None

        if self.use_CACLA_V:
            # Check which actions have a pos TDE
            pos_TDE_mask = (self.V.TDE < 0).squeeze()
            output = actions_current_state[pos_TDE_mask]

            if self.discrete_env:
                target = transformed_action_batch[pos_TDE_mask].view(output.shape[0])
            else:
                target = action_batch[pos_TDE_mask]

            # TODO: investigate why the multiplication by minus one is necessary for sample weights... seems to be for the V.TDE < 0 check. Using all actions with sample weights = TDE also works, but worse in cartpole
            # TODO: also investigate whether scaling by TDE can be beneficial. Both works at least with V.TDE < 0
            sample_weights = -1 * torch.ones(target.shape)  # .unsqueeze(1) #
            # sample_weights = self.V.TDE[pos_TDE_mask].view(output.shape)

            # print(output)
            # print(target)
            # print()
            # print(sample_weights)

        if self.use_CACLA_Q:
            # Calculate mask of pos expected Q minus Q(s, mu(s))
            # action_TDE = self.Q.expectations_next_state - self.Q(state_features, actions_current_state).detach()
            pos_TDE_mask = (self.Q.TDE < 0).squeeze()

            output = actions_current_state[pos_TDE_mask]

            if self.discrete_env:
                target = transformed_action_batch[pos_TDE_mask].view(output.shape[0])
            else:
                target = action_batch[pos_TDE_mask]

            # sample_weights = -1 * torch.ones(output.shape[0])
            sample_weights = self.Q.TDE[pos_TDE_mask].view(target.shape[0])

        # TODO: implement CACLA+Var

        # TODO - Idea: When using QV, possibly reinforce actions only if Q and V net agree (first check how often they disagree and in which cases)
        if self.use_DDPG:
            # Dirty and fast way (still does not work yet... :-( )
            q_vals = -self.Q(self.Q.F_sa(state_features, actions_current_state)).mean()
            self.optimizer.zero_grad()
            q_vals.backward()
            self.optimizer.step()
            return q_vals.detach()

            # 1. calculate derivative of Q towards actions 2. Reinforce towards actions plus gradients
            actions_current_state_detached = Variable(actions_current_state.detach(), requires_grad=True)
            state_action_features_current_policy = self.Q.F_sa(state_features, actions_current_state_detached)
            q_vals = self.Q(state_action_features_current_policy)
            actor_loss = q_vals.mean() * -1
            actor_loss.backward(retain_graph=True)  # retain necessary? I thnk so
            gradients = actions_current_state_detached.grad
            self.log.add("DDPG Action Gradient", gradients.mean(), skip_steps=10000)

            # Normalize gradients:
            # gradients = self.normalize_gradients(gradients)
            # TODO: maybe normalize within the actor optimizer...?
            # TODO Normalize over batch, then scale by inverse TDE (risky thing:what about very small TDEs?
            output = actions_current_state
            target = (actions_current_state.detach().clone() + gradients)

            # Clip actions
            target = torch.max(torch.min(target, self.action_high), self.action_low)

            # sample_weights = torch.ones(target.shape[0]).unsqueeze(1) / abs(self.Q.TDE)

            # print(sample_weights)
            # print(output)
            # print(gradients)

        if self.use_SPG:
            # Calculate mask of Q(s,a) minus Q(s, mu(s))
            with torch.no_grad():
                # TODO: either convert to max policy using the following line or pass raw output to F_sa and don't one-hot encode
                state_features_target = self.F_s.target_net(state_batch)
                actions_target_net = self.target_net(state_features_target)
                # print("Actions target net: ", actions_target_net)
                # if self.discrete_env:
                #    actions_current_policy = actions_target_net.argmax(1).unsqueeze(1)
                state_action_features_sampled_actions = self.Q.F_sa.target_net(state_features_target, action_batch)
                state_action_features_current_policy = self.Q.F_sa.target_net(state_features_target,
                                                                              actions_current_policy,
                                                                              apply_one_hot_encoding=False)
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
            sample_weights = action_TDE[pos_TDE_mask]

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
            pass

        # self.optimize_net(actions_current_state, better_actions_current_state, self.optimizer, "actor")

        # print("output", output)
        # print(target)
        # if not self.discrete_env:
        #    target = target.unsqueeze(1)

        error = 0
        if len(output) > 0:
            # Train actor towards better actions (loss = better - current)
            error = self.optimize_net(output, target, self.optimizer, sample_weights=sample_weights)
            self.log_nn_data(policy_name)

        else:
            pass
            # TODO: log for CACLA Q and CACLA V and SPG on how many actions per batch is trained
            # print("No Training for Actor...")

        if self.use_CACLA_V or self.use_CACLA_Q or self.use_SPG:
            self.log.add("Actor_actual_train_batch_size", len(output), skip_steps=10000)

        return error

    def log_nn_data(self, name=""):
        self.log_layer_data(self.layers, "Actor", extra_name=name)
        if self.F_s is not None:
            self.F_s.log_nn_data("_Actor_" + name)


    def recreate_self(self):
        return self.__class__(self.F_s, self.env, self.log, self.device, self.hyperparameters, is_target_net=True)

    def get_updateable_params(self):
        return self.layers.parameters()

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.layers, path + "actor.pth")


    def load(self, path):
        loaded_model= torch.load(path + "actor.pth")
        self.layers = loaded_model

