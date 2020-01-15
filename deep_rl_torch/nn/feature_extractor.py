import os

import torch

from .networks import OptimizableNet
from .nn_utils import *
from .normalizer import Normalizer

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
        print("merge in size:; ", merge_input_size)
        print("merge layers: ", merge_layers)
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
        if obs.ndim <= 2:
            if obs.ndim == 2:
                obs = obs.squeeze(0)
            layers_vector, act_functs_vector = create_ff_layers(len(obs), self.vector_layers, None)
            layers_vector.to(self.device)
            output_size = layers_vector[-1].out_features
            vector_normalizer = Normalizer(obs.shape, normalizer_device)

            # Add to lists:
            layer_dict = {"Layers": layers_vector, "Act_Functs": act_functs_vector, "Normalizer": vector_normalizer}
        # Create conv layers:
        elif 2 < obs.ndim <= 4:
            if obs.ndim == 4:
                obs = obs.squeeze(0)
            elif obs.ndim == 2:
                obs.unsqueeze(0)
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
        print("Creating state processor input layers...")
        # Get a sample to assess the shape of the observations easily:
        obs_space = env.observation_space
        sample = obs_space.sample()
        #print(sample.shape)
        #sample = self.env.observation(sample)
        #print("after: ", sample.shape)
        # TODO: the above must be adjusted for MineRL envs - or the obs space in those wrappers must be fixed accordingly


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
        else:
            layer_dict, output_size = self.create_layer_dict(sample, name="input_array")
            processing_list.append(layer_dict)
            merge_input_size += output_size

        return processing_list, merge_input_size

    def forward(self, state):
        #print(state.shape)
        x = self.apply_processing_list(state, self.processing_list)
        x = torch.cat(x, dim=1)
        #print(x.shape)
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
            state = apply_rec_to_dict(lambda x: x.to(self.device, non_blocking=True), state)
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

