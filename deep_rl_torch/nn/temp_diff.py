import os

import torch
from .networks import OptimizableNet
from .nn_utils import *

class TempDiffNet(OptimizableNet):
    def __init__(self, env, device, log, hyperparameters, is_target_net=False):
        super(TempDiffNet, self).__init__(env, device, log, hyperparameters)

        self.use_actor_critic = hyperparameters["use_actor_critic"]
        self.split = hyperparameters["split_Bellman"]
        self.use_target_net = hyperparameters["use_target_net"] if not is_target_net else False
        self.gamma = hyperparameters["gamma"]
        self.dtype = torch.half if hyperparameters["use_half"] else torch.float

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
        optimizer_r = self.optimizer(list(layers_r.params()) + updateable_parameters, lr=lr_r)
        return layers_r, act_functs_r, optimizer_r


    def recreate_self(self):
        new_self = self.__class__(self.input_size, self.env, None, None, self.device, self.log,
                                  self.hyperparameters, is_target_net=True)
        #if self.split:
        #    new_self.layers_r = self.layers_r
        return new_self

    def forward(self, x):
        predicted_reward = 0
        if self.split:
            predicted_reward = self.forward_r(x)  # apply_layers(x, self.layers_r, self.act_functs_r)
        predicted_state_value = self.forward_R(x)  # apply_layers(x, self.layers_TD, self.act_functs_TD)
        return predicted_state_value + predicted_reward

    def forward_r(self, x):
        return apply_layers(x, self.layers_r, self.act_functs_r)

    def forward_R(self, x):
        return apply_layers(x, self.layers_TD, self.act_functs_TD)

    def calculate_next_state_values(self, non_final_next_state_features, non_final_mask, actor=None, use_target_net=True):
        next_state_values = torch.zeros(len(non_final_mask), 1, device=self.device, dtype=self.dtype)
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
        self.predictions_next_state = self.predict_next_state(non_final_next_state_features, non_final_mask, actor, Q,
                                                              V)
        # self.log.add(self.name + " Prediction_next_state", self.predictions_next_state[0].item())
        # print("Next state features: ", non_final_next_state_features)
        # print("Prediction next state: ", predictions_next_state)

        # Compute the updated expected values. Do not add the reward, if the critic is split
        return (reward_batch if not self.split else 0) + (self.predictions_next_state * self.gamma)

    def update_traces(self, episode_transitions, lambda_val, actor=None, V=None, Q=None, last_trace_value=None):
        num_steps_in_episode = len(episode_transitions["states"])
        non_final_next_state_features = episode_transitions["non_final_next_state_features"]
        non_final_mask = episode_transitions["non_final_mask"]
        rewards = episode_transitions["rewards"]
        idxs = episode_transitions["idxs"]

        # Pre-calculate next-state-values in a large batch:
        with torch.no_grad():
            next_state_vals = self.predict_next_state(non_final_next_state_features, non_final_mask, actor=actor, Q=Q,
                                                          V=V, use_target_net=False)
        
        traces = torch.empty(num_steps_in_episode, device=self.device)
        if last_trace_value is None:
            last_trace_value = 0
        
        for idx in range(num_steps_in_episode - 1, -1, -1):
            current_trace_val = rewards[idx].clone()
            if non_final_mask[idx]:
                current_trace_val += self.gamma * (lambda_val * last_trace_value +
                                                  (1 - lambda_val) * next_state_vals[idx][0])
            traces[idx] = current_trace_val
            last_trace_value = current_trace_val

        # If split the direct reward prediction is taken care of another network:
        if self.split:
            traces -= rewards.squeeze()
            #traces = [trace - rewards[idx] for idx, trace in enumerate(traces)]

        self.traces[idxs] = traces
        return last_trace_value

    def optimize(self, transitions, actor=None, Q=None, V=None, policy_name=""):
        state_features = transitions["state_features"]
        if self.F_sa is not None:
            state_action_features = transitions["state_action_features"]
        else:
            state_action_features = None
        action_batch = transitions["action_argmax"]
        reward_batch = transitions["rewards"]
        non_final_next_state_features = transitions["non_final_next_state_features"]
        non_final_mask = transitions["non_final_mask"]
        idxs = transitions["idxs"]
        importance_weights = None
        if "importance_weights" in transitions:
            importance_weights = transitions["importance_weights"]

        # Compute V(s_t) or Q(s_t, a_t)
        predictions_current, reward_prediction = self.predict_current_state(state_features, state_action_features,
                                                                            action_batch)
        
        #print("Shape pred current: ", predictions_current.shape)

        # Train reward net if it exists:
        if self.split:
            TDE_r, loss_r = self.optimize_net(reward_prediction, reward_batch, self.optimizer_r, "r", retain_graph=True)
            #self.log_nn_data(policy_name + "_r-net_", r_net=True)
        else:
            TDE_r = 0
            loss_r = 0

        # Compute the expected values. Do not add the reward, if the critic is split
        if self.use_efficient_traces:
            expected_value_next_state = self.traces[idxs].unsqueeze(1)
            # TODO: A possible extension could be to update the traces of the sampled transitions in this method
        else:
            expected_value_next_state = self.calculate_updated_value_next_state(reward_batch,
                                                                                non_final_next_state_features,
                                                                                non_final_mask, actor, Q, V)
        #print("shape next state pred: ", expected_value_next_state.shape)
        #print(predictions_current)
        #print("expected_val: ", expected_value_next_state)
        #print()
        # TD must be stored for actor-critic updates:
        #print("TD:")
        #print(self.optimizer_TD.state_dict()["states"].keys())

        TDE_TD, loss_TD = self.optimize_net(predictions_current, expected_value_next_state, self.optimizer_TD, "TD",
                          sample_weights=importance_weights)

        self.TDE = (abs(TDE_r) + abs(TDE_TD))
        loss = loss_TD + loss_r
        return self.TDE, loss

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
        self.log_layer_data(self.layers_TD, self.name + "_TD", extra_name=name)
        if self.split:
            self.log_layer_data(self.layers_r, self.name + "_r", extra_name=name)
        #if self.F_s is not None:
        #    self.F_s.log_nn_data(self.name + name)
        #if self.F_sa is not None:
        #    self.F_sa.log_nn_data(self.name + name)

    def get_updateable_params(self):
        params = list(self.layers_TD.params())
        if self.split:
            params += list(self.layers_r.params())
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
            #print(type(next(self.layers_r.params())))
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

        self.optimizer_TD = self.optimizer(list(self.layers_TD.params()) + updateable_parameters, lr=self.lr_TD)
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
            #print("actions shape: ", actions.shape)
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
        self.optimizer_TD = self.optimizer(list(self.layers_TD.params()) + updateable_parameters, lr=self.lr_TD)

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

