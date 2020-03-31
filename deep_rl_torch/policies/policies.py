import itertools
import time
import random
import copy

import logging
import gym
import torch
import tqdm
from pynvml import nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex, nvmlInit
from pytorch_memlab import LineProfiler, profile, profile_every, set_target_gpu

# Silent error - but it will be raised in trainer.py, so it is fine. relates to apex
try:
    from apex import amp
except ImportError:
    pass

# Internal Imports:
from deep_rl_torch.experience_buffer import ReplayBuffer, CERWrapper, PERBuffer, RLDataset, PERDataset
from deep_rl_torch.nn import Q, V, Actor, ProcessState, ProcessStateAction
from deep_rl_torch.util import *


class BasePolicy:
    def __init__(self, ground_policy, F_s, F_sa, env, device, log, hyperparameters):
        self.env = env
        self.device = device
        self.log = log
        self.hyperparameters = hyperparameters
        self.ground_policy = ground_policy
        self.name = ""
        self.verbose = hyperparameters["verbose"]

        # Check env:
        self.discrete_env = True if 'Discrete' in str(env.action_space) else False
        if self.discrete_env:
            self.num_actions = self.env.action_space.n
            self.action_low = torch.zeros(self.num_actions, device=self.device)
            self.action_high = torch.ones(self.num_actions, device=self.device)
            if self.verbose:
                print("Num actions: ", self.num_actions)
        else:
            self.num_actions = len(self.env.action_space.high)
            self.action_low = torch.tensor(env.action_space.low, device=self.device)
            self.action_high = torch.tensor(env.action_space.high, device=self.device)
            if self.verbose:
                print("Env action low: ", self.action_low)
                print("Env action high: ", self.action_high)

        # Set up parameters:
        # Actor-Critic:
        self.use_actor_critic = hyperparameters["use_actor_critic"]
        self.use_CACLA_V = hyperparameters["use_CACLA_V"]
        self.use_CACLA_Q = hyperparameters["use_CACLA_Q"]
        self.use_DDPG = hyperparameters["use_DDPG"]
        self.use_SPG = hyperparameters["use_SPG"]
        self.use_GISPG = hyperparameters["use_GISPG"]
        # QV:
        self.use_QV = hyperparameters["use_QV"]
        self.use_QVMAX = hyperparameters["use_QVMAX"]
        # Exploration:
        self.gaussian_action_noise = hyperparameters["action_sigma"]
        self.boltzmann_exploration_temp = hyperparameters["boltzmann_temp"]
        self.epsilon = hyperparameters["epsilon"]
        self.epsilon_mid = hyperparameters["epsilon_mid"]
        if self.epsilon_mid:
            self.eps_factor = self.epsilon_mid ** (1 / hyperparameters["steps"])
            self.epsilon = 1
        # General:
        self.use_half = hyperparameters["use_half"]
        self.batch_size = hyperparameters["batch_size"]
        self.use_world_model = hyperparameters["use_world_model"]

        # TODO: -Include PER with prioritization based on Upper Bound of Gradient Norm.
        # TODO: -include different sampling schemes from the papers investigatin PER in SL (small and big buffer for gradient norm too)

        # TODO: -add goal to replay buffer and Transition (For HRL)
        # Eligibility traces:
        if torch.cuda.is_available():
            nvmlInit()
            self.nvml_handle = nvmlDeviceGetHandleByIndex(0)
            self.max_gpu_bytes = torch.cuda.get_device_properties(self.device).total_memory
        self.mem_usage = None
        self.current_episode = []
        self.use_efficient_traces = hyperparameters["use_efficient_traces"]
        self.elig_traces_update_steps = hyperparameters["elig_traces_update_steps"]
        self.elig_traces_anneal_lambda = hyperparameters["elig_traces_anneal_lambda"]
        self.lambda_val = hyperparameters["elig_traces_lambda"]
        # Set up replay buffer:
        self.stack_dim = hyperparameters["stack_dim"]
        self.stack_count = hyperparameters["frame_stack"]
        self.buffer_size = hyperparameters["replay_buffer_size"] + hyperparameters["num_expert_samples"]
        self.use_PER = hyperparameters["use_PER"]
        self.use_CER = hyperparameters["use_CER"]
        self.PER_alpha = hyperparameters["PER_alpha"]
        self.PER_start_beta = hyperparameters["PER_beta"]
        self.PER_beta = self.PER_start_beta
        self.PER_anneal_beta = hyperparameters["PER_anneal_beta"]
        self.PER_max_priority = hyperparameters["PER_max_priority"]
        self.PER_running_avg = hyperparameters["PER_running_avg"]
        self.importance_weights = None

        # Create replay buffer:
        self.memory = self.create_replay_buffer()

        # Feature extractors:
        self.F_s = F_s
        self.F_sa = F_sa
        self.state_feature_len = F_s.layers_merge[-1].out_features
        if F_sa is not None:
            self.state_action_feature_len = F_sa.layers_merge[-1].out_features

        # Set up Networks:
        self.use_half = hyperparameters["use_half"] and torch.cuda.is_available()
        self.nets = []
        self.actor, self.Q, self.V = self.init_actor_critic(self.F_s, self.F_sa)

    def create_replay_buffer(self):
        obs_sample = self.env.observation_space.sample()
        # action_sample = self.env.action_space.sample()
        action_space = self.env.action_space
        worker = self.hyperparameters["worker"]
        pin_mem = self.hyperparameters["pin_mem"]
        size_expert_data = 0
        update_freq = self.hyperparameters["buffer_update_steps"] * self.batch_size * bool(worker)

        args = (self.log, self.buffer_size, obs_sample, action_space, size_expert_data, self.stack_dim, self.stack_count,
                update_freq, self.hyperparameters["use_list"])
        bargs = (self.batch_size, pin_mem, worker, self.device)
        if self.use_PER:
            dataset = PERDataset(self.PER_alpha, *args, max_priority=self.PER_max_priority,
                                 running_avg=self.PER_running_avg)
            memory = PERBuffer(dataset, *bargs)
        else:
            dataset = RLDataset(*args)
            memory = ReplayBuffer(dataset, *bargs)
        # Wrap buffer to add functionalities:
        if self.use_CER:
            memory = CERWrapper(memory)
        return memory

    def random_action(self):
        action = (self.action_high - self.action_low) * torch.rand(self.num_actions, device=self.device,
                                                                   dtype=torch.float).unsqueeze(0) + self.action_low
        return action

    def boltzmann_exploration(self, action):
        pass
        # TODO: implement boltzman exploration

    def explore_discrete_actions(self, action):
        if self.boltzmann_exploration_temp > 0:
            action = self.boltzmann_exploration(action)
        else:
            action = torch.argmax(action).item()
        return action

    def add_noise(self, action):
        if self.gaussian_action_noise:
            action += torch.tensor(np.random.normal(0, self.gaussian_action_noise, len(action)), dtype=torch.float)
            action = np.clip(action, self.action_low, self.action_high)
        return action

    def choose_action(self, state, calc_state_features=True):
        state = self.state2device(state)
        with torch.no_grad():
            if calc_state_features:
                state_features = self.F_s(state)
            else:
                state_features = state
            action = self.actor(state_features)
        return action

    def explore(self, state, fully_random=False):
        # Epsilon-Greedy:
        sample = random.random()
        if fully_random or sample < self.epsilon:
            raw_action = self.random_action()
        else:
            # Raw q-vals for actions or action (actor critic):
            raw_action = self.choose_action(state)
            # Log Q-val:
            if self.log:
                if self.use_actor_critic:
                    pass
                else:
                    recent_q_val = torch.max(raw_action)
                    if self.log.is_available("Diagnostics/Q-val", factor=10, reset=False):
                        self.log.add("Diagnostics/Q-val", recent_q_val, make_distr=True,
                                     distr_steps=self.log.mean_ep_len * 5)
            # Add Gaussian noise:
            if self.gaussian_action_noise:
                raw_action = self.add_noise(raw_action)

        # If env is discrete explore accordingly and set action
        if self.discrete_env:
            action = self.explore_discrete_actions(raw_action)
        else:
            action = raw_action[0].numpy()
        return action, raw_action

    def act(self, state):
        return self.explore(state)

    def state2device(self, state):
        if isinstance(state, dict):
            state = apply_rec_to_dict(lambda x: x.to(self.device).float(), state)
        else:
            state = state.to(self.device).float()
        return state

    def exploit(self, state):
        raw_action = self.choose_action(state)
        if self.discrete_env:
            action = torch.argmax(raw_action).item()
        else:
            action = raw_action
        return action, raw_action

    def init_actor_critic(self, F_s, F_sa):
        Q_net, V_net = self.init_critic(F_s, F_sa)
        actor = self.init_actor(Q_net, V_net, F_s)
        # print(Q_net.actor)
        return actor, Q_net, V_net

    def init_critic(self, F_s, F_sa):
        if self.use_actor_critic:
            self.state_action_feature_len = F_sa.layers_merge[-1].out_features
            input_size = self.state_action_feature_len
        else:
            self.state_feature_len = F_s.layers_merge[-1].out_features
            input_size = self.state_feature_len

        Q_net = None
        if not (self.use_CACLA_V and not self.use_QVMAX):
            Q_net = Q(input_size, self.env, F_s, F_sa, self.device, self.log, self.hyperparameters)
            if self.use_half:
                Q_net = amp.initialize(Q_net, verbosity=0)
            self.nets.append(Q_net)

        V_net = None
        if self.use_QV or self.use_QVMAX or (self.use_actor_critic and self.use_CACLA_V):
            # Init Networks:
            V_net = V(self.state_feature_len, self.env, F_s, None, self.device, self.log, self.hyperparameters)
            if self.use_half:
                V_net = amp.initialize(V_net, verbosity=0)
            self.nets.append(V_net)
        return Q_net, V_net

    def train_critic(self, Q, V, transitions):
        TDE_V = 0
        loss = 0
        if self.V is not None:
            self.V.retain_graph = True
            TDE_V, loss_V = self.V.optimize(transitions, actor=self.actor, Q=Q, V=None, policy_name=self.name)
            loss += loss_V

        # Only if we use standard CACLA (and we do not train the V net using QVMAX) we do not need a Q net:
        TDE_Q = 0
        if self.Q is not None:
            TDE_Q, loss_Q = self.Q.optimize(transitions, actor=self.actor, Q=None, V=V, policy_name=self.name)
            loss += loss_Q

        TDE = (TDE_Q + TDE_V) / ((self.V is not None) + (self.Q is not None))

        TDE_abs = abs(TDE)
        return TDE_abs, loss

    def extract_features(self, transitions):
        """ Calculates the state_features and state_action features and stores them in the transitions dict """
        # Extract state (and state-action) features:
        state_batch = transitions["states"]
        non_final_next_states = transitions["non_final_next_states"]
        state_feature_batch = self.F_s(state_batch)
        if non_final_next_states is not None:
            non_final_next_state_features = self.F_s.forward_next_state(non_final_next_states)
        else:
            non_final_next_state_features = None
        transitions["state_features"] = state_feature_batch
        transitions["non_final_next_state_features"] = non_final_next_state_features

    def optimize(self):
        before_sampling = time.time()
        # Get Batch:
        transitions = self.get_transitions()
        self.log.add("Timings/Sampling_Time", time.time() - before_sampling, use_skip=True, store_episodic=True)
        # Extract state features
        self.extract_features(transitions)
        # Optimize:
        if self.use_world_model:
            self.world_model.optimize()
            # TODO: create a world model at some point
        error, loss = self.optimize_networks(transitions)

        error = abs(error) + 0.0001

        if self.use_PER:
            self.memory.update_priorities(transitions["idxs"], error)

        self.display_debug_info()

        return loss

    def update_parameters(self, n_steps, train_fraction):
        if self.epsilon_mid:
            self.epsilon *= self.eps_factor
            self.log.add("Params/Epsilon", self.epsilon)
        if self.use_PER and self.PER_anneal_beta:
            self.PER_beta = self.PER_start_beta + train_fraction * (1 - self.PER_start_beta)
        # TODO: decay temperature for Boltzmann if that exploration is used (first implement it in general)

    def get_transitions(self):
        """Gets transitions from dataloader, which is a batch of transitions.
         It is a dict of the form
         {"states": Tensor, "actions_argmax": Tensor of Ints, "actions": Tensor of raw action preferences,
          "rewards": Tensor, "non_final_next_states": Tensor, "non_final_mask": Tensor of bools, "Dones": Tensor,
           "Importance_Weights: Tensor, "Idxs": Tensor} """
        if self.use_PER:
            return self.memory.sample(self.PER_beta)
        else:
            trans = self.memory.sample()
            # print(trans)
            return trans

    def optimize_networks(self, transitions):
        raise NotImplementedError

    def scale_gradient(self):
        pass
        # TODO: implement such that all policies can do this... is it necessary??

    def display_debug_info(self):
        pass

    def calculate_Q_and_TDE(self, state, action, next_state, reward, done):
        return self.critic.calculate_Q_and_TDE(state, action, next_state, reward, done, actor=self.actor, Q=self.Q,
                                               V=self.V)

    def set_retain_graph(self, val):
        for net in self.nets:
            net.retain_graph = val


    def remember(self, state, action, reward, done, filling_buffer=False):
        if self.use_efficient_traces:
            transition = (state, action, reward, done)
            self.current_episode.append(transition)
            if done:
                for state, action, reward, done in self.current_episode:
                    self.memory.add(state, action, reward, done, store_episodes=True)
                self.current_episode = []
                # Update episode trace:
                if not filling_buffer:
                    most_recent_episode, idxs = self.memory.get_most_recent_episode()
                    self.update_episode_trace(most_recent_episode, idxs)
        else:
            self.memory.add(state, action, reward, done)

    def update_targets(self, n_steps, train_fraction=None):
        if self.use_efficient_traces:
            self.update_traces(n_steps, train_fraction=train_fraction)

        if self.Q is not None:
            self.Q.update_targets(n_steps)
        if self.V is not None:
            self.V.update_targets(n_steps)
        if self.actor is not None and self.use_actor_critic:
            self.actor.update_targets(n_steps)

    def calc_max_batch_size(self):
        if not torch.cuda.is_available():
            return 1000
        # Calculate remaining gpu mem:
        current_gpu_bytes = nvmlDeviceGetMemoryInfo(self.nvml_handle).used  # torch.cuda.memory_allocated(self.device)

        available_gpu_bytes = self.max_gpu_bytes - current_gpu_bytes
        # Leave some room, at least 4mb:
        available_gpu_bytes -= 4 * 1024 ** 2
        max_batch_size = available_gpu_bytes // self.mem_usage
        # print("Mem per transition: ", self.mem_usage / 1024 ** 2)
        # print("Available mbs: ", available_gpu_bytes / 1024 ** 2)
        # print("Max batch size: ", max_batch_size)
        return max_batch_size

    def split_batch(self, batch, start_idx, end_idx):
        return apply_rec_to_dict(lambda x: x[start_idx:end_idx], batch)

    def update_episode_trace(self, episode, idxs):
        if episode == []:
            return

        # Split batch into chunks to avoid overloading GPU memory:
        if torch.cuda.is_available():
            # ep_len = episode["rewards"].shape[0]
            ep_len = len(episode)
            max_batch_size = self.calc_max_batch_size()
            if ep_len > max_batch_size:
                ep_starts = range(0, ep_len, max_batch_size)
                ep_ends = [min(start_idx + max_batch_size, len(episode)) for start_idx in ep_starts]
                # episode_parts = [self.split_batch(episode, start_idx, end_idx) for start_idx, end_idx in zip(ep_starts, ep_end)]
                episode_parts = [episode[start_idx:end_idx] for start_idx, end_idx in zip(ep_starts, ep_end)]
            else:
                episode_parts = [episode]
        else:
            episode_parts = [episode]
        # Update traces:
        last_trace_value_Q = None
        last_trace_value_V = None
        for episode_part in episode_parts:
            episode_transitions = self.extract_batch(episode_part)
            episode_transitions["idxs"] = idxs
            self.extract_features(episode_transitions)
            if self.V is not None:
                last_trace_value_V = self.V.update_traces(episode_transitions, self.lambda_val, actor=None, V=None,
                                                          Q=self.Q, last_trace_value=last_trace_value_V)
            if self.Q is not None:
                last_trace_value_Q = self.Q.update_traces(episode_transitions, self.lambda_val, actor=self.actor,
                                                          V=self.V, Q=None, last_trace_value=last_trace_value_Q)

        # TODO: when calculating eligibility traces we could also estimate the TD error for PER

    def update_traces(self, n_steps, train_fraction=None):
        # Update lambda val:
        if self.elig_traces_anneal_lambda and train_fraction is not None:
            self.lambda_val = 1.0 * (1 - train_fraction)
        # Update traces if it's time:
        if n_steps % self.elig_traces_update_steps == 0:
            episodes, idx_list = self.memory.get_all_episodes()
            if self.verbose:
                print("Updating traces. Number of episodes to update: ", len(episodes))
            # TODO: instead of updating all episode traces, only update a fraction of them: the oldest ones (or at least do not update the most recent episodes [unless episodes are very long])
            for episode, idxs in tqdm.tqdm(zip(episodes, idx_list), disable=not self.verbose):
                self.update_episode_trace(episode, idxs)

    def freeze_normalizers(self):
        self.F_s.freeze_normalizers()
        if self.F_sa is not None:
            self.F_sa.freeze_normalizers()

    def norm_gradient(self):
        for net in self.nets:
            net.norm_gradient()

    def log_nn_data(self, name=""):
        for net in self.nets:
            net.log_nn_data(name=name)

    def init_actor(self, Q, V, F_s):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def get_updateable_params(self):
        raise NotImplementedError


class ActorCritic(BasePolicy):
    def __repr__(self):
        return "Actor-Critic"

    def __init__(self, ground_policy, F_s, F_sa, env, device, log, hyperparameters):
        super(ActorCritic, self).__init__(ground_policy, F_s, F_sa, env, device, log, hyperparameters)
        self.F_s = F_s

        self.set_name("Actor-Critic")

    def optimize_networks(self, transitions):
        # TODO: possible have an action normalizer? For state_features we could have a batchnorm layer, maybe it is better for both
        # TODO: for HRL this might be nice
        state_features = transitions["state_features"]
        action_batch = transitions["action"]

        state_action_features = self.F_sa(state_features, action_batch)
        transitions["state_action_features"] = state_action_features
        if self.Q is not None:
            self.Q.retain_graph = True
        if self.V is not None:
            self.V.retain_graph = True
        TDE, loss_critic = self.train_critic(self.Q, self.V, transitions)

        error, loss_actor = self.train_actor(transitions)

        loss = loss_critic + loss_actor

        return TDE + error, loss

    def init_actor(self, Q, V, F_s):
        actor = Actor(F_s, self.env, self.log, self.device, self.hyperparameters)
        if self.use_half:
            actor = amp.initialize(actor, verbosity=0)
        actor.Q = Q
        actor.V = V
        self.nets.append(actor)
        # Q.target_net.actor = actor

        # print(actor)
        return actor

    def train_actor(self, transitions):
        error, loss = self.actor.optimize(transitions, self.name)
        return abs(error), loss

    def set_name(self, name):
        self.name = str(name) + "_ActorCritic" if str(name) != "" else "ActorCritic"

    def save(self, path):
        path += self.name + "/"
        if not os.path.exists(path):
            os.mkdir(path)

        if self.Q is not None:
            self.Q.save(path + "Q/")
        if self.V is not None:
            self.V.save(path + "V/")
        self.actor.save(path + "actor/")

    def get_updateable_params(self):
        params = None
        for net in self.nets:
            new_params = list(net.get_updateable_params())
            if params is None:
                params = new_params
            else:
                params += new_params
        return params

        params = list(self.layers_TD.parameters())
        if self.split:
            params += list(self.layers_r.parameters())
        return params


class Q_Policy(BasePolicy):
    def __init__(self, ground_policy, F_s, F_sa, env, device, log, hyperparameters):
        super(Q_Policy, self).__init__(ground_policy, F_s, F_sa, env, device, log, hyperparameters)

        self.critic = self.Q

        self.name = "Q-Policy"

    def optimize_networks(self, transitions, retain_graph=False):
        TDE, loss = self.train_critic(self.Q, self.V, transitions)
        return TDE, loss

    def init_actor(self, Q, V, F_s):
        return Q

    def set_name(self, name):
        self.name = str(name) + "_Q-Policy" if str(name) != "" else "Q-Policy"

    def save(self, path):
        path += self.name + "/"
        if not os.path.exists(path):
            os.mkdir(path)

        if self.Q is not None:
            self.Q.save(path + "Q/")
        if self.V is not None:
            self.V.save(path + "V/")

    def get_updateable_params(self):
        params = []
        for net in self.nets:
            new_params = list(net.get_updateable_params())
            params += new_params
        return params
