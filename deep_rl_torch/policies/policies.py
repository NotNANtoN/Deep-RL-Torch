import itertools
import time
import random
import copy

import logging
import gym
import torch
from pytorch_memlab import LineProfiler, profile, profile_every, set_target_gpu
try:
    from apex import amp
except:
    pass

# Internal Imports:
from deep_rl_torch.experience_buffer import ReplayBuffer, PrioritizedReplayBuffer
from deep_rl_torch.nn import Q, V, Actor, ProcessState, ProcessStateAction
from deep_rl_torch.util import *
# Silent error - but it will be raised in trainer.py, so it is fine



class BasePolicy:
    def __init__(self, ground_policy, F_s, F_sa, env, device, log, hyperparameters):
        self.env = env
        self.device = device
        self.log = log
        self.hyperparameters = hyperparameters
        self.ground_policy = ground_policy
        self.name = ""
        self.log_freq = self.hyperparameters["log_freq"]

        # Check env:
        self.discrete_env = True if 'Discrete' in str(env.action_space) else False
        if self.discrete_env:
            self.num_actions = self.env.action_space.n
            self.action_low = torch.zeros(self.num_actions, device=self.device)
            self.action_high = torch.ones(self.num_actions, device=self.device)
            print("Num actions: ", self.num_actions)
        else:
            self.num_actions = len(self.env.action_space.high)
            self.action_low = torch.tensor(env.action_space.low, device=self.device)
            self.action_high = torch.tensor(env.action_space.high, device=self.device)
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
        self.eps_decay = hyperparameters["epsilon_decay"]
        # General:
        self.use_half = hyperparameters["use_half"]
        self.batch_size = hyperparameters["batch_size"]
        self.use_world_model = hyperparameters["use_world_model"]

        # TODO: -Include PER with prioritization based on Upper Bound of Gradient Norm.
        # TODO: -include different sampling schemes from the papers investigatin PER in SL (small and big buffer for gradient norm too)

        # TODO: -add goal to replay buffer and Transition (For HRL)
        # Eligibility traces:
        self.current_episode = []
        self.use_efficient_traces = hyperparameters["use_efficient_traces"]
        self.elig_traces_update_steps = hyperparameters["elig_traces_update_steps"]
        self.elig_traces_anneal_lambda = hyperparameters["elig_traces_anneal_lambda"]
        self.lambda_val = hyperparameters["elig_traces_lambda"]
        # Set up replay buffer:
        self.buffer_size = hyperparameters["replay_buffer_size"] + hyperparameters["num_expert_samples"]
        self.use_PER = hyperparameters["use_PER"]
        self.use_CER = hyperparameters["use_CER"]
        self.PER_alpha = hyperparameters["PER_alpha"]
        self.PER_start_beta = hyperparameters["PER_beta"]
        self.PER_beta = self.PER_start_beta
        self.anneal_beta = hyperparameters["PER_anneal_beta"]
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
        if self.use_PER:
            memory = PrioritizedReplayBuffer(self.buffer_size, self.PER_alpha, use_CER=self.use_CER)
        else:
            memory = ReplayBuffer(self.buffer_size, use_CER=self.use_CER)
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
        with torch.no_grad():
            if calc_state_features:
                state_features = self.F_s(state)
            else:
                state_features = state
            action = self.actor(state_features)
        return action

    def explore(self, state, fully_random=False):
        state = self.state2device(state)

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
                    self.recent_q_val = torch.max(raw_action)
                self.log.add("Q-val", self.recent_q_val, skip_steps=10)
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
        state = self.state2device(state)

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
            V.retain_graph = True
            TDE_V, loss_V = V.optimize(transitions, importance_weights=transitions["importance_weights"], actor=self.actor,
                               Q=Q, V=None, policy_name=self.name)
            loss += loss_V

        # Only if we use standard CACLA (and we do not train the V net using QVMAX) we do not need a Q net:
        TDE_Q = 0
        if self.Q is not None:
            TDE_Q, loss_Q = Q.optimize(transitions, importance_weights=transitions["importance_weights"], actor=self.actor,
                               Q=None, V=V, policy_name=self.name)
            loss += loss_Q

        TDE = (TDE_Q + TDE_V) / ((self.V is not None) + (self.Q is not None))

        TDE_abs = abs(TDE)
        return TDE_abs, loss

    def extract_features(self, transitions):
        """ Calculates the state_features and state_action features and stores them in the transitions dict """
        # Extract features:
        state_batch = transitions["state"]
        non_final_next_states = transitions["non_final_next_states"]
        #if isinstance(state_batch, dict):
        #    state_batch = apply_rec_to_dict(lambda x: x.to(self.device, non_blocking=True), state_batch)
        #    non_final_next_states = apply_rec_to_dict(lambda x: x.to(self.device), non_final_next_states)
        #else:
        #    state_batch = state_batch.to(self.device, non_blocking=True)
        #    non_final_next_states = non_final_next_states.to(self.device, non_blocking=True)
        state_feature_batch = self.F_s(state_batch)
        if non_final_next_states is not None:
            non_final_next_state_features = self.F_s.forward_next_state(non_final_next_states)
        transitions["state_features"] = state_feature_batch
        transitions["non_final_next_state_features"] = non_final_next_state_features
        #transitions["action"] = transitions["action"].to(self.device, non_blocking=True)
        #transitions["reward"] = transitions["reward"].to(self.device, non_blocking=True)

    def optimize(self):
        before_sampling = time.time()
        # Get Batch:
        transitions = self.get_transitions()
        self.log.add("Sampling_Time", time.time() - before_sampling, skip_steps=self.log_freq, store_episodic=True)
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


    def decay_exploration(self, n_steps):
        if self.eps_decay:
            self.epsilon *= self.eps_decay
            self.log.add("Epsilon", self.epsilon)
        # TODO: decay temperature for Boltzmann if that exploration is used (first implement it in general)

    def collate_batch(self, batch):
        for key in batch:
            pass
    
    def construct_dataloader(self):
        # TODO: change this such that it is all embedded in the exp rep buffer. Ins
        if self.use_PER:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples, replacement=True)
        else:
            sampler = torch.utils.data.sampler.RandomSampler(data, replacement=False, num_samples=None)

        self.dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                                  sampler=sampler,
                                                  pin_memory=self.pin_mem,
                                                  num_workers=self.num_sampling_workers,
                                                  collate_fn=custom_collate)

    def get_transitions_new(self):
        """ Gets transitions from dataloader, which is a batch of transitions. It is a dict of the form {"states": Tensor, "actions_argmax": Tensor of Ints, "actions": Tensor of raw action preferences, "rewards": Tensor, "non_final_next_states": Tensor, "non_final_mask": Tensor of bools, "Dones": Tensor, "Importance_Weights: Tensor, "Idxs": Tensor} """
        transitions= next(self.memory)
        return transitions
        

    def get_transitions(self):
        sampling_size = min(len(self.memory), self.batch_size)
        if self.use_PER:
            transitions, importance_weights, idxs = self.memory.sample(sampling_size, self.PER_beta)
            # print(importance_weights)
            importance_weights = torch.tensor(importance_weights, device=self.device).float()
        else:
            transitions, idxs = self.memory.sample(sampling_size)
            importance_weights = None
        # Transform the stored tuples into torch arrays:
        transitions = self.extract_batch(transitions)
        # Save PER relevant info:
        transitions["idxs"] = idxs
        transitions["importance_weights"] = importance_weights

        return transitions

    def extract_batch(self, transitions):
        # TODO: How fast is this? Can't we put it into arrays?
        # TODO: We sample, then concatenate the sampled parts into multiple arrays, transposing it before...

        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        if self.use_half:
            dtype = torch.half
        else:
            dtype = torch.float

        # Create state batch:
        if isinstance(batch.state[0], dict):
            # Concat the states per key:
            state_batch = {key: torch.cat([x[key] if key == "pov" else x[key] for x in batch.state]).type(dtype).to(self.device, non_blocking=True) for key in batch.state[0]}
        else:
            state_batch = torch.cat(batch.state).type(dtype).to(self.device, non_blocking=True)

        # print("non_final mask: ", non_final_mask)
        # print("state batch: ", state_batch)
        # print("next state batch: ", next_state_batch)
        # print("non final next state batch: ", non_final_next_states)

        # Create next state batch:
        
        non_final_next_states = [s for s in batch.next_state if s is not None]
        #print(non_final_next_states)
        if non_final_next_states:
            if isinstance(non_final_next_states[0], dict):
                non_final_next_states = {key: torch.cat([x[key] if key == "pov" else x[key] for x in non_final_next_states]).type(dtype).to(self.device, non_blocking=True)
                                         for key in non_final_next_states[0]}
            else:
                non_final_next_states = torch.cat(non_final_next_states).type(dtype).to(self.device, non_blocking=True)
        else:
            non_final_next_states = None

        # Create action batch:
        # print(batch.action)
        action_batch = torch.cat(batch.action).type(dtype).to(self.device, non_blocking=True)

        action_argmax = torch.argmax(action_batch, 1).unsqueeze(1)

        # Create Reward batch:
        reward_batch = torch.cat(batch.reward).unsqueeze(1).type(dtype).to(self.device, non_blocking=True)

        transitions = {"state": state_batch, "action": action_batch, "reward": reward_batch,
                       "non_final_next_states": non_final_next_states, "non_final_mask": non_final_mask,
                       "state_action_features": None, "importance_weights": None, "idxs": None,
                       "action_argmax": action_argmax}

        return transitions

    def optimize_networks(self, transitions):
        raise NotImplementedError

    def scale_gradient(self):
        pass
        # TODO important!: implement such that all policies can do this... is it necessary??

    def display_debug_info(self):
        pass

    def calculate_Q_and_TDE(self, state, action, next_state, reward, done):
        return self.critic.calculate_Q_and_TDE(state, action, next_state, reward, done, actor=self.actor, Q=self.Q,
                                               V=self.V)

    def set_retain_graph(self, val):
        if self.Q is not None:
            self.Q.retain_graph = val
        if self.V is not None:
            self.V.retain_graph = val
        if self.actor is not None:
            self.actor.retain_graph = val

    #     def update_traces(self, episode_transitions, actor=None, V=None, Q=None):
    #         self.use_efficient_traces = hyperparameters["use_efficient_traces"]

    def remember(self, state, action, reward, next_state, done):
        if self.use_efficient_traces:
            transition = (state, action, reward, next_state, done)
            self.current_episode.append(transition)
            if done:
                for state, action, reward, next_state, done in self.current_episode:
                    self.memory.add(state, action, reward, next_state, done, store_episodes=True)
                self.current_episode = []
                # Update episode trace:
                most_recent_episode, idxs = self.memory.get_most_recent_episode()
                self.update_episode_trace(most_recent_episode, idxs)

        else:
            self.memory.add(state, action, reward, next_state, done)

    def update_targets(self, n_steps, train_fraction=None):
        if self.use_efficient_traces:
            self.update_traces(n_steps, train_fraction=train_fraction)

        if self.use_PER and self.anneal_beta:
            self.PER_beta = self.PER_start_beta + (1 - self.PER_start_beta) * train_fraction

        if self.Q is not None:
            self.Q.update_targets(n_steps)
        if self.V is not None:
            self.V.update_targets(n_steps)
        if self.actor is not None and self.use_actor_critic:
            self.actor.update_targets(n_steps)

    def get_tensor_size(self, tensor):
        multiplier = None
        if tensor.dtype == torch.float16:
            multiplier = 16
        elif tensor.dtype == torch.int8:
            multiplier = 8
        elif tensor.dtype == torch.float32:
            multiplier = 32
        else:
            raise TypeError("Unknown input type: ", tensor.dtype)
        bit_size = tensor.view(-1).shape[0] * multiplier

        return bit_size

    def get_largest_tensor_size_from_dict(self, tensor_dict):
        biggest = 0
        for key in tensor_dict:
            value = tensor_dict[key]
            if isinstance(value, torch.tensor):
                size = self.get_tensor_size(value)
            elif isinstance(value, dict):
                size = self.get_largest_tensor_size_from_dict(value)
            else:
                raise TypeError("Unkown type " + type(value) + " in dict " + tensor_dict)

            if size > biggest:
                biggest = size

        if biggest == 0:
            raise ValueError("No tensor of a size was found in the observation sample!")
        return biggest

    def update_episode_trace(self, episode, idxs):
        if episode == []:
            return

        # Split batch into chunks to avoid overloading GPU memory:
        if torch.cuda.is_available() and 0 == 1:
            # Get size of largest tensor in observation:
            sample = self.env.observation_space.sample()
            if isinstance(sample, dict):
                size = self.get_largest_tensor_size_from_dict(sample)
            else:
                size = self.get_tensor_size(sample)
            # Calculate remaining gpu mem:
            max_gpu_mem_bits = torch.cuda.get_device_properties(self.device).total_memory
            current_gpu_mem_bits = torch.cuda.memory_allocated(self.device)
            available_gpu_mem_bits = max_gpu_mem_bits - current_gpu_mem_bits
            # TODO: to calculate memory usage for a single batch: calculate usage for 2 batchsize and tor 1. substract the latter from the first

            # TODO: subtract at least 777mb for pytorch init ( or 1668mb in total for 2 displays aded)
            # TODO: see the following lines to do it properly:
            #from pynvml import *
            #nvmlInit()
            #h = nvmlDeviceGetHandleByIndex(0)
            #info = nvmlDeviceGetMemoryInfo(h)
            #print(f'total    : {info.total}')
            #print(f'free     : {info.free}')
            #print(f'used     : {info.used}')
            # TODO: we also need to leave at least 3.5mb free

        episode_transitions = self.extract_batch(episode)
        episode_transitions["idxs"] = idxs
        self.extract_features(episode_transitions)
        if self.V is not None:
            self.V.update_traces(episode_transitions, self.lambda_val, actor=None, V=None, Q=self.Q)
        if self.Q is not None:
            self.Q.update_traces(episode_transitions, self.lambda_val, actor=self.actor, V=self.V, Q=None)

        # TODO: when calculating eligibility traces we could also estimate the TD error for PER

    def update_traces(self, n_steps, train_fraction=None):
        # Update lambda val:
        if self.elig_traces_anneal_lambda and train_fraction is not None:
            self.lambda_val = 1.0 * (1 - train_fraction)
        # Update traces if its time:
        if n_steps % self.elig_traces_update_steps == 0:
            episodes, idx_list = self.memory.get_all_episodes()
            # TODO: instead of updating all episode traces, only update a fraction of them: the oldest ones (or at least do not update the most recent episodes [unless episodes are very long])
            for episode, idxs in zip(episodes, idx_list):
                self.update_episode_trace(episode, idxs)

    def freeze_normalizers(self):
        self.F_s.freeze_normalizers()
        if self.F_sa is not None:
            self.F_sa.freeze_normalizers()


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


