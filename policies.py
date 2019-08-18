import random
import torch
import numpy as np

from util import *
from networks import Q, V, Actor, ProcessState, ProcessStateAction
from exp_rep import ReplayBuffer, PrioritizedReplayBuffer

# This is the interface for the agent being trained by a Trainer instance
class AgentInterface:
    def __init__(self):
        pass

    def optimize(self):
        raise NotImplementedError

    def exploit(self, state):
        raise NotImplementedError

    def explore(self, state):
        raise NotImplementedError

    def update_targets(self, n_steps):
        raise NotImplementedError

    def decay_exploration(self, n_steps):
        raise NotImplementedError

    def calculate_TDE(self, state, action, reward, next_state):
        raise NotImplementedError

    def remember(self, state, action, next_state, reward, done):
        raise NotImplementedError

    def display_debug_info(self):
        raise NotImplementedError


class Agent(AgentInterface):
    def __init__(self, env, device, normalizer, log, hyperparameters):
        self.discrete_env = True if "Discrete" in str(env.action_space)[:8] else False
        print(env.action_space)
        print("Env with discrete action space: ", self.discrete_env)
        self.env = env
        self.device = device
        self.normalizer = normalizer
        self.log = log
        self.parameters = hyperparameters

        self.policy = self.create_policy()

    def create_policy(self):
        # Define base policy of HRL or the general algorithm:
        # TODO: add decicion whether to create REM as base_policy
        if self.parameters["use_actor_critic"]:
            base_policy = ActorCritic
        elif self.discrete_env:
            base_policy = Q_Policy
        else:
            raise NotImplementedError("The environment must be discrete to apply Q-Learning, no other"
                                      " framework than Actor-Critic available")
        # Decide whether to use Hierarchical Reinforcement Learning:
        if self.parameters["use_hrl"]:
            policy = HierarchicalPolicy(base_policy, self.log, self.parameters)
        else:
            policy = base_policy(self.env, self.device, self.log, self.parameters, self.normalizer)
        return policy

    def remember(self, state, action, next_state, reward, done):
        self.policy.remember(state, action, next_state, reward, done)

    def optimize(self):
        self.policy.optimize()

    def explore(self, state):
        return self.policy.explore(state)

    def exploit(self, state):
        return self.policy.exploit(state)

    def decay_exploration(self, n_steps):
        self.policy.decay_exploration(n_steps)

    def calculate_TDE(self, state, action, next_state, reward, done):
        return self.policy.calculate_TDE(state, action, next_state, reward, done)

    def update_targets(self, n_steps):
        self.policy.update_targets(n_steps)

    def display_debug_info(self):
        self.policy.display_debug_info()


class BasePolicy:
    def __init__(self, env, device, log, hyperparameters, normalizer):
        self.env = env
        self.device = device
        self.log = log
        self.hyperparameters = hyperparameters
        self.normalizer = normalizer

        # Check env:
        self.discrete_env = True if 'Discrete' in str(env.action_space) else False
        if self.discrete_env:
            self.num_actions = self.env.action_space.n
        else:
            self.num_actions = len(self.env.action_space.high)

        # Set up parameters:
        self.use_actor_critic = self.hyperparameters["use_actor_critic"]
        self.use_QV = self.hyperparameters["use_QV"]
        self.use_QVMAX = self.hyperparameters["use_QVMAX"]
        self.use_world_model = self.hyperparameters["use_world_model"]
        self.gaussian_action_noise = self.hyperparameters["action_sigma"]
        self.boltzmann_exploration_temp = self.hyperparameters["boltzmann_temp"]
        self.epsilon = hyperparameters["epsilon"]
        self.eps_decay = hyperparameters["epsilon_decay"]
        self.batch_size = hyperparameters["batch_size"]

        # TODO: -Include Prioritized Experience Replay (PER): Prioritize based on absolute TDE
        # TODO: -Include PER with prioritization based on Upper Bound of Gradient Norm.
        # TODO: -include different sampling schemes from the papers investigatin PER in SL (small and big buffer for gradient norm too)

        # TODO: -add goal to replay buffer and Transition
        # TODO: -add eligibility traces to replay buffer (probably the one that update after the current episode is done and after k steps)
        # Set up replay buffer:
        self.buffer_size = hyperparameters["replay_buffer_size"]
        self.use_PER = hyperparameters["use_PER"]
        self.use_CER = hyperparameters["use_CER"]
        self.PER_alpha = hyperparameters["PER_alpha"]
        self.PER_beta = hyperparameters["PER_beta"]
        self.importance_weights = None
        # TODO: implement the option to linearly increase PER_beta to 1 over training time
        if self.use_PER:
            self.memory = PrioritizedReplayBuffer(self.buffer_size, self.PER_alpha, use_CER=self.use_CER)
        else:
            self.memory = ReplayBuffer(self.buffer_size, use_CER=self.use_CER)

        # Set up Networks:
        # TODO: move F_s and F_sa creation possibly into Agent to share among all policies
        self.F_s, self.F_sa = self.init_feature_extractors()
        self.actor, self.critic = self.init_actor_critic(self.F_s, self.F_sa)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)


    def random_action(self):
        if self.discrete_env:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long).item()
        else:
            return (self.action_high - self.action_low) * torch.rand(self.num_actions, device=self.device,
                                                                     dtype=torch.float) + self.action_low

    def boltzmann_exploration(self, action):
        pass
    # TODO: implement

    def explore_discrete_actions(self, action):
        if self.boltzmann_exploration_temp > 0:
            action = self.boltzmann_exploration(action)
        else:
            action = torch.argmax(action).item()
        return action

    def add_noise(self, action):
        if self.action_gaussian_noise_std:
            action += torch.tensor(np.random.normal(0, self.action_gaussian_noise_std, len(action)), dtype=torch.float)
            action = torch.tensor(np.clip(action, self.action_low, self.action_high))
        return action

    def state2parts(self, state):
        # TODO: adjust this function to be able to deal with vector envs, rgb envs, and mineRL
        return state, None

    def explore(self, state):
        # Epsilon-Greedy:
        if self.epsilon:
            sample = random.random()
            if sample < self.epsilon:
                return self.random_action()
        # Raw action:
        vectors, matrix = self.state2parts(state)
        state_features = self.F_s(vectors, matrix)
        action = self.actor(state_features).detach()
        # Add Gaussian noise:
        if self.gaussian_action_noise:
            action = self.add_noise(action)
        # If env is discrete explore accordingly and set action
        if self.discrete_env:
            action = self.explore_discrete_actions(action)
        return action

    def act(self, state):
        return self.explore(state)

    def exploit(self, state):
        with torch.no_grad():
            state_features = self.F_s(state, None)
            action = self.actor(state_features)
        if self.discrete_env:
            action = torch.argmax(action).item()
        return action

    def init_feature_extractors(self):
        # TODO: this currently only works for input vectors, NOT for matrices or multiple input vectors (not for MineRL)
        input_vector_len = len(self.env.observation_space.high)
        matrix_shape = None

        F_s = ProcessState(input_vector_len, matrix_shape, self.log, self.device, self.hyperparameters)
        self.state_feature_len = F_s.layers_merge[-1].out_features

        F_sa = None
        if self.use_actor_critic:
            F_sa = ProcessStateAction(state_feature_len, self.num_actions, self.device, self.log, self.hyperparameters)
            self.state_action_feature_len = F_sa.layers[-1].out_features
        return F_s, F_sa

    def init_actor_critic(self, F_s, F_sa):
        critic = self.init_critic(F_s, F_sa)
        actor = self.init_actor(critic, F_s)
        return actor, critic

    def init_critic(self, F_s, F_sa):
        # TODO: differentiate between DQN critic and AC critic by checking in Q __init__ for use_actor_critic
        if self.use_actor_critic:
            input_len = self.state_action_feature_len
        else:
            input_len = self.state_feature_len
        self.Q = Q(input_len, self.num_actions, self.discrete_env, F_s, F_sa, self.device, self.log, self.hyperparameters)

        if self.use_QV or self.use_QVMAX:
            # Init Networks:
            self.V = V(input_len, F_s, self.device, self.log, self.hyperparameters)
            # Create links for training value calculations:
            self.Q.V = self.V
            self.V.Q = self.Q
        return self.Q

    def train_critic(self, state_feature_batch, state_action_features, action_batch, reward_batch, non_final_next_state_features,
                     non_final_mask):
        if self.use_QV or self.use_QVMAX:
            TDE_V = self.V.optimize(state_feature_batch, reward_batch, non_final_next_state_features, non_final_mask,
                                    self.importance_weights)
        else:
            TDE_V = 0
        TDE_Q = self.Q.optimize(state_feature_batch, state_action_features, action_batch, reward_batch,
                        non_final_next_state_features, non_final_mask, self.importance_weights)
        TDE = (abs(TDE_Q) + abs(TDE_V)) / 2 + 0.0001
        if self.use_PER:
            self.memory.update_priorities(self.PER_idxs, TDE)

    def update_targets(self, n_steps):
        raise NotImplementedError


    def optimize(self):
        # Get Batch:
        transitions = self.get_transitions()
        state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask = self.extract_batch(transitions)
        # Extract features:
        # TODO: we really need to convert a state_batch into vector/matrix stuff instead of hardcoding
        state_feature_batch = self.F_s(state_batch, None)
        non_final_next_state_features = self.F_s(non_final_next_states, None)
        # Optimize:
        if self.use_world_model:
            self.world_model.optimize()
            # TODO: create a world model at some point
        self.optimize_networks(state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                               non_final_mask)

    def decay_exploration(self, n_steps):
        if self.eps_decay:
            self.epsilon *= self.eps_decay
        self.log.add("Epsilon", self.epsilon)
        # TODO: decay temperature for Boltzmann if that exploration is used

    def calculate_TDE(self, state, action, next_state, reward, done):
        return self.critic.calculate_TDE(state, action, next_state, reward, done)

    def get_transitions(self):
        sampling_size = min(len(self.memory), self.batch_size)
        if self.use_PER:
            transitions, self.importance_weights, self.PER_idxs = self.memory.sample(sampling_size, self.PER_beta)
            self.importance_weights = torch.from_numpy(self.importance_weights).float()
            return transitions
        else:
            return self.memory.sample(sampling_size)

    def extract_batch(self, transitions):
        # TODO: How fast is this? Can't we put it into arrays?
        # TODO: We sample, then concatenate the sampled parts into multiple arrays, transposing it before...

        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = self.normalizer.normalize(torch.cat([s for s in batch.next_state
                                                                     if s is not None]))
        state_batch = self.normalizer.normalize(torch.cat(batch.state))
        action_batch = torch.cat(batch.action)
        if self.discrete_env:
            action_batch = action_batch.long().unsqueeze(1)

        reward_batch = torch.cat(batch.reward)

        return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask

    def optimize_networks(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                               non_final_mask):
        raise NotImplementedError

    def display_debug_info(self):
        pass
        # TODO: we could display something nice in here


class ActorCritic(BasePolicy):
    def __init__(self, env, device, log, hyperparameters, normalizer):
        self.discrete_env = True if "Discrete" in str(env.action_space)[:8] else False
        if self.discrete_env:
            self.num_actions = env.action_space.n
        else:
            self.num_actions = len(env.action_space.low)
            self.action_low = torch.tensor(env.action_space.high)
            self.action_high = torch.tensor(env.action_space.low)
        super(ActorCritic, self).__init__(env, device, log, hyperparameters, normalizer)

    def optimize_networks(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                          non_final_mask):
        # TODO: possible have an action normalizer? For state_features we could have a batchnorm layer, maybe it is better for both
        # TODO: for HRL this might be nice
        state_action_features = self.F_sa(state_feature_batch, action_batch)
        self.train_critic(state_feature_batch, state_action_features, action_batch, reward_batch, non_final_next_state_features,
                          non_final_mask)
        self.train_actor(state_feature_batch, action_batch, reward_batch, non_final_next_state_features, non_final_mask)

    def init_actor(self, critic, F_s):
        self.actor = Actor(F_s, env, device, log, hyperparameters)
        self.actor.Q = critic

    def train_actor(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features, non_final_mask):
        self.actor.optimize(state_feature_batch, action_batch, reward_batch, non_final_next_state_features, non_final_mask)

    def update_targets(self, n_steps):
        self.critic.update_targets(n_steps)
        self.actor.update_targets(n_steps)


class Q_Policy(BasePolicy):
    def __init__(self, env, device, log, hyperparameters, normalizer):
        super(Q_Policy, self).__init__(env, device, log, hyperparameters, normalizer)

    def optimize_networks(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                          non_final_mask):
        self.train_critic(state_feature_batch, None, action_batch, reward_batch, non_final_next_state_features,
                             non_final_mask)

    def init_actor(self, critic, F_s):
        return critic

    def update_targets(self, n_steps):
        self.critic.update_targets(n_steps)


class REM(BasePolicy):
    def __init__(self, base_policy):
        # 1. For ensemble: simply create many base policies, train them all with .optimize() and sum action output.
        #    When creating policies, have them share seam feature processor (possibly create one in this class and pass it to base policies)
        # 2. For REM: pick random subset to train
        # 3. For new idea with Matthias: have additional SOM
        pass
