import random
import torch
import numpy as np

from util import *
from networks import Q, V, Actor, ProcessState, ProcessStateAction, create_ff_layers
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
        self.hyperparameters = hyperparameters

        self.use_actor_critic = hyperparameters["use_actor_critic"]

        self.F_s, self.F_sa = self.init_feature_extractors()
        self.policy = self.create_policy()

    def init_feature_extractors(self):
        # TODO: this currently only works for input vectors, NOT for matrices or multiple input vectors (not for MineRL)
        input_vector_len = len(self.env.observation_space.high)
        matrix_shape = None

        F_s = ProcessState(input_vector_len, matrix_shape, self.log, self.device, self.hyperparameters)

        F_sa = None
        if self.use_actor_critic:
            F_sa = ProcessStateAction(state_feature_len, self.num_actions, self.device, self.log, self.hyperparameters)
        return F_s, F_sa

    def create_policy(self):
        # Define base policy of HRL or the general algorithm and ground policy of REM:
        # TODO: add decicion whether to create REM as base_policy
        if self.hyperparameters["use_actor_critic"]:
            base_policy = ActorCritic
        elif self.discrete_env:
            base_policy = Q_Policy
        else:
            raise NotImplementedError("The environment must be discrete to apply Q-Learning, no other"
                                      " framework than Actor-Critic available")
        if self.hyperparameters["use_REM"]:
            # The REM base policy creates an ensemble of ground_policies
            ground_policy = base_policy
            base_policy = REM
        else:
            ground_policy = None
        # Decide whether to use Hierarchical Reinforcement Learning:
        if self.hyperparameters["use_hrl"]:
            policy = HierarchicalPolicy(base_policy, ground_policy, self.log, self.hyperparameters)
        else:
            print("Base Policy (will act concretely): ", base_policy)
            print("Ground Policy (will use base policy): ", ground_policy)
            policy = base_policy(ground_policy, self.F_s, self.F_sa, self.env, self.device, self.log, self.hyperparameters, self.normalizer)
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
        self.policy.F_s.update_targets(n_steps)
        if self.policy.F_sa is not None:
            self.policy.F_sa.update_targets(n_steps)
        self.policy.update_targets(n_steps)

    def display_debug_info(self):
        self.policy.display_debug_info()


class BasePolicy:
    def __init__(self, ground_policy, F_s, F_sa, env, device, log, hyperparameters, normalizer):
        self.env = env
        self.device = device
        self.log = log
        self.hyperparameters = hyperparameters
        self.normalizer = normalizer
        self.ground_policy = ground_policy

        # Check env:
        self.discrete_env = True if 'Discrete' in str(env.action_space) else False
        if self.discrete_env:
            self.num_actions = self.env.action_space.n
        else:
            self.num_actions = len(self.env.action_space.high)

        # Set up parameters:
        self.use_actor_critic = hyperparameters["use_actor_critic"]
        self.use_QV = hyperparameters["use_QV"]
        self.use_QVMAX = hyperparameters["use_QVMAX"]
        self.use_world_model = hyperparameters["use_world_model"]
        self.gaussian_action_noise = hyperparameters["action_sigma"]
        self.boltzmann_exploration_temp = hyperparameters["boltzmann_temp"]
        self.epsilon = hyperparameters["epsilon"]
        self.eps_decay = hyperparameters["epsilon_decay"]
        self.batch_size = hyperparameters["batch_size"]
        self.normalize_observations = hyperparameters["normalize_obs"]

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

        # Feature extractors:
        self.F_s = F_s
        self.F_sa = F_sa
        self.state_feature_len = F_s.layers_merge[-1].out_features
        if F_sa is not None:
            self.state_action_feature_len = F_sa.layers[-1].out_features

        # Set up Networks:
        self.actor, self.Q, self.V = self.init_actor_critic(self.F_s, self.F_sa)

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
        # TODO: also move it into F_s and let F_s just take the state as input
        return state, None

    def choose_action(self, state):
        with torch.no_grad():
            vectors, matrix = self.state2parts(state)
            state_features = self.F_s(vectors, matrix)
            action = self.actor(state_features)
        return action

    def explore(self, state):
        # Epsilon-Greedy:
        if self.epsilon:
            sample = random.random()
            if sample < self.epsilon:
                return self.random_action()
        # Raw action:
        action = self.choose_action(state)
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
        action = self.choose_action(state)
        if self.discrete_env:
            action = torch.argmax(action).item()
        return action

    def init_actor_critic(self, F_s, F_sa):
        Q_net, V_net = self.init_critic(F_s, F_sa)
        actor = self.init_actor(Q_net, V_net, F_s)
        return actor, Q_net, V_net

    def init_critic(self, F_s, F_sa):
        # TODO: differentiate between DQN critic and AC critic by checking in Q __init__ for use_actor_critic
        if self.use_actor_critic:
            input_len = self.state_action_feature_len
        else:
            input_len = self.state_feature_len
        Q_net = Q(input_len, self.num_actions, self.discrete_env, F_s, F_sa, self.device, self.log, self.hyperparameters)

        V_net = None
        if self.use_QV or self.use_QVMAX:
            # Init Networks:
            V_net = V(input_len, F_s, self.device, self.log, self.hyperparameters)
            # Create links for training value calculations:
            Q_net.V = V_net
            V_net.Q = Q_net
        return Q_net, V_net

    def train_critic(self, Q, V, state_feature_batch, state_action_features, action_batch, reward_batch, non_final_next_state_features,
                     non_final_mask, retain_graph=False):
        if self.use_QV or self.use_QVMAX:
            TDE_V = V.optimize(state_feature_batch, reward_batch, non_final_next_state_features, non_final_mask,
                                    self.importance_weights)
        else:
            TDE_V = 0
        TDE_Q = Q.optimize(state_feature_batch, state_action_features, action_batch, reward_batch,
                        non_final_next_state_features, non_final_mask, self.importance_weights)
        TDE = (abs(TDE_Q) + abs(TDE_V)) / ((self.use_QV or self.use_QVMAX) + 1) + 0.0000001
        if self.use_PER:
            self.memory.update_priorities(self.PER_idxs, TDE)


    def optimize(self):
        # Get Batch:
        transitions = self.get_transitions()
        state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask = self.extract_batch(transitions)
        # Extract features:
        # TODO: we really need to convert a state_batch into vector/matrix stuff instead of hardcoding
        state_feature_batch = self.F_s(state_batch, None)
        non_final_next_state_features = self.F_s.forward_next_state(non_final_next_states, None)
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

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        if self.normalize_observations:
            non_final_next_states = self.normalizer.normalize(non_final_next_states)
            state_batch = self.normalizer.normalize(state_batch)
        action_batch = torch.cat(batch.action)
        if self.discrete_env:
            action_batch = action_batch.long().unsqueeze(1)

        reward_batch = torch.cat(batch.reward)

        return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask

    def optimize_networks(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                               non_final_mask):
        raise NotImplementedError

    def calc_norm(self, layers):
        total_norm = torch.tensor(0.)
        for param in layers.parameters():
            total_norm += torch.norm(param)
        return total_norm

    def calculate_TDE(self, state, action, next_state, reward, done):
        return self.critic.calculate_TDE(state, action, next_state, reward, done)

    def set_retain_graph(self, val):
        if self.Q is not None:
            self.Q.retain_graph = val
        if self.V is not None:
            self.V.retain_graph = val
        if self.actor is not None:
            self.actor.retain_graph = val

    def display_debug_info(self):
        raise NotImplementedError

    def update_targets(self, n_steps):
        raise NotImplementedError

    def init_actor(self, Q, V, F_s):
        raise NotImplementedError


class ActorCritic(BasePolicy):
    def __init__(self, ground_policy, F_s, F_sa, env, device, log, hyperparameters, normalizer):
        super(ActorCritic, self).__init__(ground_policy, F_s, F_sa, env, device, log, hyperparameters, normalizer)
        self.discrete_env = True if "Discrete" in str(env.action_space)[:8] else False
        if self.discrete_env:
            self.num_actions = env.action_space.n
        else:
            self.num_actions = len(env.action_space.low)
            self.action_low = torch.tensor(env.action_space.high)
            self.action_high = torch.tensor(env.action_space.low)

    def optimize_networks(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                          non_final_mask):
        # TODO: possible have an action normalizer? For state_features we could have a batchnorm layer, maybe it is better for both
        # TODO: for HRL this might be nice
        state_action_features = self.F_sa(state_feature_batch, action_batch)
        self.train_critic(self.Q, self.V, state_feature_batch, state_action_features, action_batch, reward_batch, non_final_next_state_features,
                          non_final_mask)
        self.train_actor(state_feature_batch, action_batch, reward_batch, non_final_next_state_features, non_final_mask)

    def init_actor(self, Q, V, F_s):
        actor = Actor(F_s, env, device, log, hyperparameters)
        actor.Q = Q
        actor.V = V
        Q.actor = actor
        return actor

    def train_actor(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features, non_final_mask):
        self.actor.optimize(state_feature_batch, action_batch, reward_batch, non_final_next_state_features, non_final_mask)

    def update_targets(self, n_steps):
        self.critic.update_targets(n_steps)
        self.actor.update_targets(n_steps)





class Q_Policy(BasePolicy):
    def __init__(self, ground_policy, F_s, F_sa, env, device, log, hyperparameters, normalizer):
        super(Q_Policy, self).__init__(ground_policy, F_s, F_sa, env, device, log, hyperparameters, normalizer)
        self.critic = self.Q


    def optimize_networks(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                          non_final_mask, retain_graph=False):
        self.train_critic(self.Q, None, state_feature_batch, None, action_batch, reward_batch, non_final_next_state_features,
                             non_final_mask)

    def init_actor(self, Q, V, F_s):
        return Q

    def update_targets(self, n_steps):
        self.critic.update_targets(n_steps)

    def calc_gradient_norm(self, layers):
        total_norm = 0
        for p in layers.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)

    def display_debug_info(self):
        if self.log.do_logging:
            return
        #self.critic.display_debug_info()
        # Weights:
        feature_extractor_weight_norm = self.calc_norm(self.F_s)
        Q_weight_norm = self.calc_norm(self.Q.layers_TD)
        if self.Q.split:
            r_weight_norm = self.calc_norm(self.Q.layers_r)
            self.log.add("r Weight Norm", Q_weight_norm)
        self.log.add("F_s Weight Norm", feature_extractor_weight_norm)
        self.log.add("Q Weight Norm", Q_weight_norm)

        # Gradients:
        F_s_grad_norm = self.calc_gradient_norm(self.F_s.layers_vector)
        F_s_grad_norm = self.calc_gradient_norm(self.F_s.layers_merge)
        #F_s_grad_norm = self.calc_gradient_norm(self.F_s.layers_vector)
        Q_grad_norm = self.calc_gradient_norm(self.Q.layers_TD)
        self.log.add("F_s Vector Gradient Norm", F_s_grad_norm)
        self.log.add("F_s Merge Gradient Norm", F_s_grad_norm)
        self.log.add("Q TD Gradient Norm", Q_grad_norm)

# 1. For ensemble: simply create many base policies, train them all with .optimize() and sum action output.
#    When creating policies, have them share seam feature processor (possibly create one in this class and pass it to base policies)
# 2. For REM: pick random subset to train
# 3. For new idea with Matthias: have additional SOM
class REM(BasePolicy):
    def __init__(self, ground_policy, F_s, F_sa, env, device, log, hyperparameters, normalizer):
        super(REM, self).__init__(ground_policy, F_s, F_sa, env, device, log, hyperparameters, normalizer)

        self.num_heads = hyperparameters["REM_num_heads"]
        self.num_samples = hyperparameters["REM_num_samples"]

        # Create ensemble of ground policies:
        self.policy_heads = [ground_policy(None, F_s, F_sa, env, device, log, hyperparameters, normalizer)
                             for _ in range(self.num_heads)]

    def init_actor(self, Q, V, F_s):
        return None

    def init_critic(self, F_s, F_sa):
        return None, None

    # TODO: atm we iterate through the list of sampled base_policies - can this be done in a better way? MPI, GPU like?
    def optimize_networks(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                               non_final_mask):
        idxes = random.sample(range(self.num_heads), self.num_samples)
        for idx in idxes:
            current_policy = self.policy_heads[idx]
            is_last =  (idx == idxes[-1])
            current_policy.set_retain_graph(not is_last)
            # TODO: here we still need to bootstrap over our batch to have slightly different training data per policy
            current_policy.optimize_networks(state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                               non_final_mask)

    def choose_action(self, state):
        # Preprocess:
        vectors, matrix = self.state2parts(state)
        state_features = self.F_s(vectors, matrix)
        # Select random subset to output action:
        idxes = random.sample(range(self.num_heads), self.num_samples)
        summed_action = None
        for idx in idxes:
            current_policy = self.policy_heads[idx]
            action = current_policy.actor(state_features).detach()
            if summed_action is None:
                summed_action = action
            else:
                summed_action += action
        return summed_action / self.num_samples

    def update_targets(self, n_steps):
        # TODO: test whether sampling here could also be beneficially (might need to drop target network update steps for it)
        idxes = range(self.num_heads)
        #idxes = random.sample(range(self.num_heads), self.num_samples)
        for idx in idxes:
            self.policy_heads[idx].update_targets(n_steps)

    def calculate_TDE(self, state, action, next_state, reward, done):
        # TOOD: implement
        return torch.tensor([0])

    def display_debug_info(self):
        pass






