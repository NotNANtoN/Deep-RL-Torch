import random
import torch
import numpy as np

from util import *
from networks import Q, V, Actor


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

    def remember(self, state, action, next_state, reward):
        raise NotImplementedError

    def display_debug_info(self):
        raise NotImplementedError


class Agent(AgentInterface):
    def __init__(self, env, device, normalizer, log, hyperparameters):
        self.discrete_env = True if "Discrete" in str(env.action_space)[:8] else False
        self.env = env
        self.device = device
        self.normalizer = normalizer
        self.log = log
        self.parameters = hyperparameters
        self.memory = ReplayMemory(hyperparameters)
        # TODO: put memory into policy

        self.policy = self.create_policy()

    def create_policy(self):
        # Define base policy of HRL or the general algorithm:
        # TODO: add decicion whether to create REM as base_policy
        if self.parameters["use_actor_critic"]:
            base_policy = ActorCritic
        elif self.discrete_env:
            base_policy = Q
        else:
            raise NotImplementedError("The environment must be discrete to apply Q-Learning, no other"
                                      " framework than Actor-Critic available")
        # Decide whether to use Hierarchical Reinforcement Learning:
        if self.parameters["use_hrl"]:
            policy = HierarchicalPolicy(base_policy, self.log, self.parameters)
        else:
            policy = base_policy(self.env, self.device, self.log, self.parameters)
        return policy


    def optimize(self):
        self.policy.optimize()

    def optimize_old(self):

        # TODO: add optimization for TDEC! It would most likely be best to allow TDEC to have its own critics, therefore we need a sub-function here.

        # Get Batch:
        state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask, TDEC_reward_batch = self.get_batch()

        state_features = self.F_s(state_batch)
        state_action_features = self.F_sa(state_features,
                                          action_batch)  # TODO not necessary if CACLA and no world model etc.

        with torch.no_grad():
            non_final_next_state_features = self.F_s(non_final_next_states)

        if self.V_net:
            self.V_net.optimize(state_features, reward_batch, non_final_next_state_features, non_final_mask)

        if self.Q_net:
            self.Q_net.optimize(state_features, state_action_features, action_batch, reward_batch,
                                non_final_next_state_features, non_final_mask)
            # TODO: possibly split up Q class into one that takes state_features and one tha ttakes state-atcion features as input

        if self.actor:
            # TODO: actor has access to Q net
            self.actor.optimize(state_features, reward_batch, action_batch, non_final_next_states, non_final_mask)

        if self.world_model:
            self.world_model.optimize()
            # TODO: create a world model at some point

        # V-net updates:
        if self.V_net:
            if "r_V" in V_pred_batch_state:
                TDE_r_V = self.optimize_net(self.V_net, V_pred_batch_state["r_V"], reward_batch, name="r_V")
                TDE_R_V, V_expectations_next_state = self.optimize_critic(self.V_net, V_pred_batch_state["R_V"],
                                                                          reward_batch, non_final_next_states,
                                                                          non_final_mask, name="R_V")
                TDE_V = TDE_r_V + TDE_R_V
            else:
                TDE_V = self.optimize_critic(self.V_net, V_pred_batch_state["V"], reward_batch,
                                             non_final_next_states, non_final_mask, name="V")

        # Q-net updates:
        if self.Q_net:
            if "r_Q" in Q_pred_batch_state_action:
                TDE_r_Q = self.optimize_net(self.Q_net, Q_pred_batch_state_action["r_Q"], reward_batch, name="r_Q")
                TDE_R_Q = self.optimize_critic(self.Q_net, Q_pred_batch_state_action["R_Q"], reward_batch,
                                               non_final_next_states, non_final_mask, name="R_Q")
                TDE_Q = TDE_r_Q + TDE_R_Q
            else:
                TDE_Q, Q_expectations_next_state = self.optimize_critic(self.Q_net, Q_pred_batch_state_action["Q"],
                                                                        reward_batch, non_final_next_states,
                                                                        non_final_mask, name="Q",
                                                                        precalculated_next_state_expectations=
                                                                        V_expectations_next_state)

    def explore(self, state):
        return self.policy.explore(state)

    def exploit(self, state):
        return self.policy.exploit(state)

    def decay_exploration(self, n_steps):
        self.policy.decay_exploration(n_steps)

<<<<<<< HEAD
    # def reset_old(self):
    #     if self.USE_EXP_REP:
    #         # TODO: extend replayMemory class with prioritized exp rep
    #         self.memory = ReplayMemory(self.replay_buffer_size)
    #     else:
    #         self.initialize_workers()
    #
    #     # INIT networks:
    #
    #     # TODO: networks need to set up their own optimizers
    #
    #     # TODO: if QV_MAX, set USE_QV to True
    #     if self.USE_QV or self.USE_CACLA:
    #         # init V net
    #         if self.USE_CACLA:
    #         # init V net
    #     else:
    #         # init Q net
    #         if self.USE_AC:
    #         # if we use Actor critic (any sort) set this variable to True and initialize Actor

        if self.USE_TDEC:
            pass
            # at some point: initalize TDEC, which uses same structure as above (QV etc)

        # Initialize actor:
        if not self.discrete_env:
            self.actor = Actor(self.state_len, self.num_actions, HIDDEN_NEURONS=self.hidden_neurons,
                               HIDDEN_LAYERS=self.hidden_layers, activation_function=self.activation_function,
                               normalizer=self.normalizer, offset=self.critic_output_offset).to(self.device)
            self.target_actor = Actor(self.state_len, self.num_actions, HIDDEN_NEURONS=self.hidden_neurons,
                                      HIDDEN_LAYERS=self.hidden_layers, activation_function=self.activation_function,
                                      normalizer=self.normalizer, offset=self.critic_output_offset).to(self.device)
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_actor.eval()

            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Initialize Q and V networks:
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

=======
>>>>>>> 8fb5e298cad2a52b5e5cf186dfc054f7be8fe48f
    def calculate_TDE(self, batch, store_log=True):
        TDE = self.policy.calculate_TDE(batch)

        if store_log:
            self.log.add("TDE", TDE.item())

        return TDE

class BasePolicy:
    def __init__(self, hyperparameters):
        # TODO: move F_s F_sa creation possibly into Agent to share among all policies
        self.F_s, self.F_sa = self.init_feature_extractors()
        self.actor, self.critic = self.init_actor_critic(self.F_s, self.F_sa)
        self.epsilon = hyperparameters["epsilon"]

    def random_action(self):
        if self.discrete_env:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)
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
            action = torch.argmax(action, dim=-1)
        return action

    def add_noise(self, action):
        if self.action_gaussian_noise_std:
            action += torch.tensor(np.random.normal(0, self.action_gaussian_noise_std, len(action)), dtype=torch.float)
            action = torch.tensor(np.clip(action, self.action_low, self.action_high))
        return action

    def explore(self, state):
        # Epsilon-Greedy:
        if self.epsilon:
            sample = random.random()
            if sample < self.epsilon:
                return self.random_action()
        # Raw action:
        state_features = self.F_s(state)
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
        action = self.actor(state).detach()
        if self.discrete_env:
            action = torch.argmax(action, dim=-1)
        return action

    def init_feature_extractors(self):
        F_s = StateProcessor()
        F_sa = StateActionProcessor()
        return F_s, F_sa

    def init_actor_critic(self, F_s, F_sa, hyperparameters):
        # TODO: first create state_processor and state_action_processor and add them to the optimizers of the actor/critic
        critic = self.init_critic(F_s, F_sa, hyperparameters)
        actor = self.init_actor(critic, F_s, hyperparameters)
        return actor, critic

    def init_critic(self, F_s, F_sa, hyperparameters):
        # TODO: differentiate between DQN critic and AC critic by checking in Q __init__ for use_actor_critic
        self.Q = Q(env, device, log, hyperparameters, F_s, F_sa)

        if self.use_QV or self.use_QVMAX:
            # Init Networks:
            self.V = V(env, device, log, hyperparameters, F_s)
            # Create links for training value calculations:
            self.Q.V = self.V
            self.V.Q = self.Q
        return self.Q

    def train_critic(self, state_feature_batch, state_action_features, reward_batch, non_final_next_state_features,
                     non_final_mask):
        if self.use_QV or self.use_QVMAX:
            self.V.optimize(state_feature_batch, reward_batch, non_final_next_state_features, non_final_mask)
        self.Q.optimize(state_feature_batch, state_action_features, reward_batch, non_final_next_state_features,
                            non_final_mask)


    def optimize(self):
        # Get Batch:
        transitions = self.get_transitions()
        state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask = self.extract_batch(transitions)
        # Extract features:
        state_feature_batch = self.F_s(state_batch)
        non_final_next_state_features = self.F_s(non_final_next_states)
        # Optimize:
        self.optimize_networks(state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                               non_final_mask)

    def decay_exploration(self, n_steps):
        self.epsilon *= eps_decay
        self.log.add("Epsilon", self.epsilon)
        # TODO: decay temperature for Boltzmann if that exploration is used

    def calculate_TDE(self, batch):
        return self.critic.calculate_TDE(batch)

    def get_transitions(self):
        sampling_size = min(len(self.memory), self.batch_size)
        return self.memory.sample(sampling_size)

    def extract_batch(self, transitions):
        # TODO: How fast is this? Can't we put it into arrays?
        # TODO: We sample, then concatenate the sampled parts into multiple arrays, transposing it before...

        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = self.normalizer.normalize(torch.cat([s for s in batch.next_state
                                                                     if s is not None]))
        state_batch = self.normalizer.normalize(torch.cat(batch.state))
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask

    def optimize_networks(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                               non_final_mask):
        raise NotImplementedError


class ActorCritic(BasePolicy):
    def __init__(self, env, device, log, hyperparameters):
        self.discrete_env = True if "Discrete" in str(env.action_space)[:8] else False
        if self.discrete_env:
            self.num_actions = env.action_space.n
        else:
            self.num_actions = len(env.action_space.low)
            self.action_low = torch.tensor(env.action_space.high)
            self.action_high = torch.tensor(env.action_space.low)
        self.log = log
        self.parameters = hyperparameters
        super(ActorCritic, self).__init__(hyperparameters)


    def optimize_networks(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                          non_final_mask):
        # TODO: possible have an action normalizer? For state_features we could have a batchnorm layer, maybe it is better for both
        # TODO: for HRL this might be nice
        state_action_features = self.F_sa(state_feature_batch, action_batch)
        self.train_critic(state_feature_batch, state_action_features, reward_batch, non_final_next_state_features,
                          non_final_mask)
        self.train_actor(state_feature_batch, action_batch, reward_batch, non_final_next_state_features, non_final_mask)

    def init_actor(self, critic):
        self.actor = Actor(env, device, log, hyperparameters)
        self.actor.Q = critic

    def train_actor(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features, non_final_mask):
        self.actor.optimize(state_feature_batch, action_batch, reward_batch, non_final_next_state_features, non_final_mask)


class Q_Policy(BasePolicy):
    def __init__(self):
        super(Q_Policy, self).__init__(hyperparameters)

    def optimize_networks(self, state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                          non_final_mask):
        self.train_critic(state_feature_batch, action_batch, reward_batch, non_final_next_state_features,
                             non_final_mask)

    def init_actor(self, critic):
        return critic


class REM(BasePolicy):
    def __init__(self, base_policy):
        # 1. For ensemble: simply create many base policies, train them all with .optimize() and sum action output.
        #    When creating policies, have them share seam feature processor (possibly create one in this class and pass it to base policies)
        # 2. For REM: pick random subset to train
        # 3. For new idea with Matthias: have additional SOM
        pass
