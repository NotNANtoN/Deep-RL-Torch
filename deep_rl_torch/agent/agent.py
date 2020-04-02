# External imports:
import collections
import itertools
import logging
import gym
import torch
import time
from gym.spaces import Discrete
from pytorch_memlab import LineProfiler, profile, profile_every, set_target_gpu

# Internal Imports:
from deep_rl_torch.nn import ProcessState, ProcessStateAction
from deep_rl_torch.policies import Q_Policy, ActorCritic, REM, MineRLHierarchicalPolicy
from deep_rl_torch.nn.nn_utils import count_parameters, count_model_parameters
from deep_rl_torch.util import *

try:
    from apex import amp
except:
    print("WARNING: apex could not be imported.")


class Agent:
    def __init__(self, env, device, log, hyperparameters):
        self.verbose = hyperparameters["verbose"]
        self.discrete_env = True if "Discrete" in str(env.action_space)[:8] else False
        if self.verbose:
            print(env.action_space)
            print("Env with discrete action space: ", self.discrete_env)
        self.env = env
        self.device = device
        self.log = log
        self.hyperparameters = hyperparameters

        self.updates_per_step = hyperparameters["updates_per_step"]
        self.optimize_centrally = hyperparameters["optimize_centrally"]
        self.use_half = hyperparameters["use_half"] and torch.cuda.is_available()
        self.use_actor_critic = hyperparameters["use_actor_critic"]
        self.save_path = hyperparameters["save_path"]
        self.save_threshold = hyperparameters["save_percentage"]
        self.stored_percentage = 0
        self.load_path = hyperparameters["load_path"]
        self.batch_size = hyperparameters["batch_size"]
        # Frame stacking:
        self.use_list = hyperparameters["use_list"]
        self.stack_dim = hyperparameters["stack_dim"]
        self.stack_count = hyperparameters["frame_stack"]
        self.store_stacked = hyperparameters["store_stacked"]
        self.recent_states = collections.defaultdict(lambda: collections.deque(maxlen=self.stack_count))

        # Create NNs and support structures:
        self.F_s, self.F_sa = self.init_feature_extractors()
        self.policy = self.create_policy()

        # Collect all parameters:
        params = self.policy.get_updateable_params()
        params += self.F_s.get_updateable_params()
        if self.F_sa is not None:
            params += self.F_sa.get_updateable_params()
        if self.verbose:
            print("Total trainable parameters: ", count_parameters(params))
        # Set up Optimizer:
        if self.optimize_centrally:
            self.optimizer = self.create_optimizer(hyperparameters, params)

    def create_optimizer(self, hyperparameters, parameters_to_optimize):
        optimizer_type = hyperparameters["optimizer"]
        kwargs = {}
        kwargs["lr"] = hyperparameters["general_lr"]
        if optimizer_type == "Adam":
            if hyperparameters["Adam_epsilon"]:
                kwargs["eps"] = hyperparameters["Adam_epsilon"]
            elif hyperparameters["Adam_beta1"] or hyperparameters["Adam_beta2"]:
                assert hyperparameters["Adam_beta1"] and hyperparameters["Adam_beta2"]
                kwargs["betas"] = (hyperparameters["Adam_beta1"], hyperparameters["Adam_beta2"])
        optimizer = optimizer_type(parameters_to_optimize, **kwargs)
        if self.use_half:
            _, optimizer = amp.initialize([], optimizer)
        return optimizer
        

    def init_feature_extractors(self):
        state_sample = self.env.observation_space.sample()
        if not self.use_list:
            state_sample = self.fake_stack(state_sample)

        print("State sample shape: ", state_sample.shape)
        F_s = ProcessState(state_sample, self.env, self.log, self.device, self.hyperparameters)
        if self.log and self.verbose:
            print("F_s:")
            print(F_s)
            print("Trainable params: ", count_model_parameters(F_s))
        if self.use_half:
            F_s = amp.initialize(F_s, verbosity=0)
        F_sa = None
        if self.use_actor_critic:
            state_feature_len = F_s.layers_merge[-1].out_features
            F_sa = ProcessStateAction(state_feature_len, self.env, self.log, self.device, self.hyperparameters)
            if self.log and self.verbose:
                print("F_sa:")
                print(self.F_sa)
            if self.use_half:
                F_sa = amp.initialize(F_sa, verbosity=0)
        return F_s, F_sa

    def create_policy(self):
        # Define base policy of HRL or the general algorithm and ground policy of REM:
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
            if self.verbose:
                print("Base Policy (will act concretely): ", base_policy)
                print("Ground Policy (will use base policy): ", ground_policy)
            if self.hyperparameters["use_MineRL_policy"]:
                if self.verbose:
                    print("Use Hierarchical MineRL policy!")
                policy = MineRLHierarchicalPolicy(ground_policy, base_policy, self.F_s, self.F_sa, self.env,
                                                  self.device, self.log, self.hyperparameters)
            else:
                policy = base_policy(ground_policy, self.F_s, self.F_sa, self.env, self.device, self.log,
                                     self.hyperparameters)
        return policy

    #@profile
    def remember(self, state, action, reward, done, filling_buffer=False):
        self.policy.remember(state, action, reward, done, filling_buffer=filling_buffer)

    #@profile
    def optimize(self, steps_done, train_fraction):
        """ Takes care of general optimization procedure """
        num_updates = int(self.updates_per_step) if self.updates_per_step >= 1\
                                                    else steps_done % int(1 / self.updates_per_step) == 0
        for _ in range(num_updates):
            # Optimize the agent (on the target network)      
            self.optimize_nets()
        # Update the target networks
        self.update_targets(steps_done, train_fraction=train_fraction)
        # Reduce epsilon and other exploratory values:
        self.decay_exploration(steps_done, train_fraction)
        self.save(train_fraction)
                    
    def backward(self, loss):
        if self.use_half:
            with amp.scale_loss(loss, self.optimizer) as loss_scaled:
                loss_scaled.backward()
        else:
            loss.backward()
            
    def optimize_nets(self):
        """ Optimizes the networks by sampling a batch """
        loss = self.policy.optimize()

        if self.optimize_centrally:
            all_nets = [self.F_s, self.policy]
            if self.F_sa is not None:
                all_nets.append(self.F_sa)

            self.optimizer.zero_grad()
            self.backward(loss)
            # Scale the norm of gradients if necessary:
            for net in all_nets:
                net.norm_gradient()
                net.scale_gradient()
            # Scale gradient of networks according to how many outgoing networks it receives gradients from
            #self.F_s.scale_gradient()
            #if self.F_sa is not None:
            #    self.F_sa.scale_gradient()
            #self.policy.scale_gradient()
            # TODO: do we need to scale the gradients of the policy?
            #  Could be scaled according to ratio of network lr to general_lr but it is much smarter to scale loss directly
            self.optimizer.step()
    
            # Log gradients and weights:
            for net in all_nets:
                net.log_nn_data()

    def observe(self, obs, source):
        state = self.make_state(obs, source)
        self.F_s.observe(state)
    
    def calc_mem_usage(self):
        """ Calculates the memory usage of training.
        
        Returns the use CUDA memory of optimizing the network on a single transition.
        
        Requires that enough transitions are stored in memory such that self.policy.optimize is callable"""
        if not torch.cuda.is_available():
            return
        # create tensor on GPU
        test_tensor = torch.tensor([0], dtype=torch.bool, device=self.device)
        batch_results = []
        for _ in range(1):
            # Measure baseline:
            torch.cuda.reset_peak_memory_stats()
            base_use = torch.cuda.max_memory_allocated()
            if self.verbose:
                print("base: ", base_use / 1024 / 1024)
            loss = self.policy.optimize()
            self.backward(loss)
            after_opt = torch.cuda.max_memory_allocated()
            if self.verbose:
                print("after: ", after_opt / 1024 / 1024)
            batch_use = after_opt - base_use
            batch_results.append(batch_use)
        mean_batch = np.mean(batch_results)
        single_transition = mean_batch / self.batch_size
        self.policy.mem_usage = single_transition
        if self.verbose:
            print("GPU Mem usage per transition in Mb: ", single_transition / 1024 / 1024)
        return single_transition

    def fake_stack(self, state_sample):
        shp = list(state_sample.shape)
        if self.stack_count <= 1:
            if shp[0] == 1 or len(shp) == 1:
                state_sample = state_sample
            else:
                state_sample = np.expand_dims(state_sample, 0)
        elif shp[0] == 1 or len(shp) == 1:
            state_sample = np.concatenate([state_sample] * self.stack_count)
        else:
            state_sample = np.stack([state_sample] * self.stack_count)
        return state_sample

    def make_state(self, obs, source):
        if self.stack_count <= 1 or self.use_list:
            if not isinstance(obs, torch.Tensor):
                state = obs.make_state()
            else:
                state = obs
        else:
            obs = obs.squeeze(0)
            self.recent_states[source].append(obs)
            obs_list = list(self.recent_states[source])
            for _ in range(self.stack_count - len(obs_list)):
                obs_list.append(obs_list[-1])
            state = torch.cat(obs_list, dim=self.stack_dim).unsqueeze(0)
        return state

    def clean_state(self, source):
        if source in self.recent_states:
            del self.recent_states[source]

    def explore(self, obs, source, fully_random=False):
        state = self.make_state(obs, source)
        return self.policy.explore(state, fully_random)

    def exploit(self, obs, source):
        state = self.make_state(obs, source)
        return self.policy.exploit(state)

    def decay_exploration(self, n_steps, train_fraction):
        self.policy.update_parameters(n_steps, train_fraction)

    def calculate_Q_and_TDE(self, state, action, next_state, reward, done):
        return self.policy.calculate_TDE(state, action, next_state, reward, done)

    def update_targets(self, n_steps, train_fraction):
        self.policy.F_s.update_targets(n_steps)
        if self.policy.F_sa is not None:
            self.policy.F_sa.update_targets(n_steps)
        self.policy.update_targets(n_steps, train_fraction)

    def save(self, train_fraction):
        # Check if a certain percentage of training time was reached:
        if self.save_threshold and train_fraction - self.stored_percentage >= self.save_threshold:
            self.stored_percentage = train_fraction

            # "train" folder
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            # tb_comment folder (model specific):
            tb_folder = self.save_path + self.hyperparameters["tb_comment"] + "/"
            if not os.path.exists(tb_folder):
                os.mkdir(tb_folder)
            # Folder of the current training percentage
            current_folder = tb_folder + str(int(train_fraction * 100)) + "/"
            if not os.path.exists(current_folder):
                os.mkdir(current_folder)

            # Save models:
            # torch.save(self.policy.F_s, current_folder + "F_s.pth")
            self.policy.F_s.save(current_folder + "F_s/")
            # self.policy.F_s.save(self.save_path)
            if self.policy.F_sa is not None:
                self.policy.F_sa.save(current_folder + "F_sa/")
            self.policy.save(current_folder)

            # TODO: track performance of the model from every last checkpoint and save the model as "best" if it had the best performance so far

    def load(self):
        if self.load_path:
            path = self.load_path
        else:
            path = self.save_path + self.hyperparameters["tb_comment"] + "/"
            saved_models = os.listdir(path)
            best = ""
            for dir in saved_models:
                if best == "" or int(dir) > int(best):
                    best = dir
            path += best + "/"
            # TODO: best could/should also be the model with the highest test score instead of the latest model
        print("Loading model from: ", path)

        self.policy.F_s.load(path + "F_s/")
        if self.policy.F_sa is not None:
            self.policy.F_sa.load(path + "F_sa/")
        self.policy.load(path)

    def display_debug_info(self):
        self.policy.display_debug_info()

    def freeze_normalizers(self):
        self.policy.freeze_normalizers()

