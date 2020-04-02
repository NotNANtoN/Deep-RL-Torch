import random
import os

import torch

from .policies import BasePolicy

# 1. For ensemble: simply create many base policies, train them all with .optimize() and sum action output.
#    When creating policies, have them share seam feature processor (possibly create one in this class and pass it to base policies)
# 2. For REM: pick random subset to train
# 3. For new idea with Matthias: have additional SOM
class REM(BasePolicy):
    def __init__(self, ground_policy, F_s, F_sa, env, device, log, hyperparameters):
        super(REM, self).__init__(ground_policy, F_s, F_sa, env, device, log, hyperparameters)

        self.name = "REM"
        self.num_heads = hyperparameters["REM_num_heads"]
        self.num_samples = hyperparameters["REM_num_samples"]

        # Create ensemble of ground policies:
        # TODO: maybe let all the critics of the ground policy share their reward nets if they are split
        self.policy_heads = [ground_policy(None, F_s, F_sa, env, device, log, hyperparameters)
                             for _ in range(self.num_heads)]
        for head in self.policy_heads:
            head.set_retain_graph(True)

        # Sample idxs:
        self.idxs = None

    def set_name(self, name):
        self.name = "REM" + str(name)
        for idx, head in enumerate(self.policy_heads):
            head.set_name(self.name + str(idx))

    def init_actor(self, Q, V, F_s):
        return None

    def init_critic(self, F_s, F_sa):
        return None, None

    # TODO: atm we iterate through the list of sampled base_policies - can this be done in a better way? MPI, GPU like?
    def optimize_networks(self, transitions):
        self.idcs = random.sample(range(self.num_heads), self.num_samples)
        error = 0
        loss = 0
        for idx in self.idcs:
            current_policy = self.policy_heads[idx]
            # TODO: here we might bootstrap over our batch to have slightly different training data per policy!
            error_current, loss_current = current_policy.optimize_networks(transitions)
            error += error_current
            loss += loss_current
        return error / self.num_samples, loss

    def choose_action(self, state, calc_state_features=True):
        # Preprocess:
        if calc_state_features:
            state = self.state2device(state)
            state_features = self.F_s(state)
        else:
            state_features = state
        # Select random subset to output action:
        idxes = random.sample(range(self.num_heads), self.num_samples)
        summed_action = None
        for idx in idxes:
            current_policy = self.policy_heads[idx]
            with torch.no_grad():
                action = current_policy.actor(state_features)
            if summed_action is None:
                summed_action = action
            else:
                summed_action += action
        return summed_action / self.num_samples

    def update_targets(self, n_steps, train_fraction=None):
        for head in self.policy_heads:
            head.update_targets(n_steps, train_fraction=train_fraction)

    def update_parameters(self, n_steps, train_fraction):
        for head in self.policy_heads:
            head.update_parameters(n_steps, train_fraction)

    def calculate_TDE(self, state, action, next_state, reward, done):
        q = 0
        tde = 0
        for policy in self.policy_heads:
            q_current, tde_current = policy.calculate_Q_and_TDE(state, action, next_state, reward, done)
            q += q_current
            tde += tde_current
        return q, tde_current


    def display_debug_info(self):
        pass

    def save(self, path):
        path += "REM/"
        if not os.path.exists(path):
            os.mkdir(path)

        for idx, policy in enumerate(self.policy_heads):
            policy_path = path + str(idx) + "/"
            if not os.path.exists(policy_path):
                os.mkdir(policy_path)
            policy.save(policy_path)

    def get_updateable_params(self):
        params = None
        for net in self.policy_heads:
            new_params = list(net.get_updateable_params())
            if params is None:
                params = new_params
            else:
                params += new_params
        return params

    def log_nn_data(self, name=""):
        for idx, policy in enumerate(self.policy_heads):
            if self.idxs is not None and idx in self.idcs:
                policy.log_nn_data(name=name + policy.name)


