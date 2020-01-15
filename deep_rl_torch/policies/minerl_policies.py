import copy

import torch
from gym.spaces import Discrete

from .policies import BasePolicy

class MineRLPolicy(BasePolicy):
    def __init__(self, ground_policy, base_policy, F_s, F_sa, env, device, log, hyperparameters):
        super(MineRLPolicy, self).__init__(ground_policy, F_s, F_sa, env, device, log, hyperparameters)

        self.base_policy = base_policy

        self.jump_options = env.jump_options
        self.attack_options = env.attack_options
        self.lateral_options = env.lateral_options
        self.straight_options = env.straight_options
        self.camera_x_options = env.camera_x_options_string
        self.camera_y_options = env.camera_y_options_string

        self.num_jump_actions = len(self.jump_options)
        self.num_attack_actions = len(self.attack_options)
        self.num_lateral_actions = len(self.lateral_options)
        self.num_straight_actions = len(self.straight_options)
        self.num_camera_x_actions = len(self.camera_x_options)
        self.num_camera_y_actions = len(self.camera_y_options)
        self.num_camera_actions = self.num_camera_x_actions * self.num_camera_y_actions
        self.num_move_actions = self.num_jump_actions * self.num_attack_actions * self.num_lateral_actions * \
                                self.num_straight_actions * self.num_camera_actions

    def create_adjusted_action_policy(self, num_actions, shift, action_mapping, counter, move_policy=False, name=""):
        action_space = Discrete(num_actions)
        real_action_space = self.env.action_space
        self.env.action_space = action_space
        if num_actions == 1:
            new_policy = self.DummyPolicy(self.batch_size)
        elif move_policy:
            new_policy = MineRLMovePolicy(self.ground_policy, self.base_policy, self.F_s, self.F_sa, self.env,
                                          self.device,
                                          self.log, self.hyperparameters)
        else:
            new_policy = self.base_policy(self.ground_policy, self.F_s, self.F_sa, self.env, self.device, self.log,
                                          self.hyperparameters)
        new_policy.set_retain_graph(True)
        new_policy.shift = shift
        new_policy.set_name(name)
        action_mapping.extend([counter for _ in range(num_actions)])
        self.env.action_space = real_action_space
        return new_policy, shift + num_actions

    def init_actor(self, Q, V, F_s):
        return None

    def init_critic(self, F_s, F_sa):
        return None, None

    def calculate_Q_and_TDE(self, state, action, next_state, reward, done):
        return 0, 0

    def display_debug_info(self):
        pass

    def set_name(self, name):
        self.name = str(name)

    class DummyPolicy():
        def __init__(self, batch_size):
            self.batch_size = batch_size
            self.name = ""

        def choose_action(self, state, calc_state_features=False):
            return torch.ones(1, 1)

        def optimize_networks(self, transitions):
            return torch.zeros(self.batch_size, 1), 0

        def set_retain_graph(self, val):
            pass

        def set_name(self, val):
            pass

        def get_updateable_params(self):
            return []

        def update_targets(self, steps, train_fraction=None):
            pass

        def save(self, path):
            pass

        def load(self, path):
            pass

        def log_nn_data(self, name=""):
            pass


class MineRLObtainPolicy(MineRLPolicy):
    def __init__(self, ground_policy, base_policy, F_s, F_sa, env, device, log, hyperparameters):
        super(MineRLObtainPolicy, self).__init__(ground_policy, base_policy, F_s, F_sa, env, device, log,
                                                 hyperparameters)

        # Create policies:
        print("Creating high-level policy:")
        self.decider, _ = self.create_adjusted_action_policy(6, 0, [], 0)
        shift = 0
        self.action_mapping = []
        print("Creating low-level policies:")
        self.mover, shift = self.create_adjusted_action_policy(9, shift, self.action_mapping, 0)
        self.placer, shift = self.create_adjusted_action_policy(6, shift, self.action_mapping, 1)
        self.equipper, shift = self.create_adjusted_action_policy(7, shift, self.action_mapping, 2)
        self.crafter, shift = self.create_adjusted_action_policy(4, shift, self.action_mapping, 3)
        self.nearby_crafter, shift = self.create_adjusted_action_policy(7, shift, self.action_mapping, 4)
        self.nearby_smelter, shift = self.create_adjusted_action_policy(2, shift, self.action_mapping, 5)
        print()

        self.lower_level_policies = (self.mover, self.placer, self.equipper, self.crafter, self.nearby_crafter,
                                     self.nearby_smelter)

    def action2high_low_level(self, actions):
        high_lvl = []
        low_lvl = []
        for action in actions:
            action = action.item()
            high_lvl_action = self.action_mapping[action]
            low_lvl_action = action - self.lower_level_policies[high_lvl_action].shift
            high_lvl.append(high_lvl_action)
            low_lvl.append(low_lvl_action)
        high_lvl = torch.tensor(high_lvl).unsqueeze(1)
        low_lvl = torch.tensor(low_lvl).unsqueeze(1)
        return high_lvl, low_lvl

    def get_masks(self, actions, num_low_lvl=6):
        # Aggregate idxs for lower-level policies to operate on:
        idxs = [[] for _ in range(num_low_lvl)]
        for idx, action in enumerate(actions):
            idxs[action.item()].append(idx)
        return idxs

    def optimize_networks(self, transitions):
        error = 0
        # Save actions:
        original_actions = transitions["action_argmax"].clone()
        # Transform action idx such as 34 into e.g. ([3], [8])
        high_level_actions, low_level_actions = self.action2high_low_level(original_actions)
        # Train high-level policy:
        transitions["action_argmax"] = high_level_actions
        error += self.decider.optimize_networks(transitions)
        # Get mask of which low-level policy trains on which part of the transitions:
        mask_list = self.get_masks(high_level_actions)
        # Train low-level policies:
        for policy_idx, idx_mask in enumerate(mask_list):
            if not idx_mask:
                continue
            transitions["action_argmax"][idx_mask] = low_level_actions[idx_mask]
            # Apply mask to transition dict and dicts within dict:
            partial_transitions = {
                key: None if transitions[key] is None else transitions[key] if isinstance(transitions[key], list) else {
                    sub_key: transitions[key][sub_key][idx_mask] for sub_key in transitions[key]} if isinstance(
                    transitions[key], dict) else transitions[key][idx_mask] for key in transitions}
            policy = self.lower_level_policies[policy_idx]
            error[idx_mask] += policy.optimize_networks(partial_transitions)
        # Reset actions just in case:
        transitions["action_argmax"] = original_actions
        return error

    def apply_lower_level_policy(self, policy, state):
        with torch.no_grad():
            actions_vals = policy(state)
        action_idxs = torch.argmax(actions_vals, dim=1) + policy.shift
        return action_idxs

    def choose_action(self, state, calc_state_features=True):
        # Preprocess:
        if calc_state_features:
            state_features = self.F_s(state)
        else:
            state_features = state
        # Preprocess:
        action_q_vals = torch.zeros(state_features.shape[0], self.num_actions)
        # Apply high-level policy:
        with torch.no_grad():
            action = self.decider.choose_action(state_features, calc_state_features=False)
        high_level_actions = torch.argmax(action, dim=1)
        # print("High level actions: ", high_level_actions)
        masks = self.get_masks(high_level_actions)
        # print("Masks: ", masks)
        # Apply lower-level policies:
        for policy_idx, mask in enumerate(masks):
            if not mask:
                continue
            # print("Mask: ", mask)
            # print("state shape: ", state_features.shape)
            # print("action q vals masked shape: ", action_q_vals[mask].shape)
            # print("action q vals masked: ", action_q_vals[mask])
            # print("state masked shape: ", state_features[mask].shape)
            policy = self.lower_level_policies[policy_idx]
            shift = policy.shift
            low_lvl_action = policy.choose_action(state_features, calc_state_features=False)
            # print("low level action shape: ", low_lvl_action.shape)
            # print("low level action: ", low_lvl_action)
            action_q_vals[0][shift: shift + len(low_lvl_action[0])] = low_lvl_action[0]
        return action_q_vals

    def update_targets(self, n_steps, train_fraction=None):
        self.decider.update_targets(n_steps, train_fraction=train_fraction)
        for policy in self.lower_level_policies:
            policy.update_targets(n_steps, train_fraction=train_fraction)


class MineRLMovePolicy(MineRLPolicy):
    def __init__(self, ground_policy, base_policy, F_s, F_sa, env, device, log, hyperparameters):
        super(MineRLMovePolicy, self).__init__(ground_policy, base_policy, F_s, F_sa, env, device, log, hyperparameters)

        self.name = "Mover"

        self._noop_template = env.noop
        print("Creating Move Policy: ")
        self.attacker = self.create_adjusted_action_policy(self.attack_options, name="attacker")
        self.lateralus = self.create_adjusted_action_policy(self.lateral_options, name="lateralus")
        self.straightener = self.create_adjusted_action_policy(self.straight_options, name="straightener")
        self.jumper = self.create_adjusted_action_policy(self.jump_options, name="jumper")
        self.camera_xer = self.create_adjusted_action_policy(self.camera_x_options, name="camera_x")
        self.camera_yer = self.create_adjusted_action_policy(self.camera_y_options, name="camera_y")
        self.policies = [self.attacker, self.lateralus, self.straightener, self.jumper, self.camera_xer,
                         self.camera_yer]
        print()

    def set_name(self, name):
        self.name = str(name)

    def create_adjusted_action_policy(self, options, name=""):
        action_space = Discrete(len(options))
        real_action_space = self.env.action_space
        self.env.action_space = action_space
        if len(options) == 1:
            new_policy = self.DummyPolicy(self.batch_size)
        else:
            new_policy = self.base_policy(self.ground_policy, self.F_s, self.F_sa, self.env, self.device, self.log,
                                      self.hyperparameters)
        new_policy.set_retain_graph(True)
        new_policy.set_name(name)
        for idx, option in enumerate(options):
            if "none" in option:
                options[idx] = None

        new_policy.options = options
        self.env.action_space = real_action_space
        return new_policy

    def get_policy_options(self, policy, state):
        action = policy.choose_action(state, calc_state_features=False)
        action_idxs = torch.argmax(action, dim=1)
        return [policy.options[idx] for idx in action_idxs]

    def apply_options(self, options, noops):
        actions = []
        for idx, noop in enumerate(noops):
            options_to_apply = [option[idx] for option in options]
            for option in options_to_apply:
                if option is None:
                    continue
                elif option[0] == "x":
                    noop["camera"][0] = int(option[2:])
                elif option[0] == "y":
                    noop["camera"][1] = int(option[2:])
                else:
                    noop[option] = 1
            actions.append(noop)
        return actions

    def choose_action(self, state, calc_state_features=True):
        # Preprocess:
        if calc_state_features:
            state_features = self.F_s(state)
        else:
            state_features = state
        # Init q val tensor and action templates
        action_q_vals = torch.zeros(self.num_actions, device=self.device)
        noops = [copy.deepcopy(self._noop_template) for _ in range(state_features.shape[0])]
        # Apply policies and extract semantics:
        options = [self.get_policy_options(policy, state_features) for policy in self.policies]
        actions = self.apply_options(options, noops)
        # Transform actions to match output format:
        action_idx = self.env.dicts2idxs(actions)
        # print("in choose_action of MovePOlicy")
        # print("action q vals shape: ", action_q_vals.shape)
        # print("actoin idx: ", action_idx)
        action_q_vals[
            action_idx] = 1  # Hacky way so that this action is chosen, as the interface requires us to return Q-vals for all possible actions
        # print()
        return action_q_vals.unsqueeze(0)


    def get_action_idxs_for_policy(self, policy, action_dicts):
        action_idxs = torch.zeros(len(action_dicts), device=self.device).long()
        for dict_idx, action_dict in enumerate(action_dicts):
            action_idx = None
            none_idx = None
            for idx, option in enumerate(policy.options):
                if option is None:
                    none_idx = idx
                elif option[0] == "x":
                    number = float(option[2:])
                    if action_dict["camera"][0] == number:
                        action_idx = idx
                        break
                elif option[0] == "y":
                    number = float(option[2:])
                    if action_dict["camera"][1] == number:
                        action_idx = idx
                        break
                else:
                    if action_dict[option] == 1:
                        action_idx = idx
                        break
            if action_idx is None:
                action_idx = none_idx
            action_idxs[dict_idx] = action_idx
        return action_idxs.unsqueeze(1)

    def optimize_networks(self, transitions):
        error = 0
        loss = 0
        # Save actions:
        original_actions = transitions["action_argmax"].clone()
        # Transform actions in dicts:
        action_dicts = [self.env.action(idx.item()) for idx in original_actions]
        for policy in self.policies:
            action_idxs = self.get_action_idxs_for_policy(policy, action_dicts)
            transitions["action_argmax"] = action_idxs

            error_it, loss_it = policy.optimize_networks(transitions)
            error += error_it

            loss += loss_it
        transitions["action_argmax"] = original_actions
        return error, loss

    def update_targets(self, n_steps, train_fraction=None):
        for policy in self.policies:
            policy.update_targets(n_steps, train_fraction=train_fraction)

    def calculate_TDE(self, state, action, next_state, reward, done):
        q = 0
        tde = 0
        for policy in self.policies:
            q_current, tde_current = policy.calculate_Q_and_TDE(state, action, next_state, reward, done)
            q += q_current
            tde += tde_current
        return q, tde_current

    def save(self, path):
        path += self.name + "/"
        if not os.path.exists(path):
            os.mkdir(path)

        for idx, policy in enumerate(self.policies):
            policy.save(path)

    def get_updateable_params(self):
        params = []
        for policy in self.policies:
            params += list(policy.get_updateable_params())
        return params

    def log_nn_data(self, name=""):
        for policy in self.policies:
            policy.log_nn_data(name=name + "_ " + policy.name)



class MineRLHierarchicalPolicy(MineRLPolicy):
    def __init__(self, ground_policy, base_policy, F_s, F_sa, env, device, log, hyperparameters):
        super(MineRLHierarchicalPolicy, self).__init__(ground_policy, base_policy, F_s, F_sa, env, device, log,
                                                       hyperparameters)

        self.name = "HierarchicalMineRLPolicy"
        # Create policies:
        noop = self.env.noop

        shift = 0
        count = 0
        self.action_mapping = []
        print("Creating low-level policies:")
        self.mover, shift = self.create_adjusted_action_policy(self.num_move_actions, shift, self.action_mapping, count,
                                                               move_policy=True, name="mover")
        count += 1
        self.lower_level_policies = [self.mover]
        if "place" in noop:
            num_choices = self.env.wrapping_action_space.spaces["place"].n - 1
            self.placer, shift = self.create_adjusted_action_policy(num_choices, shift, self.action_mapping, count, name="placer")
            self.lower_level_policies.append(self.placer)
            count += 1
        if "equip" in noop:
            num_choices = self.env.wrapping_action_space.spaces["equip"].n - 1
            self.equipper, shift = self.create_adjusted_action_policy(num_choices, shift, self.action_mapping, count, name="equipper")
            self.lower_level_policies.append(self.equipper)
            count += 1
        if "craft" in noop:
            num_choices = self.env.wrapping_action_space.spaces["craft"].n - 1
            self.crafter, shift = self.create_adjusted_action_policy(num_choices, shift, self.action_mapping, count, name="crafter")
            self.lower_level_policies.append(self.crafter)
            count += 1
        if "nearbyCraft" in noop:
            num_choices = self.env.wrapping_action_space.spaces["nearbyCraft"].n - 1
            self.nearby_crafter, shift = self.create_adjusted_action_policy(num_choices, shift, self.action_mapping,
                                                                            count, name="nearbyCrafter")
            self.lower_level_policies.append(self.nearby_crafter)
            count += 1
        if "nearbySmelt" in noop:
            num_choices = self.env.wrapping_action_space.spaces["nearbySmelt"].n - 1
            self.nearby_smelter, shift = self.create_adjusted_action_policy(num_choices, shift, self.action_mapping,
                                                                            count, name="nearbySmelter")
            self.lower_level_policies.append(self.nearby_smelter)
            count += 1
        if len(self.lower_level_policies) > 1:
            print("Creating high-level policy:")
            self.decider, _ = self.create_adjusted_action_policy(len(self.lower_level_policies), 0, [], 0)
        else:
            self.decider = None
        print()


    def action2high_low_level(self, actions):
        high_lvl = []
        low_lvl = []
        for action in actions:
            action = action.item()
            high_lvl_action = self.action_mapping[action]
            low_lvl_action = action - self.lower_level_policies[high_lvl_action].shift
            # if high_lvl_action != 0:
            # print("raw action: ", action)
            # print("high level action: ", high_lvl_action)
            # print("low level action: ", low_lvl_action)
            high_lvl.append(high_lvl_action)
            low_lvl.append(low_lvl_action)
        high_lvl = torch.tensor(high_lvl, device=self.device).unsqueeze(1)
        low_lvl = torch.tensor(low_lvl, device=self.device).unsqueeze(1)
        return high_lvl, low_lvl

    def get_masks(self, actions, num_low_lvl=6):
        # TODO: num_low_lvl needs to be determined properly when calling this function!
        # Aggregate idxs for lower-level policies to operate on:
        idxs = [[] for _ in range(num_low_lvl)]
        for idx, action in enumerate(actions):
            idxs[action.item()].append(idx)
        return idxs

    def apply_mask_to_transitions(self, transitions, idx_mask):
        masked_transitions = {}

        # Deal with non final next states:
        if transitions["non_final_mask"] is None:
            masked_transitions["non_final_mask"] = None
        else:
            non_final_mask = transitions["non_final_mask"]

            def apply_idx_mask_to_mask(idx_mask, mask):
                non_final_idx = 0
                transformed_mask = []  # torch.zeros(non_finals.shape ,dtype=torch.bool)
                idx_mask_cpy = idx_mask[:]
                # Iterate through the non_final_mask to find out which non_final_next states are masked by the idx_mask:
                for mask_idx, non_final_bool in enumerate(mask):
                    if non_final_bool:
                        if mask_idx == idx_mask_cpy[0]:
                            transformed_mask.append(non_final_idx)
                        non_final_idx += 1
                    if mask_idx == idx_mask_cpy[0]:
                        del idx_mask_cpy[0]  # as we iterate from the start the first idx in idx mask is always deleted
                        if len(idx_mask_cpy) == 0:
                            break
                return transformed_mask

            transformed_mask = apply_idx_mask_to_mask(idx_mask, non_final_mask)

            non_finals = transitions["non_final_next_states"]
            masked_transitions["non_final_next_states"] = {key: non_finals[key][transformed_mask] for key in non_finals}
            masked_transitions["non_final_next_state_features"] = transitions["non_final_next_state_features"][
                transformed_mask]
            masked_transitions["non_final_mask"] = transitions["non_final_mask"][idx_mask]

        # Deal with the rest:
        for key in transitions:
            content = transitions[key]
            if content is None:
                new_content = None
            elif key in ("non_final_mask", "non_final_next_states", "non_final_next_state_features"):
                continue
            elif isinstance(content, list):
                new_content = content  # For PER idx
            elif isinstance(content, dict):
                new_dict = {}
                for sub_key in content:
                    sub_content = content[sub_key]
                    new_sub_content = sub_content[idx_mask]
                    new_dict[sub_key] = new_sub_content
                new_content = new_dict
            else:
                new_content = content[idx_mask]
            masked_transitions[key] = new_content

        return masked_transitions

    def optimize_networks(self, transitions):
        loss = 0
        # Save actions:
        original_actions = transitions["action_argmax"].clone()
        # Transform action idx such as 34 into e.g. ([3], [8])
        high_level_actions, low_level_actions = self.action2high_low_level(original_actions)
        # Train high-level policy:
        if self.decider is not None:
            transitions["action_argmax"] = high_level_actions
            error, decider_loss = self.decider.optimize_networks(transitions)
            loss += decider_loss
        else:
            error = torch.zeros(len(original_actions), 1)
        # Get mask of which low-level policy trains on which part of the transitions:
        mask_list = self.get_masks(high_level_actions)
        # Train low-level policies:
        for policy_idx, idx_mask in enumerate(mask_list):
            if not idx_mask:
                continue
            transitions["action_argmax"][idx_mask] = low_level_actions[idx_mask]
            # Apply mask to transition dict and dicts within dict:
            partial_transitions = self.apply_mask_to_transitions(transitions, idx_mask)
            policy = self.lower_level_policies[policy_idx]
            policy_error, policy_loss = policy.optimize_networks(partial_transitions)
            loss += policy_loss

            error[idx_mask] += policy_error

        # Reset actions just in case:
        transitions["action_argmax"] = original_actions
        return error, loss

        # TODO: this needs to be debugged for environments such as tree, because it is not sure if it works without a decider

    def apply_lower_level_policy(self, policy, state):
        with torch.no_grad():
            actions_vals = policy(state)
        action_idxs = torch.argmax(actions_vals, dim=1) + policy.shift
        return action_idxs

    def choose_action(self, state):
        # Preprocess:
        state_features = self.F_s(state)
        # Preprocess:
        action_q_vals = torch.zeros(state_features.shape[0], self.num_actions, device=self.device)
        # Apply high-level policy:
        with torch.no_grad():
            if self.decider is not None:
                action = self.decider.choose_action(state_features, calc_state_features=False)
            else:
                action = torch.tensor([[0]])
        high_level_actions = torch.argmax(action, dim=1)
        # print("High level actions: ", high_level_actions)
        masks = self.get_masks(high_level_actions)
        # print("Masks: ", masks)
        # Apply lower-level policies:
        for policy_idx, mask in enumerate(masks):
            if mask == []:
                continue
            # print("Mask: ", mask)
            # print("state shape: ", state_features.shape)
            # print("action q vals masked shape: ", action_q_vals[mask].shape)
            # print("action q vals masked: ", action_q_vals[mask])
            # print("state masked shape: ", state_features[mask].shape)
            policy = self.lower_level_policies[policy_idx]
            shift = policy.shift
            low_lvl_action = policy.choose_action(state_features, calc_state_features=False)
            # print("low level action shape: ", low_lvl_action.shape)
            # print("low level action: ", low_lvl_action)
            # print("low_lvl_ action: ", low_lvl_action)
            # print("shift of policy: ", shift)
            # print("low lvl action len : ", (len(low_lvl_action[0])))
            # print("policy idx: ", policy_idx)
            action_q_vals[0][shift: shift + len(low_lvl_action[0])] = low_lvl_action[0]
        return action_q_vals

    def calculate_TDE(self, state, action, next_state, reward, done):
        # Preprocess:
        with torch.no_grad():
            state_features = self.F_s(state)
        high_level_actions, low_level_actions = self.action2high_low_level(action)
        if self.decider is not None:
            q, tde = self.decider.calculate_Q_and_TDE(state_features, high_level_actions, next_state, reward, done)
        else:
            q, tde = 0, 0

        policy = self.lower_level_policies[high_level_actions[0]]
        q_lower, tde_lower = policy.calculate_Q_and_TDE(state_features, low_level_actions, next_state, reward, done)
        return q + q_lower, tde + tde_lower
        # TODO: check


    def update_targets(self, n_steps, train_fraction=None):
        if self.decider is not None:
            self.decider.update_targets(n_steps, train_fraction=train_fraction)
        for policy in self.lower_level_policies:
            policy.update_targets(n_steps, train_fraction=train_fraction)

    def calculate_Q_and_TDE(self, state, action, next_state, reward, done):
        # TODO: implement
        return torch.tensor([0])

    def display_debug_info(self):
        pass

    def save(self, path):
        path += self.name + "/"
        if not os.path.exists(path):
            os.mkdir(path)

        if self.decider is not None:
            self.decider.save(path)
        for idx, policy in enumerate(self.lower_level_policies):
            policy.save(path)

    def get_updateable_params(self):
        if self.decider is not None:
            params = list(self.decider.get_updateable_params())
        else:
            params = []
        for policy in self.lower_level_policies:
            params += list(policy.get_updateable_params())
        #params.extend([policy.get_updateable_params() for policy in self.lower_level_policies])
        return params

    def log_nn_data(self, name=""):
        if self.decider is not None:
            self.decider.log_nn_data(name + "decider")
        for policy in self.lower_level_policies:
            policy.log_nn_data(name + policy.name)
