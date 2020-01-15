from .networks import OptimizableNet

class Actor(OptimizableNet):
    def __init__(self, F_s, env, log, device, hyperparameters, is_target_net=False):
        super(Actor, self).__init__(env, device, log, hyperparameters, is_target_net=is_target_net)

        self.name = "Actor"

        # Initiate arrays for output function:
        self.relu_mask = None
        self.sigmoid_mask = None
        self.tanh_mask = None
        self.relu_idxs = None
        self.tanh_idxs = None
        self.sigmoid_idxs = None
        self.scaling = None
        self.offset = None

        # Create layers
        input_size = F_s.layers_merge[-1].out_features
        output_size = self.num_actions if self.discrete_env else len(self.action_low)
        layers = hyperparameters["layers_actor"]
        self.layers, self.act_functs = create_ff_layers(input_size, layers, output_size)
        self.act_func_output_layer = self.create_output_act_func()
        # Put feature extractor on GPU if possible:
        self.to(device)

        # Define optimizer and previous networks
        self.lr = hyperparameters["lr_actor"]
        if not is_target_net:
            self.F_s = F_s
            updateable_parameters = list(self.F_s.get_updateable_params())
        else:
            updateable_parameters = []
        self.optimizer = self.optimizer(list(self.get_updateable_params()) + updateable_parameters, lr=self.lr)

        if self.use_target_net:
            self.target_net = self.create_target_net()

    def forward(self, x):
        x = apply_layers(x, self.layers, self.act_functs)
        x = self.act_func_output_layer(x)
        # print(x)
        return x

    def compute_loss(self, output, target, sample_weights=None):
        # TODO: test if actor training might be better without CrossEntropyLoss. It might be, because we do not need to convert to long!
        if self.use_DDPG:
            loss = abs(target - output)
        elif self.discrete_env:
            loss_func = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(output, target)
        else:
            # TODO: this loss does not help combat the vanishing gradient problem that we have because of the use of sigmoid activations to squash our actions into the correct range
            loss = F.smooth_l1_loss(output, target, reduction='none')

        if sample_weights is not None:
            loss *= sample_weights.squeeze()
        return loss.mean()

    def output_function_continuous(self, x):
        if self.relu_idxs:
            # x[self.relu_idxs] = F.relu(x[self.relu_idxs])
            y = F.relu(x[:, self.relu_idxs])
            with torch.no_grad():
                x[:, self.relu_idxs] = y
        if self.sigmoid_idxs:
            # x[self.sigmoid_idxs] = torch.sigmoid(x[self.sigmoid_idxs])
            y = x[:, self.sigmoid_idxs].sigmoid()
            with torch.no_grad():
                x[:, self.sigmoid_idxs] = y
        if self.tanh_idxs:
            # print("first: ", x)
            # print(self.tanh_idxs)
            # print(x[:, self.tanh_idxs])
            y = x[:, self.tanh_idxs].tanh()
            # x[self.tanh_mask] = torch.tanh(x[self.tanh_mask])
            # print("after: ", y)
            with torch.no_grad():
                x[:, self.tanh_idxs] = y
            # print("inserted: ", x)
        #print("Action: ", x)
        return (x * self.scaling) + self.offset

    def create_output_act_func(self):
        print("Action_space: ", self.env.action_space)
        relu_idxs = []
        tanh_idxs = []
        sigmoid_idxs = []

        # Init masks:
        # self.relu_mask = torch.zeros(self.batch_size, len(self.action_low))
        # self.relu_mask.scatter_(1, torch.tensor(relu_idxs).long(), 1.)
        # self.tanh_mask = torch.zeros(self.batch_size, len(self.action_low))
        # self.tanh_mask.scatter_(1, torch.tensor(tanh_idxs).long(), 1.)
        # self.sigmoid_mask = torch.zeros(self.batch_size, len(self.action_low))
        # self.sigmoid_mask.scatter_(1, torch.tensor(sigmoid_idxs).long(), 1.)

        if self.discrete_env:
            print("Actor has only sigmoidal activation function")
            print()
            return torch.sigmoid
        else:
            self.scaling = torch.ones(len(self.action_low))
            self.offset = torch.zeros(len(self.action_low))
            for i in range(len(self.action_low)):
                low = self.action_low[i]
                high = self.action_high[i]
                if not (low and high):
                    if low == -math.inf or high == math.inf:
                        relu_idxs.append(i)
                        # self.relu_mask[i] = 1.0
                    else:
                        sigmoid_idxs.append(i)
                        # self.sigmoid_mask[i] = 1.0
                        self.scaling[i] = high + low
                elif low == high * -1:
                    if low != -math.inf:
                        tanh_idxs.append(i)
                        # self.tanh_mask[i] = 1.0
                        self.scaling[i] = high
                else:
                    self.offset[i] = (high - low) / 2
                    self.scaling[i] = high - offset[i]
                    tanh_idxs.append(i)
            num_linear_actions = len(self.scaling) - len(tanh_idxs) - len(relu_idxs) - len(sigmoid_idxs)
            print("Actor has ", len(relu_idxs), " ReLU, ", len(tanh_idxs), " tanh, ", len(sigmoid_idxs),
                  " sigmoid, and ", num_linear_actions, " linear actions.")
            print("Action Scaling: ", self.scaling)
            print("Action Offset: ", self.offset)
            print()

        self.tanh_idxs = tanh_idxs
        self.relu_idxs = relu_idxs
        self.sigmoid_idxs = sigmoid_idxs

        return self.output_function_continuous

    def optimize(self, transitions, policy_name=""):
        # Only for debugging:
        # torch.autograd.set_detect_anomaly(True)

        state_batch = transitions["state"]
        state_features = transitions["state_features"]
        action_batch = transitions["action"]

        # TODO: also do it for SPG?
        if self.discrete_env and self.use_CACLA_V or self.use_CACLA_Q:
            transformed_action_batch = torch.argmax(action_batch, dim=1)

        # Calculate current actions for state_batch:
        actions_current_state = self(state_features)
        better_actions_current_state = actions_current_state.detach().clone()
        # if self.discrete_env:
        #    action_batch = one_hot_encode(action_batch, self.num_actions)
        sample_weights = None

        if self.use_CACLA_V:
            # Check which actions have a pos TDE
            pos_TDE_mask = (self.V.TDE < 0).squeeze()
            output = actions_current_state[pos_TDE_mask]

            if self.discrete_env:
                target = transformed_action_batch[pos_TDE_mask].view(output.shape[0])
            else:
                target = action_batch[pos_TDE_mask]

            # TODO: investigate why the multiplication by minus one is necessary for sample weights... seems to be for the V.TDE < 0 check. Using all actions with sample weights = TDE also works, but worse in cartpole
            # TODO: also investigate whether scaling by TDE can be beneficial. Both works at least with V.TDE < 0
            sample_weights = -1 * torch.ones(target.shape)  # .unsqueeze(1) #
            # sample_weights = self.V.TDE[pos_TDE_mask].view(output.shape)

            # print(output)
            # print(target)
            # print()
            # print(sample_weights)

        if self.use_CACLA_Q:
            # Calculate mask of pos expected Q minus Q(s, mu(s))
            # action_TDE = self.Q.expectations_next_state - self.Q(state_features, actions_current_state).detach()
            pos_TDE_mask = (self.Q.TDE < 0).squeeze()

            output = actions_current_state[pos_TDE_mask]

            if self.discrete_env:
                target = transformed_action_batch[pos_TDE_mask].view(output.shape[0])
            else:
                target = action_batch[pos_TDE_mask]

            # sample_weights = -1 * torch.ones(output.shape[0])
            sample_weights = self.Q.TDE[pos_TDE_mask].view(target.shape[0])

        # TODO: implement CACLA+Var

        # TODO - Idea: When using QV, possibly reinforce actions only if Q and V net agree (first check how often they disagree and in which cases)
        if self.use_DDPG:
            # Dirty and fast way (still does not work yet... :-( )
            q_vals = -self.Q(self.Q.F_sa(state_features, actions_current_state)).mean()
            self.optimizer.zero_grad()
            q_vals.backward()
            self.optimizer.step()
            return q_vals.detach()

            # 1. calculate derivative of Q towards actions 2. Reinforce towards actions plus gradients
            actions_current_state_detached = Variable(actions_current_state.detach(), requires_grad=True)
            state_action_features_current_policy = self.Q.F_sa(state_features, actions_current_state_detached)
            q_vals = self.Q(state_action_features_current_policy)
            actor_loss = q_vals.mean() * -1
            actor_loss.backward(retain_graph=True)  # retain necessary? I thnk so
            gradients = actions_current_state_detached.grad
            self.log.add("DDPG Action Gradient", gradients.mean(), skip_steps=self.log_freq)

            # Normalize gradients:
            # gradients = self.normalize_gradients(gradients)
            # TODO: maybe normalize within the actor optimizer...?
            # TODO Normalize over batch, then scale by inverse TDE (risky thing:what about very small TDEs?
            output = actions_current_state
            target = (actions_current_state.detach().clone() + gradients)

            # Clip actions
            target = torch.max(torch.min(target, self.action_high), self.action_low)

            # sample_weights = torch.ones(target.shape[0]).unsqueeze(1) / abs(self.Q.TDE)

            # print(sample_weights)
            # print(output)
            # print(gradients)

        if self.use_SPG:
            # Calculate mask of Q(s,a) minus Q(s, mu(s))
            with torch.no_grad():
                # TODO: either convert to max policy using the following line or pass raw output to F_sa and don't one-hot encode
                state_features_target = self.F_s.target_net(state_batch)
                actions_target_net = self.target_net(state_features_target)
                # print("Actions target net: ", actions_target_net)
                # if self.discrete_env:
                #    actions_current_policy = actions_target_net.argmax(1).unsqueeze(1)
                state_action_features_sampled_actions = self.Q.F_sa.target_net(state_features_target, action_batch)
                state_action_features_current_policy = self.Q.F_sa.target_net(state_features_target,
                                                                              actions_current_policy,
                                                                              apply_one_hot_encoding=False)
                Q_val_sampled_actions = self.Q.target_net(state_action_features_sampled_actions)
                Q_val_current_policy = self.Q.target_net(state_action_features_current_policy)
                action_TDE = Q_val_sampled_actions - Q_val_current_policy
                # print("action TDE: ", action_TDE)
            pos_TDE_mask = (action_TDE > 0).squeeze()

            # better_actions_current_state[pos_TDE_mask] = action_batch[pos_TDE_mask]

            output = actions_current_state[pos_TDE_mask]
            # print("Output: ", output)
            target = action_batch[pos_TDE_mask].view(output.shape[0])
            # print("Target: ", target)
            sample_weights = action_TDE[pos_TDE_mask]

            # 1. Get batch_actions and batch_best_actions (implement best_actions everywhere)
            # 2. Calculate eval of current action
            # 3. Compare batch_action and batch_best_actions to evals of current actions
            # 4. Sample around best action with Gaussian noise until better action is found, then sample around this
            # 5. Reinforce towards best actions
        if self.use_GISPG:
            # Gradient Informed SPG
            # Option one:
            # Do SPG, but for every action apply DDPG to get the DDPG action and check if it is better than the non-
            # DDPG action.
            # Option two:
            # Do SPG, but do not sample with Gaussian noise. Instead always walk towards gradient of best action,
            #  with magnitude that decreases over one sampling period
            #
            pass

        # self.optimize_net(actions_current_state, better_actions_current_state, self.optimizer, "actor")

        # print("output", output)
        # print(target)
        # if not self.discrete_env:
        #    target = target.unsqueeze(1)

        if len(output) > 0:
            # Train actor towards better actions (loss = better - current)
            error, loss = self.optimize_net(output, target, self.optimizer, sample_weights=sample_weights)
            if not self.optimize_centrally:
                self.log_nn_data(policy_name)

        else:
            error = 0
            loss = 0
            # TODO: log for CACLA Q and CACLA V and SPG on how many actions per batch is trained
            # print("No Training for Actor...")

        if self.use_CACLA_V or self.use_CACLA_Q or self.use_SPG:
            self.log.add("Actor_actual_train_batch_size", len(output), skip_steps=self.log_freq)

        return error, loss

    def log_nn_data(self, name=""):
        self.log_layer_data(self.layers, "Actor", extra_name=name)
        if self.F_s is not None:
            self.F_s.log_nn_data("_Actor_" + name)


    def recreate_self(self):
        return self.__class__(self.F_s, self.env, self.log, self.device, self.hyperparameters, is_target_net=True)

    def get_updateable_params(self):
        return self.layers.parameters()

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.layers, path + "actor.pth")


    def load(self, path):
        loaded_model= torch.load(path + "actor.pth")
        self.layers = loaded_model

