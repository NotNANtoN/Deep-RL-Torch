import minerl
import gym

gym.logger.set_level(40)
from itertools import count


import torch
from tqdm import tqdm
import time

from networks import *
from policies import Agent
from env_wrappers import FrameSkip
from util import display_top_memory_users



class Trainer:
    def __init__(self, env_name, hyperparameters, log=True, tb_comment="", log_NNs=False):
        # Init logging:
        self.path = os.getcwd()
        self.log = Log(self.path + '/tb_log', log, tb_comment, log_NNs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_val = hyperparameters["matrix_max_val"]
        self.rgb2gray = hyperparameters["rgb_to_gray"]

        # Init env:
        self.env_name = env_name
        self.env = gym.make(env_name)
        # Apply Wrappers:
        if hyperparameters["frameskip"] > 1:
            self.env = FrameSkip(self.env, skip=hyperparameters["frameskip"])
        if hyperparameters["convert_2_torch_wrapper"]:
            wrapper = hyperparameters["convert_2_torch_wrapper"]
            self.env = wrapper(self.env, self.device, self.max_val, self.rgb2gray)

        if hyperparameters["action_wrapper"]:
            always_keys = hyperparameters["always_keys"]
            exclude_keys = hyperparameters["exclude_keys"]
            action_wrapper = hyperparameters["action_wrapper"]

            self.env = action_wrapper(self.env, always_keys=always_keys, exclude_keys=exclude_keys)
            print("Num actions after wrapping: ", self.env.action_space.n)

        if hyperparameters["max_episode_steps"] > 0:
            self.max_steps_per_episode = hyperparameters["max_episode_steps"]
        elif self.env_name == "LunarLander-v2":
            self.max_steps_per_episode = 1000
        elif self.env_name == "CartPole-v1":
            self.max_steps_per_episode = 500
        elif self.env_name == "CartPole-v0":
            self.max_steps_per_episode = 200
        elif self.env_name == "Pendulum-v0":
            self.max_steps_per_episode = 200
        else:
            self.max_steps_per_episode = 0
        self.reward_std = hyperparameters["reward_std"]
        self.use_exp_rep = hyperparameters["use_exp_rep"]
        self.n_initial_random_actions = hyperparameters["n_initial_random_actions"]
        self.updates_per_step = hyperparameters["network_updates_per_step"]

        # copied from Old class:
        self.normalize_observations = hyperparameters["normalize_obs"]
        self.freeze_normalizer = hyperparameters["freeze_normalize_after_initial"]
        # if self.normalize_observations:
        #    self.normalizer = Normalizer(self.state_len)
        # else:
        #    self.normalizer = None

        # Show proper tqdm progress if possible:
        self.disable_tqdm = hyperparameters["tqdm"] == 0
        if self.max_steps_per_episode:
            self.tqdm_episode_len = self.max_steps_per_episode
        #elif self.env._max_episode_steps:
        #    self.tqdm_episode_len = env._max_episode_steps
        else:
            self.tqdm_episode_len = None

        self.use_expert_data = hyperparameters["use_expert_data"]
        self.pretrain_percentage = hyperparameters["pretrain_percentage"]
        self.do_pretrain = self.pretrain_percentage > 0
        self.pretrain_weight_decay = hyperparameters["pretrain_weight_decay"]

        # Load expert data:
        if self.use_expert_data:
            expert_data = self.load_expert_data()
            num_expert_samples = len(expert_data)
            hyperparameters["num_expert_samples"] = num_expert_samples
        else:
            hyperparameters["num_expert_samples"] = 0
        # Init Policy:
        self.policy = Agent(self.env, self.device, self.log, hyperparameters)
        # Load expert data into policy buffer:
        if self.use_expert_data:
            self.move_expert_data_into_buffer(expert_data)


    def reset(self):
        self.episode_durations = []
        self.policy.reset()

    def optimize(self):
        self.policy.optimize()

    def modify_env_reward(self, reward):
        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        if self.reward_std:
            reward += torch.tensor(np.random.normal(0, self.reward_std))
        return reward


    def move_expert_data_into_buffer(self, data):
        print("Moving Expert Data into the replay buffer...")
        pbar = tqdm(total=len(data), disable=self.disable_tqdm)
        while len(data) > 0:
            pbar.update(1)
            state, raw_action, reward, next_state, done = data[0]

            # TODO: transform state and next_state into the same obs space as the original env (e.g. convert treechop to obtainDiamond etc)
            for key in raw_action:
                if key == "camera":
                    raw_action[key] = tuple(raw_action["camera"][0])
                else:
                    raw_action[key] = int(raw_action[key])
            raw_action["sneak"] = 0
            raw_action["sprint"] = 1
            if raw_action["right"] and raw_action["left"]:
                raw_action["right"] = 0
                raw_action["left"] = 0
            if raw_action["forward"] and raw_action["back"]:
                raw_action["forward"] = 0
                raw_action["back"] = 0
            if ("place" in raw_action and raw_action["place"]) or\
                    ("craft" in raw_action and raw_action["craft"]) or\
                    ("nearbyCraft" in raw_action and raw_action["nearbyCraft"]) or\
                    ("nearbySmelt" in raw_action and raw_action["nearbySmelt"]) or\
                    ("equip" in raw_action and raw_action["equip"]):
                raw_action["left"] = 0
                raw_action["right"] = 0
                raw_action["forward"] = 0
                raw_action["back"] = 0
                raw_action["camera"] = (0, 0)
                raw_action["jump"] = 0
                raw_action["attack"] = 0

            # TODO: move as many of those checks above into the dict2idx function!
            action = torch.zeros(1, self.env.action_space.n, device=self.device, dtype=torch.float)
            action_idx = self.env.dict2idx(raw_action)
            action[0][action_idx] = 1.0
            reward = self.modify_env_reward(reward)[0]

            state = self.env.observation(state, expert_data=True)
            next_state = self.env.observation(next_state, expert_data=True)
            # To initialize the normalizer:
            if self.normalize_observations:
                self.policy.F_s(state)

            self.policy.remember(state, action, next_state, reward, done)
            # Delete data from data list when processed to save memory
            del data[0]
        # TODO: maybe preprocess the data somehow. We could train during the gading for example. Or we calculate eligibility traces upon seeing a done.
        pbar.close()


    def use_data_pipeline_MineRL(self, pipeline):
        return [sample for sample in tqdm(pipeline.sarsd_iter(num_epochs=1, max_sequence_len=1), disable=self.disable_tqdm)]

        data = []
        for sample in tqdm(pipeline.sarsd_iter(num_epochs=1, max_sequence_len=1), disable=self.disable_tqdm):
            data.append(sample)

            #if len(data) > 10000:
            #    break
        return data

        # TODO: apply frameskip here! (if used)

    def load_expert_data_MineRL(self):
        print("Loading expert MineRL data...")

        #env_name_data = 'MineRLObtainDiamond-v0'
        env_name_data = self.env_name
        data_pipeline = minerl.data.make(
            env_name_data,
            data_dir='data')
        data = self.use_data_pipeline_MineRL(data_pipeline)

        return data

        # TODO: store data in a file depending on mineRL version to have quicker loading

        # TODO: use data from all envs (except navigate)

    def load_expert_data(self):
        if "MineRL" in self.env_name:
            return self.load_expert_data_MineRL()
        else:
            raise NotImplementedError("No expert data loading for this environment is implemented at the moment.")

    def pretrain(self, steps, hours, start_time):
        print("Pretraining on expert data...")
        # TODO: implement supervised leanring according to DQfD

        # TODO: implement weight decay according to DQfD
        #self.policy.set_weight_decay(self.pretrain_weight_decay)
        if steps:
            for step in tqdm(range(steps), disable=self.disable_tqdm):
                # Perform one step of the optimization
                self.policy.optimize()

                # Update the target network
                self.policy.update_targets(step)
        elif time:
            # TODO: add some tqdm option here
            for t in count():
                # Perform one step of the optimization
                self.policy.optimize()

                # Update the target network
                self.policy.update_targets(t)

                if (time.time() - start_time) / 360 > hours:
                    break


        #self.policy.set_weight_decay(0)

    def fill_replay_buffer(self, n_actions):
        print("Filling Replay Buffer....")
        state = self.env.reset()
        if not isinstance(state, dict):
            state = torch.tensor([state], device=self.device).float()

        # Fill exp replay buffer so that we can start training immediately:
        for _ in tqdm(range(n_actions), disable=self.disable_tqdm):

            # To initialize the normalizer:
            if self.normalize_observations:
                self.policy.F_s(state)

            action, next_state, reward, done = self._act(self.env, state, store_in_exp_rep=True, render=False,
                                                         explore=True, fully_random=True)

            state = next_state
            if done:
                state = self.env.reset()
                if not isinstance(state, dict):
                    state = torch.tensor([state], device=self.device).float()
        print("Done with filling replay buffer.")
        print()


    def _act(self, env, state, explore=True, render=False, store_in_exp_rep=True, fully_random=False):
        # Select an action
        if explore:
            # Raw actions are the logits for the actions. Useful for e.g. DDPG training in discrete envs.
            action, raw_action = self.policy.explore(state, fully_random=fully_random)
        else:
            action, raw_action = self.policy.exploit(state)

        self.log.add("ActionIdx", action, make_distribution=True, skip_steps=10000)

        # Apply the action:
        next_state, reward, done, _ = env.step(action)
        # Add possible noise to the reward:
        reward = self.modify_env_reward(reward)
        # Define next state in case it is terminal:
        if done:
            next_state = None
        else:
            if not isinstance(next_state, dict):
                next_state = torch.tensor([next_state], device=self.device).float()
            #next_state = torch.tensor([next_state], device=self.device).float()
        # Store the transition in memory
        if self.use_exp_rep and store_in_exp_rep:
            self.policy.remember(state, raw_action, next_state, reward, done)
        # Calculate TDE for debugging purposes:
        # TODO: implement logging of predicted Q value and TDE
        #TDE = self.policy.calculate_TDE(state, raw_action, next_state, reward, done)
        #self.log.add("TDE_live", TDE.item())
        # Render:
        if render:
            self.env.render()

        return action, next_state, reward, done

    def _act_in_test_env(self, test_env, test_state, test_episode_rewards):
        _, next_state, reward, done = self._act(test_env, test_state, explore=False, store_in_exp_rep=False, render=False)

        test_episode_rewards.append(reward)
        self.log.add("Test_Env Reward", np.sum(test_episode_rewards))
        if done or (self.max_steps_per_episode > 0 and len(test_episode_rewards) >= self.max_steps_per_episode):
            next_state = test_env.reset()
            next_state = torch.tensor([next_state], device=self.device).float()
            test_episode_rewards.clear()

        return next_state

    def _display_debug_info(self, i_episode, steps_done):
        episode_return = self.log.get_episodic("Return")
        optimize_time = self.log.get_episodic("Optimize_Time")
        non_optimize_time = self.log.get_episodic("Non-Optimize_Time")
        print("#Episode ", i_episode)
        print("#Steps: ", steps_done)
        print(" Return:", episode_return[0])
        print(" Opt-time: ", round(np.mean(optimize_time), 4), "s")
        print(" Non-Opt-time: ", round(np.mean(non_optimize_time), 4), "s")
        if i_episode % 10 == 0:
            pass
            # TODO: do an extensive test in test_env every N steps
            # Not needed anymore, because he have tensorboard now
            #plot_rewards(rewards)
            #plot_rewards(self.log.storage["Test_Env Reward"], "Test_Env Reward")
            #plot_rewards(self.log.storage["Return"], "Return", xlabel="Episodes")
        print()

    def run(self, n_hours=0, n_episodes=0, n_steps=0, verbose=False, render=False, on_server=True):
        assert (bool(n_steps) ^ bool(n_episodes) ^ bool(n_hours))

        steps_done = 0
        i_episode = 0
        start_time = time.time()
        # Fill replay buffer with random actions:
        if not self.use_expert_data:
            self.fill_replay_buffer(n_actions=self.n_initial_random_actions)

        if self.freeze_normalizer:
            print("Freeze observation Normalizer.")
            self.policy.freeze_normalizers()

        if self.use_expert_data and self.do_pretrain:
            pretrain_steps = int(self.pretrain_percentage * n_steps)
            pretrain_episodes = int(self.pretrain_percentage * n_episodes)
            pretrain_time = int(self.pretrain_percentage * n_hours)
            if pretrain_episodes:
                pretrain_steps = pretrain_episodes * 5000
                # TODO: insetad of hardcoding 5000 get an expective episode duration from somewhere

            self.pretrain(pretrain_steps, pretrain_time, start_time)
            steps_done += pretrain_steps
            i_episode += pretrain_episodes

        # Initialize test environment:
        # test_env = gym.make(self.env_name).unwrapped
        # test_state = test_env.reset()
        # test_state = torch.tensor([test_state], device=self.device).float()

        # Do the actual training:
        time_after_optimize = None
        pbar = tqdm(total=n_steps, desc="Total Training", disable=self.disable_tqdm)
        while (n_steps and steps_done < n_steps) or (n_episodes and i_episode < n_episodes) or\
                (n_hours and (time.time() - start_time) / 360 < n_hours):
            i_episode += 1
            # Initialize the environment and state
            state = self.env.reset()
            if not isinstance(state, dict):
                state = torch.tensor([state], device=self.device).float()

            for t in tqdm(count(), desc="Episode Progress", total=self.tqdm_episode_len, disable=self.disable_tqdm):
                steps_done += 1
                if not verbose and not on_server:
                    print("Episode loading:  " + str(round(self.steps_done / n_steps * 100, 2)) + "%")  # , end="\r")

                # Act in exploratory env:
                action, next_state, reward, done = self._act(self.env, state, render=render, store_in_exp_rep=True,
                                                             explore=True)

                # Act in test env (no exploration in that env):
                # TODO: replace the one-step in test env per step in train env by a proper evaluation every N steps (e.g. every 100/1000 training steps, test for 10 episodes in test env and record mean)
                # test_state = self._act_in_test_env(test_env, test_state)

                # Move to the next state
                state = next_state

                # Reduce epsilon and other exploratory values:
                self.policy.decay_exploration(steps_done)

                time_before_optimize = time.time()
                # Log time between optimizations:
                non_optimize_time = 0
                if time_after_optimize is not None:
                    non_optimize_time = time_before_optimize - time_after_optimize
                self.log.add("Non-Optimize_Time", non_optimize_time, skip_steps=10000, store_episodic=True)

                num_updates = int(self.updates_per_step) if self.updates_per_step >= 1\
                                                    else n_steps % int(1 / self.updates_per_step) == 0
                for _ in range(num_updates):
                    # Perform one step of the optimization (on the target network)
                    self.policy.optimize()

                    # Update the target network
                    self.policy.update_targets(steps_done)
                time_after_optimize = time.time()

                # Log reward and time:
                self.log.add("Train_Reward", reward.item(), skip_steps=10000, store_episodic=True)
                self.log.add("Train_Sum_Episode_Reward", np.sum(self.log.get_episodic("Train_Reward")), skip_steps=10000)
                self.log.add("Optimize_Time", time_after_optimize - time_before_optimize, skip_steps=10000,
                             store_episodic=True)

                if render:
                    self.env.render()

                self.log.step()
                if done or (self.max_steps_per_episode > 0 and t >= self.max_steps_per_episode) \
                        or (n_steps and steps_done >= n_steps) or (n_episodes and i_episode >= n_episodes) or\
                        (n_hours and (time.time() - start_time) / 360 >= n_hours):
                    pbar.update(t)

                    self.log.add("Return", np.sum(self.log.get_episodic("Train_Reward")), steps=i_episode, store_episodic=True)
                    if verbose:
                        self._display_debug_info(i_episode, steps_done)
                    self.log.flush_episodic()
                    break

        print('Done.')
        self.env.close()
        pbar.close()
        return i_episode



class TrainerOld(object):
    def __init__(self, env_name, device):
        num_initial_states = 5
        self.initial_states = torch.tensor([self.env.reset() for _ in range(num_initial_states)], device=self.device,
                                           dtype=torch.float)


    def initialize_workers(self):
        self.workers = [{"env": gym.make(self.env_name).unwrapped, "episode length": 0} for i in range(self.batch_size)]
        for worker in self.workers:
            worker["state"] = worker["env"].reset()
            worker["state"] = torch.tensor([worker["state"]], device=self.device).float()

    def collect_experiences(self):
        # Do one step with each worker and return transition batch
        transition_list = []
        dones = 0
        for idx in range(len(self.workers)):
            worker = self.workers[idx]
            worker_env = worker["env"]
            worker_state = worker["state"]
            action = self.select_action(worker_state, self.epsilon)
            worker["episode length"] += 1
            next_state, reward, done, _ = worker_env.step(action.item())

            if done:
                next_state = None
            else:
                next_state = torch.tensor([next_state], device=self.device).float()
            reward = torch.tensor([reward], device=self.device, dtype=torch.float)
            if self.TDEC_ENABLED:
                TDE = self.calculateTDE(self.workers[idx]["state"], action, next_state, reward)
            else:
                TDE = None

            trans = Transition(self.workers[idx]["state"], action, next_state, reward, TDE)
            transition_list.append(trans)

            if done or worker["episode length"] > self.max_steps_per_episode > 0:
                dones += 1
                worker["episode length"] = 0
                worker["state"] = worker["env"].reset()
                worker["state"] = torch.tensor([worker["state"]], device=self.device).float()
            else:
                worker["state"] = next_state

        return transition_list

    def calculate_initial_state_val(self):
        with torch.no_grad():
            predictions_inital_states = self.target_net(self.initial_states).view(-1, self.num_actions,
                                                                                  self.num_Q_output_slots)
            initial_state_value = torch.mean(predictions_inital_states[:, :, 1]).item()
        return initial_state_value

    # should not be needed anymore: check if it is used anywhere before deleting
    # def getActionIdxs(self, action_batch):
    #    return torch.cat([action_batch * self.num_Q_output_slots + i for i in range(self.num_Q_output_slots)], dim=1)