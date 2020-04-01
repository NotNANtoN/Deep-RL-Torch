import collections
import time
import itertools
import os
import psutil

import numpy as np
import minerl
import aigar
import gym

gym.logger.set_level(40)
import torch
from tqdm import tqdm
from pytorch_memlab import profile

from .agent import Agent
from .env_wrappers import FrameSkip, FrameStack
from .util import display_top_memory_users, apply_rec_to_dict
from .log import Log


def calc_train_fraction(total_steps, steps_done, n_episodes, i_episode, n_hours, start_time):
    if total_steps:
        fraction = steps_done / total_steps
    elif n_episodes:
        fraction = i_episode / n_episodes
    else:
        time_diff = (time.time() - start_time) / 360
        fraction = time_diff / n_hours
    return fraction


class Trainer:
    def __init__(self, env_name, hyperparameters, log=True, tb_comment="", verbose=False):
        # Init logging:
        self.path = os.getcwd()
        self.log = Log(self.path + '/tb_log', log, tb_comment, env_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rgb2gray = hyperparameters["rgb_to_gray"]
        hyperparameters["verbose"] = verbose
        self.hyperparameters = hyperparameters
        self.verbose = verbose
        # Logging:
        self.psutil_process = psutil.Process()

        # Cuda there?
        self.cuda = torch.cuda.is_available()

        # Evaluation params:
        self.eval_rounds = hyperparameters["eval_rounds"]
        self.eval_percentage = hyperparameters["eval_percentage"]
        self.stored_percentage = 0

        # Init env:
        self.env_name = env_name
        self.env = self.create_env(hyperparameters)
        if self.eval_rounds > 0:
            self.test_env = self.create_env(hyperparameters)

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

        # Exploration params:
        self.n_initial_random_actions = hyperparameters["initial_steps"]
        self.explore_until_reward = hyperparameters["explore_until_reward"]

        self.normalize_observations = hyperparameters["normalize_obs"]
        self.freeze_normalizer = hyperparameters["freeze_normalize_after_initial"]
        self.log_freq = hyperparameters["log_freq"]


        # Show proper tqdm progress if possible:
        self.disable_tqdm = hyperparameters["tqdm"] == 0
        if self.max_steps_per_episode:
            self.tqdm_episode_len = self.max_steps_per_episode
        # elif self.env._max_episode_steps:
        #    self.tqdm_episode_len = env._max_episode_steps
        else:
            self.tqdm_episode_len = None
        
        # Pretrain hyperparams:
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

        # Agent params:
        
        # Init Agent:
        self.agent = Agent(self.env, self.device, self.log, hyperparameters)
        if hyperparameters["load"]:
            self.agent.load()
        # Load expert data into policy buffer:
        if self.use_expert_data:
            self.move_expert_data_into_buffer(expert_data)

    def create_env(self, hyperparameters):
        # Init env:
        env = gym.make(self.env_name)
        # Apply Wrappers:
        if hyperparameters["frameskip"] > 1:
            env = FrameSkip(env, skip=hyperparameters["frameskip"])
        if hyperparameters["convert_2_torch_wrapper"]:
            wrapper = hyperparameters["convert_2_torch_wrapper"]
            env = wrapper(env, self.rgb2gray)
        if hyperparameters["frame_stack"] > 1 and hyperparameters["use_list"]:
            env = FrameStack(env, hyperparameters["frame_stack"], stack_dim=hyperparameters["stack_dim"],
                             store_stacked=hyperparameters["store_stacked"])
        if hyperparameters["action_wrapper"]:
            always_keys = hyperparameters["always_keys"]
            exclude_keys = hyperparameters["exclude_keys"]
            action_wrapper = hyperparameters["action_wrapper"]

            env = action_wrapper(env, always_keys=always_keys, exclude_keys=exclude_keys, env_name=self.env_name)
        return env

    def reset(self):
        raise NotImplementedError
        # self.episode_durations = []
        # self.agent.reset()

    def optimize(self):
        self.agent.optimize()

    def modify_env_reward(self, reward):
        reward = torch.tensor([reward], dtype=torch.float)
        if self.reward_std:
            reward += torch.tensor(np.random.normal(0, self.reward_std))
        return reward

    def move_expert_data_into_buffer(self, data):
        if self.verbose:
            print("Moving Expert Data into the replay buffer...")
        pbar = tqdm(total=len(data), disable=self.disable_tqdm)
        source = "expert_data"
        while len(data) > 0:
            pbar.update(1)
            state, action, reward, next_state, done = data[0]

            # To initialize the normalizer:
            if self.normalize_observations:
                self.agent.observe(state, source, done)
                # TODO: normalize actions too # self.policy.F_sa.observe(action)

            self.agent.remember(state, action, reward, done, filling_buffer=True)
            # Delete data from data list when processed to save memory
            del data[0]
        pbar.close()

    def use_data_pipeline_MineRL(self, pipeline):
        # return [sample for sample in tqdm(pipeline.sarsd_iter(num_epochs=1, max_sequence_len=1), disable=self.disable_tqdm)]

        data = []
        for state, raw_action, reward, next_state, done in tqdm(pipeline.sarsd_iter(num_epochs=1, max_sequence_len=1),
                                                                disable=self.disable_tqdm):

            # TODO: think about filtering out the end of an episode that did not lead to a reward. So if [0,0,1,2,0, 0,1,0 0 ,0] are the episode rewards, it should be cut to [0, 0, 1, 2, 0, 0, 1]
            # TODO: this could be good because it filters unsuccesfull runs... but maybe it is good to have some unsucessfull runs if we assume that the player handled the difficulties in a relatively good way

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
            if ("place" in raw_action and raw_action["place"]) or \
                    ("craft" in raw_action and raw_action["craft"]) or \
                    ("nearbyCraft" in raw_action and raw_action["nearbyCraft"]) or \
                    ("nearbySmelt" in raw_action and raw_action["nearbySmelt"]) or \
                    ("equip" in raw_action and raw_action["equip"]):
                raw_action["left"] = 0
                raw_action["right"] = 0
                raw_action["forward"] = 0
                raw_action["back"] = 0
                raw_action["camera"] = (0, 0)
                raw_action["jump"] = 0
                raw_action["attack"] = 0

            # TODO: move as many of those checks above into the dict2idx function!
            action = torch.zeros(1, self.env.action_space.n, dtype=torch.float)

            action_idx = self.env.dict2idx(raw_action)
            action[0][action_idx] = 1.0
            reward = self.modify_env_reward(reward)[0]

            state = self.env.observation(state, expert_data=True)
            next_state = self.env.observation(next_state, expert_data=True)

            sample = (state, action, reward, next_state, done)
            data.append(sample)

        return data

        # TODO: apply frameskip here! (if used)

    def load_expert_data_MineRL(self):
        if self.verbose:
            print("Loading expert MineRL data...")
        folder = 'data'
        if not os.path.exists(folder):
            os.mkdir(folder)
        minerl.data.download(folder)
        # ver_or_download_data(self.env_name)
        # env_name_data = 'MineRLObtainDiamond-v0'
        data_pipeline = minerl.data.make(
                self.env_name,
                data_dir='data')
        data = self.use_data_pipeline_MineRL(data_pipeline)

        return data

        # TODO: possibly store data in a file depending on mineRL version to have quicker loading

        # TODO: use data from all envs (except navigate)!!!

    def load_expert_data(self):
        if "MineRL" in self.env_name:
            return self.load_expert_data_MineRL()
        else:
            raise NotImplementedError("No expert data loading for this environment is implemented at the moment.")

    def pretrain(self, steps, hours, start_time):
        if self.verbose:
            print("Pretraining on expert data...")
        # TODO: implement supervised leanring according to DQfD

        # TODO: implement weight decay according to DQfD
        # self.policy.set_weight_decay(self.pretrain_weight_decay)
        if steps:
            for step in tqdm(range(steps), disable=self.disable_tqdm):
                # Perform one step of the optimization
                self.agent.optimize()

                train_fraction = step / steps
                # Update the target network
                self.agent.update_targets(step, train_fraction)

                self.log.step()
        elif time:
            pbar = tqdm(disable=self.disable_tqdm, total=hours)
            for step in itertools.count():
                pbar.update((time.time() - start_time) / 360)

                # Perform one step of the optimization
                self.agent.optimize()

                train_fraction = (time.time() - start_time) / 360 / hours
                # Update the target network
                self.agent.update_targets(step, train_fraction)

                self.log.step()

                if (time.time() - start_time) / 360 >= hours:
                    break
        if self.verbose:
            print("Done Pretraining.")
        return step

        # self.policy.set_weight_decay(0)

    def fill_replay_buffer(self, total_steps):
        source = "filling"
        assert total_steps > 0
        if self.verbose:
            print("Filling Replay Buffer....")
        state = self.env.reset()
        rewards = collections.defaultdict(int)

        # Fill exp replay buffer so that we can start training immediately:
        pbar = tqdm(disable=self.disable_tqdm)
        i = 0
        done_count = 0
        do_break = False
        while True:
            pbar.update(1)

            action, next_state, reward, done = self._act(self.env, state, source, store_in_exp_rep=True, render=False,
                                                         explore=True, filling_buffer=True)
            # To initialize the normalizer:
            if self.normalize_observations:
                self.agent.observe(state, source)
                # TODO: normalize (observe) actions too for actor critic

            state = next_state
            if done:
                done_count += 1
                state = self.env.reset()

            if self.explore_until_reward and not do_break:
                if isinstance(reward, torch.FloatTensor) or isinstance(reward, torch.LongTensor):
                    reward = reward.item()
                rewards[reward] += 1
                if len(rewards) > 1:
                    if self.verbose:
                        print("Encountered a new reward value. Rewards; ", rewards)
                    do_break = True
            else:
                i += 1
                if i >= total_steps:
                    do_break = True

            # For eligibility traces we need to complete at least one episode to properly start training
            if do_break:
                break

        if self.verbose:
            print("Done with filling replay buffer.")
            print()

    def _act(self, env, state, source, explore=True, render=False, store_in_exp_rep=True, filling_buffer=False):
        # Select an action
        if explore:
            # Raw actions are the logits for the actions. Useful for e.g. DDPG training in discrete envs.
            action, raw_action = self.agent.explore(state, source, fully_random=filling_buffer)
        else:
            action, raw_action = self.agent.exploit(state, source)

        if not filling_buffer and self.log.is_available("ActionIdx", factor=10, reset=False):
            self.log.add("ActionIdx", action, make_distr=True, distr_steps=self.log.mean_ep_len)

        # Apply the action:
        next_obs, reward, done, _ = env.step(action)
        # Add possible noise to the reward:
        if explore:
            reward = self.modify_env_reward(reward)
        # Count ep len:
        self.log.count_eps_step(source)
        # Define next state in case it is terminal:
        if done:
            next_obs = None
            self.agent.clean_state(source)
            self.log.count_eps(source)
        # Store the transition in memory:
        if self.use_exp_rep and store_in_exp_rep:
            self.agent.remember(state, raw_action, reward, done, filling_buffer=filling_buffer)
        # Calculate TDE for debugging purposes:
        # TODO: implement logging of TDE
        # tde = self.policy.calculate_Q_and_TDE(state, raw_action, next_state, reward, done)
        # self.log.add("TDE_live", tde)
        # Render:
        if render:
            self.env.render()

        return action, next_obs, reward, done

    def evaluate_model(self):
        reward_sum = 0
        source = "Evaluation"
        for i in range(self.eval_rounds):
            current_state = self.test_env.reset()
            for t in itertools.count():
                action, next_obs, reward, done = self._act(self.test_env, current_state, source, explore=True, store_in_exp_rep=False)
                reward_sum += reward
                current_state = next_obs
                if done:
                    break
        reward_mean = reward_sum / self.eval_rounds
        return reward_mean

    def _display_debug_info(self, i_episode, steps_done, train_fraction):
        episode_return = self.log.get_episodic("Metrics/Return")
        # sampling_time = self.log.get_episodic("Sampling_Time")
        optimize_time = self.log.get_episodic("Timings/Optimize_Time")
        non_optimize_time = self.log.get_episodic("Timings/Non-Optimize_Time")
        print("Ep:", i_episode, " Step:", steps_done, round(train_fraction * 100, 1), "%", "Ret:", episode_return[0], )

    def log_usage(self):
        if self.log.is_available("Usage", factor=10):
            cpu_ram_GB = self.psutil_process.memory_info()[0] / 1024. ** 3
            cpu_usage = self.psutil_process.cpu_percent()

            self.log.add("Usage/CPU_mem", cpu_ram_GB)
            self.log.add("Usage/CPU_usage", cpu_usage)
            if self.cuda:
                gpu_mem = torch.cuda.memory_allocated() / 8 / 1024. ** 3
                self.log.add("Usage/GPU_mem", gpu_mem)

    def run(self, n_hours=0.0, n_episodes=0, total_steps=0, verbose=False, render=False, disable_tqdm=True):
        assert (bool(total_steps) ^ bool(n_episodes) ^ bool(n_hours))
        verbose = verbose or self.verbose
        steps_done = 0
        i_episode = 0
        start_time = time.time()
        state = None
        # Fill replay buffer with random actions:
        if not self.use_expert_data:
            self.fill_replay_buffer(total_steps=self.n_initial_random_actions)
            # episodes, time = self.fill_replay_buffer(total_steps=self.n_initial_random_actions)
            # if total_steps == 0:
            #    if n_episodes != 0:
            #        total_steps = self.n_initial_random_actions / episodes * n_episodes
            #    elif n_hours != 0:
            #        total_steps = self.n_initial_random_actions / time * n_hours
        # Based on how long it took to gather data, determine how often metrics are logged:
        # TODO: only works for total_steps, not for n_episodes nor for n_hours
        self.log_steps = total_steps // self.log_freq
        self.log.set_log_steps(self.log_steps)
        # Freeze input feature normalizer after collecting a set of experiences:
        if self.freeze_normalizer:
            if verbose:
                print("Freeze observation Normalizer.")
            self.agent.freeze_normalizers()
        # Calculate the memory usage of training. Used for eligibility traces and could be used for batch_size determination:
        self.agent.calc_mem_usage()
        # Update targets to calc initial elig traces:
        self.agent.update_targets(0, 0)
        # Pretrain if expert data is available:
        if self.use_expert_data and self.do_pretrain:
            pretrain_steps = int(self.pretrain_percentage * total_steps)
            pretrain_episodes = int(self.pretrain_percentage * n_episodes)
            pretrain_time = int(self.pretrain_percentage * n_hours)
            if pretrain_episodes:
                pretrain_steps = pretrain_episodes * 5000
                # TODO: instead of hardcoding 5000 get an expected episode duration by taking the mean episode length of the data
            n_pretrain_steps_done = self.pretrain(pretrain_steps, pretrain_time, start_time)
            steps_done += n_pretrain_steps_done
            i_episode += pretrain_episodes

        # Do the actual training:
        if verbose:
            print("Start training in the env:")
        time_after_optimize = None
        if total_steps == 0:
            disable_tqdm = True
        pbar = tqdm(total=total_steps, desc="Training", disable=disable_tqdm)
        train_fraction = calc_train_fraction(total_steps, steps_done, n_episodes, i_episode, n_hours, start_time)
        source = "train"
        while train_fraction < 1:
            i_episode += 1
            # Initialize the environment and state. Do not reset
            if state is None:
                state = self.env.reset()

            for t in tqdm(itertools.count(), desc="Episode Progress", total=self.tqdm_episode_len,
                          disable=self.disable_tqdm):
                steps_done += 1

                # Act in train env:
                action, next_state, reward, done = self._act(self.env, state, source, render=render,
                                                             store_in_exp_rep=True, explore=True)
                # Evaluate agent thoroughly sometimes:
                if self.eval_rounds > 0 and (train_fraction >= self.eval_percentage + self.stored_percentage
                                             or train_fraction == 0):
                    self.stored_percentage = train_fraction
                    test_return = self.evaluate_model()
                    self.log.add("Metrics/Test Return", test_return, steps=steps_done)  # steps=train_fraction * 100)
                    if verbose:
                        print("Model performance after ", steps_done, "steps: ", test_return)
                        print()

                # Move to the next state
                state = next_state

                # Log timings:
                time_before_optimize = time.time()
                if time_after_optimize is not None:
                    non_optimize_time = time_before_optimize - time_after_optimize
                    self.log.add("Timings/Non-Optimize_Time", non_optimize_time, use_skip=True, store_episodic=True)

                # Optimize the agent (on the target network)      
                self.agent.optimize(steps_done, train_fraction)

                # Log reward and time:
                self.log.add("Metrics/Reward", reward.item(), use_skip=True, store_episodic=True)
                time_after_optimize = time.time()
                self.log.add("Timings/Optimize_Time", time_after_optimize - time_before_optimize, use_skip=True,
                             store_episodic=True)
                # Log RAM and GPU usage:
                self.log_usage()
                # Count steps in logger:
                self.log.step()

                # Check if training or the episode is done:
                episodes_done = (n_episodes and i_episode >= n_episodes)
                time_done = (n_hours and (time.time() - start_time) / 360 >= n_hours)
                if done or episodes_done or time_done:
                    episode_return = np.sum(self.log.get_episodic("Metrics/Reward"))
                    self.log.add("Metrics/Return", episode_return, store_episodic=True)
                    self.log.add("Metrics/Episode Len", t, steps=i_episode)
                    self.log.add("Metrics/Mean Ep Len", self.log.mean_ep_len)
                    if verbose:
                        self._display_debug_info(i_episode, steps_done, train_fraction)
                    self.log.flush_episodic()
                    state = None
                    break
                train_fraction = calc_train_fraction(total_steps, steps_done, n_episodes, i_episode, n_hours,
                                                     start_time)
            pbar.set_postfix(ep_return=episode_return)
            pbar.update(t)

        # Save the model:
        self.agent.update_targets(steps_done, train_fraction=1.0)
        if verbose:
            print('Done.')

        self.env.close()
        pbar.close()
        return i_episode, self.log

    def close(self):
        self.env.close()
        self.log.flush()
