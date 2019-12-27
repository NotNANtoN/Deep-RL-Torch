import collections

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
from util import display_top_memory_users, apply_rec_to_dict
from verify_or_download_data import ver_or_download_data

from pytorch_memlab import profile

def calc_train_fraction(n_steps, steps_done, n_episodes, i_episode, n_hours, start_time):
    if n_steps:
        fraction = steps_done / n_steps
    elif n_episodes:
        fraction = i_episode / n_episodes
    else:
        time_diff = (time.time() - start_time) / 360
        fraction = time_diff / n_hours
    return fraction

class Trainer:
    def __init__(self, env_name, hyperparameters, log=True, tb_comment=""):
        # Init logging:
        self.path = os.getcwd()
        self.log = Log(self.path + '/tb_log', log, tb_comment, env_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_val = hyperparameters["matrix_max_val"]
        self.rgb2gray = hyperparameters["rgb_to_gray"]
        self.pin_tensors = hyperparameters["pin_tensors"]
        self.store_on_gpu = hyperparameters["store_on_gpu"]
        self.hyperparameters = hyperparameters

        # Init env:
        self.env_name = env_name
        self.env = self.create_env(hyperparameters)

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
        self.use_elig_traces = hyperparameters["use_efficient_traces"]
        self.n_initial_random_actions = hyperparameters["n_initial_random_actions"]
        self.explore_until_reward = hyperparameters["explore_until_reward"]

        self.updates_per_step = hyperparameters["network_updates_per_step"]

        # copied from Old class:
        self.normalize_observations = hyperparameters["normalize_obs"]
        self.freeze_normalizer = hyperparameters["freeze_normalize_after_initial"]
        # if self.normalize_observations:
        #    self.normalizer = Normalizer(self.state_len)
        # else:
        #    self.normalizer = None
        self.log_freq = hyperparameters["log_freq"]

        # Evaluation params:
        self.eval_rounds = hyperparameters["eval_rounds"]
        self.eval_percentage = hyperparameters["eval_percentage"]
        self.stored_percentage = 0

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

        if hyperparameters["action_wrapper"]:
            always_keys = hyperparameters["always_keys"]
            exclude_keys = hyperparameters["exclude_keys"]
            action_wrapper = hyperparameters["action_wrapper"]

            env = action_wrapper(env, always_keys=always_keys, exclude_keys=exclude_keys, env_name=self.env_name)
        return env

    def reset(self):
        self.episode_durations = []
        self.agent.reset()

    def optimize(self):
        self.agent.optimize()

    def modify_env_reward(self, reward):
        reward = torch.tensor([reward], dtype=torch.float)
        reward = self.prep_for_GPU(reward)
        if self.reward_std:
            reward += torch.tensor(np.random.normal(0, self.reward_std))
        return reward

    def prep_for_GPU(self, tensor):
        if torch.cuda.is_available():
            if self.store_on_gpu:
                tensor = tensor.cuda()
            elif self.pin_tensors:
                tensor = tensor.cpu().pin_memory()
            else:
                tensor = tensor.cpu()
        return tensor

    def move_expert_data_into_buffer(self, data):
        print("Moving Expert Data into the replay buffer...")
        pbar = tqdm(total=len(data), disable=self.disable_tqdm)
        while len(data) > 0:
            pbar.update(1)
            state, action, reward, next_state, done = data[0]

            # To initialize the normalizer:
            if self.normalize_observations:
                self.agent.F_s.observe(state)
                # TODO: normalize actions too # self.policy.F_sa.observe(action)

            self.agent.remember(state, action, next_state, reward, done)
            # Delete data from data list when processed to save memory
            del data[0]
        pbar.close()


    def use_data_pipeline_MineRL(self, pipeline):
        #return [sample for sample in tqdm(pipeline.sarsd_iter(num_epochs=1, max_sequence_len=1), disable=self.disable_tqdm)]

        data = []
        for state, raw_action, reward, next_state, done in tqdm(pipeline.sarsd_iter(num_epochs=1, max_sequence_len=1), disable=self.disable_tqdm):

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
            action = torch.zeros(1, self.env.action_space.n, dtype=torch.float)
            action = self.prep_for_GPU(action)

            action_idx = self.env.dict2idx(raw_action)
            action[0][action_idx] = 1.0
            reward = self.modify_env_reward(reward)[0]

            state = self.env.observation(state, expert_data=True)
            next_state = self.env.observation(next_state, expert_data=True)
            state = apply_rec_to_dict(self.prep_for_GPU, state)
            next_state = apply_rec_to_dict(self.prep_for_GPU, next_state)

            sample = (state, action, reward, next_state, done)
            data.append(sample)

        return data

        # TODO: apply frameskip here! (if used)

    def load_expert_data_MineRL(self):
        print("Loading expert MineRL data...")

        ver_or_download_data(self.env_name)
        #env_name_data = 'MineRLObtainDiamond-v0'
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
        print("Pretraining on expert data...")
        # TODO: implement supervised leanring according to DQfD

        # TODO: implement weight decay according to DQfD
        #self.policy.set_weight_decay(self.pretrain_weight_decay)
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
            for t in count():
                pbar.update((time.time() - start_time) / 360)

                # Perform one step of the optimization
                self.agent.optimize()

                train_fraction = (time.time() - start_time) / 360 / hours
                # Update the target network
                self.agent.update_targets(t, train_fraction)

                self.log.step()

                if (time.time() - start_time) / 360 >= hours:
                    break
        print("Done Pretraining.")

        #self.policy.set_weight_decay(0)

    def fill_replay_buffer(self, n_steps):
        assert n_steps > 0
        print("Filling Replay Buffer....")
        state = self.env.reset()
        if not isinstance(state, dict):
            state = self.prep_for_GPU(state)
        else:
            state = apply_rec_to_dict(self.prep_for_GPU, state)
        #TODO: move these preparations to one place, add general env wrapper for easy envs

        rewards = collections.defaultdict(int)

        # Fill exp replay buffer so that we can start training immediately:
        pbar = tqdm(disable=self.disable_tqdm)
        i = 0
        done = True
        done_count = 0
        do_break = False
        while True:
            pbar.update(1)
            # To initialize the normalizer:
            if self.normalize_observations:
                self.agent.F_s.observe(state)
                # TODO: normalize (observe) actions too for actor critic
            action, next_state, reward, done = self._act(self.env, state, store_in_exp_rep=True, render=False,
                                                         explore=True, fully_random=True)

            state = next_state
            if done:
                done_count += 1
                state = self.env.reset()
                
                if not isinstance(state, dict):
                    state = self.prep_for_GPU(state)
                else:
                    state = apply_rec_to_dict(self.prep_for_GPU, state)


            if self.explore_until_reward and not do_break:
                if isinstance(reward, torch.FloatTensor) or isinstance(reward, torch.LongTensor):
                    reward = reward.item()
                rewards[reward] += 1
                if len(rewards) > 1:
                    print("Encountered a new reward value. Rewards; ", rewards)
                    do_break = True
            else:
                i += 1
                if i >= n_steps:
                    do_break = True

            # For eligibility traces we need to complete at least one episode to properly start training
            if do_break:
                if self.use_elig_traces:
                    if done_count >= 1:
                        break
                else:
                    break


        print("Done with filling replay buffer.")
        print()


    def _act(self, env, state, explore=True, render=False, store_in_exp_rep=True, fully_random=False):
        # Select an action
        if explore:
            # Raw actions are the logits for the actions. Useful for e.g. DDPG training in discrete envs.
            action, raw_action = self.agent.explore(state, fully_random=fully_random)
        else:
            action, raw_action = self.agent.exploit(state)

        self.log.add("ActionIdx", action, make_distribution=True, skip_steps=self.log_freq)

        # Apply the action:
        next_state, reward, done, _ = env.step(action)
        # Add possible noise to the reward:
        reward = self.modify_env_reward(reward)
        # Move action to GPU if desired:
        raw_action = self.prep_for_GPU(raw_action)
        # Define next state in case it is terminal:
        if done:
            next_state = None
        else:
            if not isinstance(next_state, dict):
                next_state = self.prep_for_GPU(next_state)
            else:
                next_state = apply_rec_to_dict(self.prep_for_GPU, next_state)
        # Store the transition in memory:
        if self.use_exp_rep and store_in_exp_rep:
            self.agent.remember(state, raw_action, next_state, reward, done)
        # Calculate TDE for debugging purposes:
        # TODO: implement logging of TDE
        # tde = self.policy.calculate_Q_and_TDE(state, raw_action, next_state, reward, done)
        #self.log.add("TDE_live", tde)
        # Render:
        if render:
            self.env.render()

        return action, next_state, reward, done

    def evaluate_model(self):
        reward_sum = 0
        test_env = self.create_env(self.hyperparameters)
        for i in range(self.eval_rounds):
            test_state = test_env.reset()
            for t in itertools.count():
                action, _ = self.agent.exploit(test_state)
                test_state, reward, done, _ = test_env.step(action)
                reward_sum += reward
                if done:
                    break
        reward_sum /= self.eval_rounds
        return reward_sum


    def _act_in_test_env(self, test_state, test_episode_rewards):
        if self.test_env is None:
            self.test_env = gym.make(self.env_name)
        _, next_state, reward, done = self._act(self.test_env, test_state, explore=False, store_in_exp_rep=False, render=False)

        test_episode_rewards.append(reward)
        self.log.add("Test_Env Reward", np.sum(test_episode_rewards))
        if done or (self.max_steps_per_episode > 0 and len(test_episode_rewards) >= self.max_steps_per_episode):
            next_state = self.test_env.reset()
            test_episode_rewards.clear()

        return next_state

    def _display_debug_info(self, i_episode, steps_done, train_fraction):
        episode_return = self.log.get_episodic("Return")
        sampling_time = self.log.get_episodic("Sampling_Time")
        optimize_time = self.log.get_episodic("Optimize_Time")
        non_optimize_time = self.log.get_episodic("Non-Optimize_Time")
        print(round(train_fraction * 100, 1), "%")
        print("#Episode ", i_episode)
        print("#Steps: ", steps_done)
        print("Return:", episode_return[0])

        #print( "Total Opt-time: ", round(np.mean(optimize_time), 4), "s")
        #print(" Sampling-time: ", round(np.mean(sampling_time), 4), "s")
        #print(" Non-Opt-time: ", round(np.mean(non_optimize_time), 4), "s")
        if i_episode % 10 == 0:
            pass
            # TODO: do an extensive test in test_env every N percentage points of training
            # Not needed anymore, because he have tensorboard now
            #plot_rewards(rewards)
            #plot_rewards(self.log.storage["Test_Env Reward"], "Test_Env Reward")
            #plot_rewards(self.log.storage["Return"], "Return", xlabel="Episodes")
        print()

    def close(self):
        self.env.close()
        #self.log.close()

    def run(self, n_hours=0.0, n_episodes=0, n_steps=0, verbose=False, render=False, on_server=True):
        assert (bool(n_steps) ^ bool(n_episodes) ^ bool(n_hours))

        steps_done = 0
        i_episode = 0
        start_time = time.time()
        state = None
        # Fill replay buffer with random actions:
        if not self.use_expert_data:
            self.fill_replay_buffer(n_steps=self.n_initial_random_actions)

        if self.freeze_normalizer:
            print("Freeze observation Normalizer.")
            self.agent.freeze_normalizers()

        if self.use_expert_data and self.do_pretrain:
            pretrain_steps = int(self.pretrain_percentage * n_steps)
            pretrain_episodes = int(self.pretrain_percentage * n_episodes)
            pretrain_time = int(self.pretrain_percentage * n_hours)
            if pretrain_episodes:
                pretrain_steps = pretrain_episodes * 5000
                # TODO: instead of hardcoding 5000 get an expected episode duration from somewhere

            self.pretrain(pretrain_steps, pretrain_time, start_time)
            steps_done += pretrain_steps
            i_episode += pretrain_episodes

        # TODO: For MineRL only train for a certain amount of time: stop after something like 99% of all training time at least leave 10-30 mins empty

        # Do the actual training:
        print("Start training in the env:")
        time_after_optimize = None
        #pbar = tqdm(total=n_steps, desc="Total Training", disable=self.disable_tqdm)
        train_fraction = calc_train_fraction(n_steps, steps_done, n_episodes, i_episode, n_hours, start_time)
        print("train frac: ", train_fraction)
        while train_fraction < 1:
            i_episode += 1
            # Initialize the environment and state. Do not reset
            if state is None:
                state = self.env.reset()
                if not isinstance(state, dict):
                    state = self.prep_for_GPU(state)
                else:
                    state = apply_rec_to_dict(self.prep_for_GPU, state)

            for t in tqdm(count(), desc="Episode Progress", total=self.tqdm_episode_len, disable=self.disable_tqdm):
                print("new ep")
                steps_done += 1

                # Act in train env:
                action, next_state, reward, done = self._act(self.env, state, render=render, store_in_exp_rep=True,
                                                             explore=True)

                # Evaluate agent thoroughly sometimes:
                if self.eval_rounds > 0 and train_fraction - self.stored_percentage >= self.eval_percentage \
                        or train_fraction == 0:
                    self.stored_percentage = train_fraction
                    test_return = self.evaluate_model()
                    self.log.add("Test Return", test_return, steps=steps_done)#steps=train_fraction * 100)
                    print("Model performance after ", steps_done, "steps: ", test_return)

                # Move to the next state
                state = next_state

                # Reduce epsilon and other exploratory values:
                self.agent.decay_exploration(steps_done)

                time_before_optimize = time.time()
                # Log time between optimizations:
                non_optimize_time = 0
                if time_after_optimize is not None:
                    non_optimize_time = time_before_optimize - time_after_optimize
                self.log.add("Non-Optimize_Time", non_optimize_time, skip_steps=self.log_freq, store_episodic=True)

                num_updates = int(self.updates_per_step) if self.updates_per_step >= 1\
                                                    else steps_done % int(1 / self.updates_per_step) == 0
                for _ in range(num_updates):
                    # Perform one step of the optimization (on the target network)      
                    self.agent.optimize()
                    # Update the target network
                    self.agent.update_targets(steps_done, train_fraction=train_fraction)
                time_after_optimize = time.time()

                # Log reward and time:
                self.log.add("Train_Reward", reward.item(), skip_steps=10, store_episodic=True)
                #if len(self.log.get_episodic("Train_Reward")) >= 1:
                 #   self.log.add("Train_Sum_Episode_Reward", np.sum(self.log.get_episodic("Train_Reward")),
                 #                skip_steps=self.log_freq)
                self.log.add("Optimize_Time", time_after_optimize - time_before_optimize, skip_steps=self.log_freq,
                             store_episodic=True)

                if render:

                    self.env.render()

                self.log.step()
                if done or (n_steps and steps_done >= n_steps) or (n_episodes and i_episode >= n_episodes) or\
                        (n_hours and (time.time() - start_time) / 360 >= n_hours):


                    #pbar.update(t)

                    self.log.add("Return", np.sum(self.log.get_episodic("Train_Reward")), steps=i_episode, store_episodic=True)
                    self.log.add("Episode Len", t, steps=i_episode)
                    if verbose:
                        self._display_debug_info(i_episode, steps_done, train_fraction)
                    self.log.flush_episodic()
                    state = None
                    break
                train_fraction = calc_train_fraction(n_steps, steps_done, n_episodes, i_episode, n_hours, start_time)
            train_fraction = calc_train_fraction(n_steps, steps_done, n_episodes, i_episode, n_hours, start_time)

        # Save the model:
        self.agent.update_targets(steps_done, train_fraction=1.0)
        print('Done.')
        self.log.flush()
        self.env.close()
        #pbar.close()
        return i_episode

