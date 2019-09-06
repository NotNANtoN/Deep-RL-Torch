import minerl
# push
import gym

gym.logger.set_level(40)
from itertools import count


import torch

import time

from networks import *
from policies import Agent
from env_wrappers import FrameSkip



class Trainer:
    def __init__(self, env_name, hyperparameters, log=True, tb_comment="", log_NNs=False):
        # Init logging:
        self.path = os.getcwd()
        self.log = Log(self.path + '/tb_log', log, tb_comment, log_NNs)
        self.steps_done = 0
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
            forward_when_jump = hyperparameters["forward_when_jump"]
            self.env = action_wrapper(self.env, always_keys=always_keys, exclude_keys=exclude_keys,
                                      forward_when_jump=forward_when_jump)
        # Extract relevant hyperparameters:
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

        # Init Policy:
        self.policy = Agent(self.env, self.device, self.log, hyperparameters)

    def reset(self):
        self.steps_done = 0
        self.episode_durations = []
        self.policy.reset()

    def optimize(self):
        self.policy.optimize()

    def modify_env_reward(self, reward):
        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        if self.reward_std:
            reward += torch.tensor(np.random.normal(0, self.reward_std))
        return reward

    def fill_replay_buffer(self, n_actions):
        state = self.env.reset()
        if not isinstance(state, dict):
            state = torch.tensor([state], device=self.device).float()

        # Fill exp replay buffer so that we can start training immediately:
        for _ in range(n_actions):

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

    def _act(self, env, state, explore=True, render=False, store_in_exp_rep=True, fully_random=False):
        # Select an action
        if explore:
            # Raw actions are the logits for the actions. Useful for e.g. DDPG training in discrete envs.
            action, raw_action = self.policy.explore(state, fully_random=fully_random)
        else:
            action, raw_action = self.policy.exploit(state)

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
        TDE = self.policy.calculate_TDE(state, raw_action, next_state, reward, done)
        self.log.add("TDE live", TDE.item())
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

    def _display_debug_info(self, i_episode):
        episode_return = self.log.get_episodic("Return")
        optimize_time = self.log.get_episodic("Optimize Time")
        non_optimize_time = self.log.get_episodic("Non-Optimize Time")
        print("#Episode ", i_episode)
        print("#Steps: ", self.steps_done)
        print(" Return:", episode_return[0])
        print(" Opt-time: ", round(np.mean(optimize_time), 4), "s")
        print(" Non-Opt-time: ", round(np.mean(non_optimize_time), 4), "s")
        if i_episode % 10 == 0:
            pass
            # TODO: to an extensive test in test_env every N steps
            # Not needed anymore, because he have tensorboard now
            #plot_rewards(rewards)
            #plot_rewards(self.log.storage["Test_Env Reward"], "Test_Env Reward")
            #plot_rewards(self.log.storage["Return"], "Return", xlabel="Episodes")
        print()

    def run(self, n_steps, verbose=False, render=False, on_server=True):
        # Fill replay buffer with random actions:
        self.fill_replay_buffer(n_actions=self.n_initial_random_actions)

        if self.freeze_normalizer:
            self.policy.freeze_normalizers()

        # Initialize test environment:
        # test_env = gym.make(self.env_name).unwrapped
        # test_state = test_env.reset()
        # test_state = torch.tensor([test_state], device=self.device).float()

        # Do the actual training:
        time_after_optimize = None
        i_episode = 0
        while self.steps_done < n_steps:
            i_episode += 1
            # Initialize the environment and state
            state = self.env.reset()
            if not isinstance(state, dict):
                state = torch.tensor([state], device=self.device).float()

            for t in count():
                self.steps_done += 1
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
                self.policy.decay_exploration(self.steps_done)

                time_before_optimize = time.time()
                # Log time between optimizations:
                if time_after_optimize is not None:
                    self.log.add("Non-Optimize Time", time_before_optimize - time_after_optimize)

                num_updates = self.updates_per_step if self.updates_per_step >= 1\
                                                    else n_steps % (1 / self.updates_per_step) == 0
                for _ in range(num_updates):
                    # Perform one step of the optimization (on the target network)
                    self.policy.optimize()

                    # Update the target network
                    self.policy.update_targets(self.steps_done)
                time_after_optimize = time.time()

                # Log reward and time:
                self.log.add("Train Reward", reward.item())
                self.log.add("Train Cum. Episode Reward", np.sum(self.log.get_episodic("Train Reward")))
                self.log.add("Optimize Time", time_after_optimize - time_before_optimize)

                if render:
                    self.env.render()

                self.log.step()
                if done or (self.max_steps_per_episode > 0 and t >= self.max_steps_per_episode) \
                        or self.steps_done >= n_steps:
                    self.log.add("Return", np.sum(self.log.get_episodic("Train Reward")), steps=i_episode)
                    if verbose:
                        self._display_debug_info(i_episode)
                    self.log.flush_episodic()
                    break

        print('Done.')
        self.env.close()
        return i_episode, rewards, self.log.storage



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