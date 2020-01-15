import random

import numpy as np
import torch

from deep_rl_torch.util import Transition


class RLDataset(torch.utils.data.Dataset):
    def __init__(self, size, sample, size_expert_data):
        pass
        
    def __len__(self):
        """ Return number of transitions stored so far """
        pass
        
    def __getitem__(self, index):
        """ Return a single transition """
        pass
        
    def update_stored_hidden_states(self, idxs, hidden_states, seq_lens):
        """ For R2D2, eventually. Updates the stored hidden states """
        pass
       
    

class ReplayBufferNew:
    def __init__(self, size, sample, use_CER=False, size_expert_data=0):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        
        if isinstance(sample, torch.Tensor):
            pass
        elif isinstance(sample, list):
            pass
        elif isinstance(sample, dict):
            pass
        else:
            raise Error("Unknown type: " + type(sample) + " of the env sample. Can only use Tensor, list or dict.")
        
        
        self.data = RLDataset(size, sample, size_expert_data)
        
        #self._storage = []
        #self._expert_data_storage = []
        #self._maxsize = size + size_expert_data
        #self._next_idx = 0

        self.use_CER = use_CER
        #self.size_expert_data = size_expert_data

        #self.episodic_transitions = [[]]
        #self.episodic_idxs = [[]]
        
class IcuDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, args, device):
        self.r2d2 = args.r2d2
        self.r2d2_burn_len = args.r2d2_burn_len
        self.r2d2_train_len = args.r2d2_train_len
        self.hidden_size = args.hidden_size
        self.device = device
        use_outcome = args.use_outcome
        self.use_outcome = use_outcome
        use_meds = args.use_meds
        predict_outcome_time = args.predict_outcome_time

        # Load data
        data = torch.load(data_path + "input.pt")
        outcome = torch.load(data_path + "outcome.pt")
        meds = torch.load(data_path + "meds.pt")


    def get_model_input_output(self):
        """ Returns number of features for input and target for first timestep of first patient"""
        return len(self.inputs[0][0]), len(self.targets[0][0])

    def __len__(self):
        if self.r2d2:
            # TODO: possibly change len to include all possible combinations!
            # make it independent from r2d2
            self.len = len(self.targets)
        else:
            self.len = len(self.targets)
        return self.len

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]

        if self.r2d2:
            len_x = len(x)
            # TODO: add option to always get full seq for patients with short length
            seq_idx = random.randint(0, len_x)
            start_idx = max(seq_idx - self.r2d2_burn_len, 0)
            end_idx = min(seq_idx + self.r2d2_train_len, len_x)
            x = x[start_idx:end_idx]
            y = y[start_idx:end_idx]
            seq_len = end_idx - start_idx
            if start_idx == 0:
                hidden_state = (self.dummy_hidden_state.clone(), self.dummy_hidden_state.clone())
            else:
                hidden_state = self.hidden_states[index][start_idx - 1]
                cell_state = self.hidden_cell_states[index][start_idx - 1]
                hidden_state = (hidden_state, cell_state)
        else:
            start_idx = 0
            seq_len = len(x)
            hidden_state = (self.dummy_hidden_state.clone(), self.dummy_hidden_state.clone())

        return x, y, (index, start_idx), seq_len, hidden_state

    def update_stored_hidden_states(self, idxs, hidden_states, seq_lens):
        # hidden_state shape: [seq_len, batch_size, hidden_size]
        outputs = hidden_states[0].permute(1, 0, 2)
        cell_states = hidden_states[1].permute(1, 0, 2)
        for i, (pat_idx, start_idx) in enumerate(idxs):
            seq_len = seq_lens[i]
            end_idx = start_idx + seq_len
            self.hidden_states[pat_idx][start_idx:end_idx] = outputs[i][:seq_len]
            self.hidden_cell_states[pat_idx][start_idx:end_idx] = cell_states[i][:seq_len]

class ReplayBuffer(object):
    def __init__(self, size, use_CER=False, size_expert_data=0):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._expert_data_storage = []
        self._maxsize = size + size_expert_data
        self._next_idx = 0

        self.use_CER = use_CER
        self.size_expert_data = size_expert_data

        self.episodic_transitions = [[]]
        self.episodic_idxs = [[]]

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, state, action, next_state, reward, done, store_episodes=False):
        data = Transition(state, action, next_state, reward, done)

        # Store data:
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        # Also store pointer to transitions aggregated per episode for stuff like eligibility traces:
        if store_episodes:
            self.episodic_transitions[-1].append(data)
            self.episodic_idxs[-1].append(self._next_idx)
            if data.done:
                self.episodic_transitions.append([])
                self.episodic_idxs.append([])

            if len(self._storage) == self._maxsize:
                del self.episodic_transitions[0][0]
                if self.episodic_transitions[0] == []:
                    del self.episodic_transitions[0]
                del self.episodic_idxs[0][0]
                if self.episodic_idxs[0] == []:
                    del self.episodic_idxs[0]

        self._next_idx += 1
        remainder = self._next_idx % self._maxsize
        self._next_idx = 0 + self.size_expert_data if remainder == 0 else remainder


    def add_old(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        return [self._storage[idx] for idx in idxes]

    def _encode_sample_old(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def last_idx(self):
        previous_idx =  self._next_idx - 1
        if previous_idx < 0:
            return min(self._maxsize - 1, len(self._storage))
        else:
            return previous_idx

    def sample(self, batch_size, beta=0, remove_samples_from_buffer=False):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        # Duplicates allowed:
        #idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        # No duplicates:
        idxes = []
        if self.use_CER:
            batch_size -= 1
            idxes.append(self.last_idx())

        idxes.extend(random.sample(range(len(self._storage)), batch_size))
        samples = self._encode_sample(idxes)

        if remove_samples_from_buffer:
            for i in sorted(idxes, reverse=True):
                del self._storage[i]
            
        return samples, idxes

    def get_all_episodes(self):
        return self.episodic_transitions, self.episodic_idxs

    def get_most_recent_episode(self):
        if self.episodic_transitions[-1] == []:
            return self.episodic_transitions[-2], self.episodic_idxs[-2]
        else:
            return self.episodic_transitions[-1], self.episodic_idxs[-1]

    def get_all_episodes_old(self):
        # TODO: this method does not take into account that an episode might span from the end of the buffer to the start - so the trace of one episode will likely be cut if the buffer is full. Not sure how bad this is
        episodes = []
        current_episode = []
        current_idxs = []
        idx_list = []

        for idx, transition in enumerate(self._storage[::-1]):
            real_idx = len(self._storage) - 1 - idx
            if transition.done or real_idx == self._next_idx - 1:
                if len(current_episode) != 0:  # extra check for initial transition
                    episodes.append(current_episode)
                    idx_list.append(current_idxs)
                current_episode = [transition]
                current_idxs = [real_idx]
            else:
                #current_episode.append(transition)
                #current_idxs.append(real_idx)
                current_episode.insert(0, transition)
                current_idxs.insert(0, real_idx)
        if len(current_episode) != 0:
            episodes.append(current_episode)
            idx_list.append(current_idxs)

        return episodes, idx_list

    def get_most_recent_episode_old(self):
        episode = []
        idxs = []
        for idx, transition in enumerate(self._storage[::-1]):
            real_idx = len(self._storage) - 1 - idx
            if idx != 0 and (transition.done or real_idx == self._next_idx - 1):
                return episode, idxs
            else:
                #episode.append(transition)
                #idxs.append(real_idx)
                episode.insert(0, transition)
                idxs.insert(0, real_idx)
        return episode, idxs

