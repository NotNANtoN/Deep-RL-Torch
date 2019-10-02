import random
import numpy as np
from util import Transition
from segment_tree import SumSegmentTree, MinSegmentTree

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

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, use_CER=False, max_priority=1.0):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size, use_CER=use_CER)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = max_priority
        self.counts = np.zeros(size)

    def calculate_priority_of_last_add(self, idx):
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
        # TODO: this should be changed such that the cpu calculates the priority directly (or just use CER)

    def add(self, state, action, next_state, reward, done, store_episodes=False):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(state, action, next_state, reward, done, store_episodes=store_episodes)
        self.calculate_priority_of_last_add(idx)

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=0, remove_samples_from_buffer=False):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = []
        if self.use_CER:
            batch_size -= 1
            idxes.append(self.last_idx())

        idxes.extend(self._sample_proportional(batch_size))

        self.counts[idxes] += 1

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample, weights, idxes

    def update_priorities(self, idxes, priorities, running_avg=0):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            old_priority = self._it_sum[idx]
            new_priority = running_avg * old_priority + (1 - running_avg) * priority ** self._alpha
            self._it_sum[idx] = new_priority
            self._it_min[idx] = new_priority

            self._max_priority = max(self._max_priority, priority)

