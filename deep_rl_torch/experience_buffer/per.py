import numpy as np
import random

import torch


from .base_replay import ReplayBuffer, RLDataset
from .segment_tree import SumSegmentTree, MinSegmentTree


class PERDataset(RLDataset):
    def __init__(self, alpha, *args, max_priority=1.0, running_avg=0.0):
        super().__init__(*args)
        self.alpha = alpha
        self.max_priority = max_priority
        self.running_avg = running_avg
        self.beta = None
        self.max_weight = None
        # Create Sum tres:
        it_capacity = 1
        while it_capacity < self.max_size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        
        
    def __getitem__(self, index):
        out = super().__getitem__(index)
        #out = self.data[index]
        weight = self._calc_weight(index)
        weight = torch.tensor(weight)
        out.append(weight)
        return out
        
    def add(self, *args, **kwargs):
        index = self.next_idx
        self._calculate_priority_of_last_add(index)
        super().add(*args, **kwargs)
        
    def sample_idx(self):
        mass = random.random() * self._it_sum.sum(0, len(self) - 1)
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx
    
    def update_priorities(self, idcs, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idcs) == len(priorities)
        for idx, priority in zip(idcs, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            old_priority = self._it_sum[idx] * self.running_avg
            new_priority = old_priority + (priority ** self.alpha) * (1 - self.running_avg) 
            self._it_sum[idx] = new_priority
            self._it_min[idx] = new_priority
            self.max_priority = max(self.max_priority, new_priority)
    
    def calc_and_save_max_weight(self):
        tree_min = self._it_min.min()
        tree_sum = self._it_sum.sum()
        p_min = tree_min / tree_sum
        self.max_weight = (p_min * len(self)) ** (-1 * self.beta)
        self.tree_sum = tree_sum
        
    def _calc_weight(self, index):
        p_sample = self._it_sum[index] / self.tree_sum
        weight = (p_sample * len(self)) ** (-1 * self.beta)
        weight /= self.max_weight
        return weight
        
    def _calculate_priority_of_last_add(self, idx):
        self._it_sum[idx] = self.max_priority ** self.alpha
        self._it_min[idx] = self.max_priority ** self.alpha

        
class PERBuffer(ReplayBuffer):
    def __init__(self, *args): 
        super().__init__(*args)
        # Add name to output dict:
        self.transition_names.append("importance_weights")
    
    def sample(self, beta):
        self.data.beta = beta
        self.data.calc_and_save_max_weight()
        out = super().sample()
        return out
    
    def update_priorities(self, idcs, priorities):
        idcs = idcs.cpu().numpy()
        priorities = priorities.cpu().numpy()
        self.data.update_priorities(idcs, priorities)
        
        
        
        
        
# old class:
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

        weights = []
        tree_min = self._it_min.min()
        tree_sum = self._it_sum.sum()
        p_min = tree_min / tree_sum
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / tree_sum
            weight = (p_sample * len(self._storage)) ** (-beta)
            weight /= max_weight
            weights.append(weight)
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

