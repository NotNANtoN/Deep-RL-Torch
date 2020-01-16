import random

import numpy as np
import torch

from deep_rl_torch.util import Transition


                                                  
class RLDataset(torch.utils.data.Dataset):
    def __init__(self, max_size, sample, action_sample, size_expert_data, max_weight=0):
        self.max_weight = max_weight
        self.max_size = max_size + size_expert_data
        self.size_expert_data = size_expert_data
        if isinstance(sample, torch.Tensor):
            pass
        elif isinstance(sample, list):
            raise NotImplementedError("No dict support yet")
        elif isinstance(sample, dict):
            raise NotImplementedError("No dict support yet")
        else:
            raise Error("Unknown type: " + type(sample) + " of the env sample. Can only use Tensor, list or dict.")
        shp = list(sample.shape)
        action_shp = list(action_sample.shape)
        self.states = torch.empty([max_size] + shp, dtype=sample.dtype)
        self.rewards = torch.empty(max_size)
        self.actions = torch.empty([max_size] + action_shp)
        self.dones = torch.empty(max_size, dtype=torch.bool)
        self.weights = torch.empty(max_size)
        self.next_idx = 0
        self.looped_once = False
        
    def add(self, state, action, next_state, reward, done, store_episodes=False):
        # 1. store data
        # 2. store pointer to episodic stuff
        # 3. increment index
       
        # 1.
        self.states[self.next_idx] = state
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward
        self.dones[self.next_idx] = done
        self.weights[self.next_idx] = self.max_weight
        
        # 2.
        pass
        
        # 3.
        self.increment_idx()
        
        # TODO: this way we cannot sample the lastly stored pair - no next state is stored yet!
        
        
    def increment_idx(self):
        """ Loop the idx from front to end. Skip expert data, as we want to keep that forever"""
        self._next_idx += 1
        remainder = self._next_idx % self.maxsize
        if remainder == 0: 
            self._next_idx = 0 + self.size_expert_data
            self.looped_once = True
            
    def __len__(self):
        """ Return number of transitions stored so far """
        if self.looped_once:
            return self.max_size
        else:
            return self._next_idx
        
    def __getitem__(self, index):
        """ Return a single transition """
        next_state = self.state[index + 1] if not self.dones[index] else None
        return self.state[index], self.actions[index], self.rewards[index], next_state, index, self.weights[index]
        
    def update_stored_hidden_states(self, idxs, hidden_states, seq_lens):
        """ For R2D2, eventually. Updates the stored hidden states """
        pass
       
    

class ReplayBufferNew:
    def __init__(self, size, sample, action_sample, batch_size, pin_mem, num_sampling_workers, use_CER=False, size_expert_data=0):
        self.batch_size = batch_size
        self.use_CER = use_CER
        
        self.data = RLDataset(size, sample, action_sample, size_expert_data)
        
        #self._storage = []
        #self._expert_data_storage = []
        #self._maxsize = size + size_expert_data
        #self._next_idx = 0

        #self.size_expert_data = size_expert_data

        #self.episodic_transitions = [[]]
        #self.episodic_idxs = [[]]
        
        sampler = self.construct_dataloader(self.data)
        self.dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                  sampler=sampler,
                                                  pin_memory=pin_mem,
                                                  num_workers=num_sampling_workers,
                                                  collate_fn=custom_collate)
                                                  
    def add(self, state, action, next_state, reward, done, store_episodes=False):
        self.data.add(state, action, next_state, reward, done, store_episodes)
        

    def get_transitions_new(self):
        """ Gets transitions from dataloader, which is a batch of transitions. It is a dict of the form {"states": Tensor, "actions_argmax": Tensor of Ints, "actions": Tensor of raw action preferences, "rewards": Tensor, "non_final_next_states": Tensor, "non_final_mask": Tensor of bools, "Dones": Tensor, "Importance_Weights: Tensor, "Idxs": Tensor} """
        transitions= next(self.dataloader)
        return transitions

    
    def __len__(self):
        return len(self.data)
            
    def collate_batch(self, batch):
        batch = [[sample[idx] for sample in batch] for idx in len(batch)]
        names = ["state", "action", "reward", "next_state", "idxs", "importance_weights"]
        batch_dict = {key: value for key, value in zip(names, batch)}
        # Next state:
        batch_dict["non_final_mask"] = [val is not None for val in batch_dict["next_state"]]
        batch_dict["non_final_next_states"] = [state for val, non_final in zip(batch_dict["next_state"], batch_dict["non_final_mask"]) if non_final]
        del batch_dict["next_state"]
        # Action argmax:
        batch_dict["action_argmax"] = torch.argmax(batch_dict["actions"], 1).unsqueeze(1)
        # Stack in tensors:
        # TODO: does not work if state is a list or dict - there individual stacking per list/dict is needed
        for key in batch_dict:
            batch_dict[key] = torch.stack(batch_dict[key])
        return batch_dict
    
    def construct_sampler(self, data):
        return torch.utils.data.sampler.RandomSampler(data, replacement=False, num_samples=self.batch_size)


        
   
class PrioritizedReplayBuffer(ReplayBufferNew):
    def __init__(self, size, sample, action_sample, batch_size, pin_mem, num_sampling_workers, alpha, use_CER=False, max_priority=1.0, size_expert_data=0):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size, sample, action_sample, batch_size, pin_mem, num_sampling_workers, use_CER=use_CER, size_expert_data=size_expert_data)
        assert alpha >= 0
        self._alpha = alpha
        
    def construct_sampler(self, data):
        return torch.utils.data.sampler.WeightedRandomSampler(weights, self.batch_size, replacement=False)
        


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

