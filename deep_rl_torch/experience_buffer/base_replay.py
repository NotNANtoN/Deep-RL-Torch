import random
from collections import namedtuple

import numpy as np
import torch
import gym

from deep_rl_torch.util import apply_rec_to_dict

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))
                                                  
class RLDataset(torch.utils.data.Dataset):
    def __init__(self, max_size, sample, action_space, size_expert_data, max_weight=1):
        self.max_weight = max_weight
        self.max_size = max_size + size_expert_data
        self.size_expert_data = size_expert_data
        if isinstance(sample, torch.Tensor) or isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)
            shp = list(sample.shape)
        elif isinstance(sample, list):
            raise NotImplementedError("No list support yet")
        elif isinstance(sample, dict):
            raise NotImplementedError("No dict support yet")
        else:
            raise TypeError("Unknown type: " + str(type(sample)) + " of the env sample. Can only use Tensor, list or dict.")
            
        if isinstance(action_space, gym.spaces.Box):
            action_shp = [len(action_space.low)]
        elif isinstance(action_space, gym.spaces.Discrete):
            action_shp = [action_space.n]
        else:
            raise TypeError("Unsupport action space type: " + str(action_space))
        # TODO: change zeros back to empty when done with debugging
        self.states = torch.zeros([max_size] + shp, dtype=sample.dtype)
        self.rewards = torch.zeros(max_size)
        self.actions = torch.zeros([max_size] + action_shp)
        self.dones = torch.zeros(max_size, dtype=torch.bool)
        self.weights = torch.zeros(max_size)
        self.next_idx = 0
        self.curr_idx = 0
        self.looped_once = False
        
    def add(self, state, action, next_state, reward, done, store_episodes=False):
        # 1. store data 2. store pointer to episodic stuff 3. increment index
       
        # 1.
        self.states[self.next_idx] = state
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward
        self.dones[self.next_idx] = done
        self.weights[self.next_idx] = self.max_weight
        
        # 2.
        pass
        
        # 3.
        self.curr_idx = self.next_idx
        self.next_idx = self.increment_idx(self.next_idx)
                
    def increment_idx(self, index):
        """ Loop the idx from front to end. Skip expert data, as we want to keep that forever"""
        index += 1
        remainder = index % self.max_size
        if remainder == 0: 
            index = 0 + self.size_expert_data
            self.looped_once = True
        return index
            
    def decrement_idx(self, index):
        index -= 1
        if index < 0:
            index = len(self) - 1
        return index
        
    def __len__(self):
        """ Return number of transitions stored so far """
        if self.looped_once:
            return self.max_size
        else:
            return self.next_idx
        
    def __getitem__(self, index):
        """ Return a single transition """
        # Check if the last state is being attempted to sampled - it has no next state yet:
        if index == self.curr_idx:
            index = self.decrement_idx(index)
        # Check if there is a next_state:
        next_state = self.states[index + 1] if not self.dones[index] else None
        return self.states[index], self.actions[index], self.rewards[index], next_state, torch.tensor(index), self.weights[index]
    
    def __iter__(self):
        while True:
            idx = random.randint(0, len(self))
            yield self[idx]
        
    def update_stored_hidden_states(self, idxs, hidden_states, seq_lens):
        """ For R2D2, eventually. Updates the stored hidden states """
        pass
       
    
    def stack_frames(self, frames):
        return torch.cat(list(frames), dim=self.stack_dim)
    

class ReplayBufferNew(object):
    def __init__(self, size, sample, action_sample, batch_size, pin_mem, workers, device, use_CER=False, size_expert_data=0):
        self.batch_size = batch_size
        self.use_CER = use_CER
        self.device = device
        
        self.data = RLDataset(size, sample, action_sample, size_expert_data)
        
        #self._storage = []
        #self._expert_data_storage = []
        #self._maxsize = size + size_expert_data
        #self._next_idx = 0

        #self.size_expert_data = size_expert_data

        #self.episodic_transitions = [[]]
        #self.episodic_idxs = [[]]
        
        sampler = self.construct_sampler(self.data)
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=batch_size,# sampler=sampler,
                                                  pin_memory=pin_mem,
                                                  num_workers=workers,
                                                  collate_fn=self.collate_batch)
        self.iter = iter(self.dataloader)
        
                                                  
    def add(self, state, action, next_state, reward, done, store_episodes=False):
        self.data.add(state, action, next_state, reward, done, store_episodes)
        
    def sample(self):
        try:
            out = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            out = next(self.iter)
        out = self.move_batch(out)
        return out
    
    def sample_unsafe(self):
        return next(self.iter)
        
    def sample_returns_generator(self):
        """ Gets transitions from dataloader, which is a batch of transitions. It is a dict of the form {"states": Tensor, "actions_argmax": Tensor of Ints, "actions": Tensor of raw action preferences, "rewards": Tensor, "non_final_next_states": Tensor, "non_final_mask": Tensor of bools, "Dones": Tensor, "Importance_Weights: Tensor, "Idxs": Tensor} """    
        for transitions in self.dataloader:
            #print(transitions)
            yield transitions
        
        #transitions= next(self.dataloader)
        #return transitions
        
    def get_transitions_iter(self):
        """ Gets transitions from dataloader, which is a batch of transitions. It is a dict of the form {"states": Tensor, "actions_argmax": Tensor of Ints, "actions": Tensor of raw action preferences, "rewards": Tensor, "non_final_next_states": Tensor, "non_final_mask": Tensor of bools, "Dones": Tensor, "Importance_Weights: Tensor, "Idxs": Tensor} """
        return next(iter(self.dataloader))
    
    def get_transitions_iter(self):
        """ Gets transitions from dataloader, which is a batch of transitions. It is a dict of the form {"states": Tensor, "actions_argmax": Tensor of Ints, "actions": Tensor of raw action preferences, "rewards": Tensor, "non_final_next_states": Tensor, "non_final_mask": Tensor of bools, "Dones": Tensor, "Importance_Weights: Tensor, "Idxs": Tensor} """
        return next(iter(self.dataloader))

    def __len__(self):
        size = len(self.data)
        print("len of loader: ", size)
        return size
            
    def stack(self, data):
        if isinstance(data, dict):
            return apply_rec_to_dict(lambda x: torch.stack(x), data)
        return torch.stack(data)
    
    def move_batch(self, batch):
        for key in batch:
            content = batch[key]
            if content is None:
                continue
            if isinstance(content, dict):
                new_cont = apply_rec_to_dict(lambda x: x.to(self.device))
            else:
                new_cont = content.to(self.device)
            batch[key] = new_cont
        return batch
        
    def collate_batch(self, batch):
        # Create dict:
        batch = [[sample[idx] for sample in batch] for idx in range(len(batch[0]))]
        names = ["states", "actions", "rewards", "next_states", "idxs", "importance_weights"]
        batch_dict = {key: value for key, value in zip(names, batch)}
        # Next state:
        batch_dict["non_final_mask"] = torch.tensor([val is not None for val in batch_dict["next_states"]]).bool()
        batch_dict["non_final_next_states"] = [state for state in batch_dict["next_states"] if state is not None]
        if batch_dict["non_final_next_states"] != []:
            batch_dict["non_final_next_states"] = self.stack(batch_dict["non_final_next_states"])
        else:
            batch_dict["non_final_next_states"] = None
        del batch_dict["next_states"]
        # Action argmax:
        batch_dict["action_argmax"] = torch.argmax(self.stack(batch_dict["actions"]), 1).unsqueeze(1)
        # Stack in tensors:
        for key in batch_dict:
            if key in ["action_argmax", "non_final_mask", "non_final_next_states"]:
                continue
            content = self.stack(batch_dict[key])
            batch_dict[key] = content
        batch_dict["rewards"] = batch_dict["rewards"].unsqueeze(1)
        return batch_dict
    
    
    def construct_sampler(self, data):
        return torch.utils.data.sampler.RandomSampler(data, replacement=True, num_samples=self.batch_size)


class PrioritizedReplayBufferNew(ReplayBufferNew):
    def __init__(self, size, sample, action_sample, batch_size, pin_mem, workers, alpha, use_CER=False, max_priority=1.0, size_expert_data=0):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the oldest memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        self.weights = torch.ones(0)
        super().__init__(size, sample, action_sample, batch_size, pin_mem, workers, use_CER=use_CER, size_expert_data=size_expert_data)
        assert alpha >= 0
        self._alpha = alpha
        self.max_priority = max_priority
        
    def construct_sampler(self, data):
        return torch.utils.data.sampler.WeightedRandomSampler(self.weights, self.batch_size, replacement=True)
    
    def update_weights(self, idcs, weights):
        self.dataloader.sampler.weights[idcs] = weights
    
    def add_weight(self):
        weights = self.dataloader.sampler.weights
        new_weights = torch.cat([weights, torch.tensor([self.max_priority]).double()])
        self.dataloader.sampler.weights = new_weights
        
    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)
        if not self.data.looped_once:
            self.add_weight()
        


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

