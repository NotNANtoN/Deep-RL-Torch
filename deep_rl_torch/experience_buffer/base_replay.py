import random

import numpy as np
import torch
import gym

from deep_rl_torch.util import apply_rec_to_dict, apply_to_state

                                                  
class RLDataset(torch.utils.data.IterableDataset):
    def __init__(self, log, max_size, sample, action_space, size_expert_data, stack_dim, stack_count, update_freq,
                 use_list):
        self.max_size = max_size + size_expert_data
        self.size_expert_data = size_expert_data
        self.stack_dim = stack_dim
        self.stack_count = stack_count
        self.update_freq = update_freq
        self.log = log
        self.use_list = use_list
        if not use_list:
            # Get state space shape:
            if isinstance(sample, torch.Tensor) or isinstance(sample, np.ndarray):
                if isinstance(sample, np.ndarray):
                    sample = torch.from_numpy(sample)
                shp = list(sample.shape)
            elif isinstance(sample, list):
                raise NotImplementedError("No observation list support yet")
            elif isinstance(sample, dict):
                raise NotImplementedError("No obsservation dict support yet")
            else:
                raise TypeError("Unknown type: " + str(type(sample)) + " of env sample. Can only use Tensor, list or dict.")
            # Get action space shape:
            if isinstance(action_space, gym.spaces.Box):
                action_shp = [len(action_space.low)]
            elif isinstance(action_space, gym.spaces.Discrete):
                action_shp = [action_space.n]
            else:
                raise TypeError("Unsupport action space type: " + str(action_space))

            self.states = torch.empty([max_size] + shp, dtype=sample.dtype)
            self.rewards = torch.empty(max_size)
            self.actions = torch.empty([max_size] + action_shp)
            self.dones = torch.empty(max_size, dtype=torch.bool)
        else:
            self.states = []
            self.rewards = []
            self.actions = []
            self.dones = []

        # Indexing fields:
        self.next_idx = 0
        self.curr_idx = 0
        self.looped_once = False
    
    def __len__(self):
        """ Return number of transitions stored so far """
        if self.looped_once:
            return self.max_size
        else:
            return self.next_idx
        
    def __getitem__(self, index):
        """ Return a single transition """
        if self.log.is_available("Sampled Idx", factor=1, reset=False):
            self.log.add("Sampled Idx", index, make_distr=True, distr_steps=self.log.log_steps // 4)
        # Check if the last state is being attempted to sampled - it has no next state yet:
        if index == self.curr_idx:
            index = self.decrement_idx(index)
        elif index >= len(self):
            raise ValueError("Error: index " + str(index) + " is too large for buffer of size " + str(len(self)))
        # Check if there is a next_state, if so stack frames:
        next_index = self.increment_idx(index)
        is_end = self.is_episode_boundary(index)
        if not self.use_list:
            # Stack state:
            state = self.stack_last_frames_idx(index)
            next_state = self.stack_last_frames_idx(next_index) if not is_end else None
        else:
            if not isinstance(self.states[index], torch.Tensor):
                state = self.states[index].make_state().squeeze()
                next_state = self.states[next_index].make_state().squeeze() if not is_end else None
            else:
                state = self.states[index].squeeze(0)
                next_state = self.states[next_index].squeeze(0) if not is_end else None

        return [state, self.actions[index].squeeze(), self.rewards[index].squeeze(), next_state, torch.tensor(index)]
    
    def __iter__(self):
        count = 0
        while True:
            count += 1
            idx = self.sample_idx()
            yield self[idx]
            if count == self.update_freq:
                raise StopIteration
            
    def sample_idx(self):
        return random.randint(0, len(self) - 1)

    def add(self, state, action, reward, done, store_episodes=False):
        # Mark episodic boundaries:
        #if self.dones[self.next_idx]:
        #    self.done_idcs.remove(self.next_idx)
        #if done:
        #    self.done_idcs.add(self.next_idx)
            
        # Store data:
        if self.use_list and not self.looped_once:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
        else:
            self.states[self.next_idx] = state
            self.actions[self.next_idx] = action
            self.rewards[self.next_idx] = reward
            self.dones[self.next_idx] = done
        
        # Take care of idcs:
        self.curr_idx = self.next_idx
        self.next_idx = self.increment_idx(self.next_idx)
                
    def increment_idx(self, index):
        """ Loop the idx from front to end. Skip expert data, as we want to keep that forever"""
        index += 1
        if index == self.max_size:
            index = 0 + self.size_expert_data
            self.looped_once = True
        return index
            
    def decrement_idx(self, index):
        index -= 1
        if index < 0:
            index = len(self) - 1
        return index
        
    def update_stored_hidden_states(self, idxs, hidden_states, seq_lens):
        """ For R2D2, eventually. Updates the stored hidden states.
        Note: might be better to store hidden states in
        network similar as is being done with eligibility traces! """
        pass
        
    def is_episode_boundary(self, idx):
        return self.dones[idx] or idx == self.curr_idx
        
    def get_preceeding_frames(self, idx, frame):
        """Create list of frames going back from idx with the current frame already filled in frames.
        If an episode boundary is encountered, the last frame is repeated
        """
        frames = [frame]
        done_flag = False
        for i in range(self.stack_count - 1):
            idx = self.decrement_idx(idx)
            if self.is_episode_boundary(idx):
                done_flag = True
            if not done_flag:
                frame = self.states[idx]
            frames.append(frame)

        return frames
    
    def stack_last_frames_idx(self, idx):
        """Stack frames for buffer based on index"""
        frame = self.states[idx]
        frames = self.get_preceeding_frames(idx, frame)
        return self.stack_frames(frames)  
    
    def stack_last_frames(self, frame):
        """Stack frames for actor based on frame. assumes the frame to be the most recently added frame"""
        frame = frame.squeeze(0)
        frames = self.get_preceeding_frames(self.curr_idx, frame)
        return self.stack_frames(frames).unsqueeze(0)

    
    def stack_frames(self, frames):
        return torch.cat(list(frames), dim=self.stack_dim)
    
    def get_last_transition_idx(self):
        idx = self.decrement_idx(self.curr_idx)
        return idx
    
    def get_episodes(self):
        episode_list = [idx for idx in self.done_idcs]
        
        pass
        #return self.episodic_transitions, self.episodic_idxs

    def get_most_recent_episode(self):
        pass
        #if self.episodic_transitions[-1] == []:
        #    return self.episodic_transitions[-2], self.episodic_idxs[-2]
        #else:
        #    return self.episodic_transitions[-1], self.episodic_idxs[-1]
    
    
class ReplayBuffer:
    def __init__(self, data, batch_size, pin_mem, workers, device):
        self.batch_size = batch_size
        self.pin_mem = pin_mem
        self.workers = workers
        self.device = device
        self.data = data
        
        #self._expert_data_storage = []
        #self._maxsize = size + size_expert_data
        #self.size_expert_data = size_expert_data

        #self.episodic_transitions = [[]]
        #self.episodic_idxs = [[]]
        
        # Create dataloder    
        self.transition_names = ["states", "actions", "rewards", "next_states", "idxs"]
        #self.construct_loader(self.data, batch_size, self.collate_batch)
        self.iter = None #iter([])
        #self.construct_loader(self.data, self.batch_size, self.collate_batch)
        self.stop_iteration_functions = []
        
        
    def construct_loader(self, data, batch_size, collate_fn):
        self.dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=None,
                                                  pin_memory=self.pin_mem,
                                                  num_workers=self.workers,
                                                  collate_fn=collate_fn)
        self.iter = iter(self.dataloader)

        
    def stack_last_frames(self, state):
        return self.data.stack_last_frames(state)
                                                  
    def add(self, state, action, reward, done, store_episodes=False):
        self.data.add(state, action, reward, done, store_episodes)
        
    def sample(self):
        if self.iter is None:
            self.construct_loader(self.data, self.batch_size, self.collate_batch)

        try:
            out = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            out = next(self.iter)
            for fn in self.stop_iteration_functions:
                fn()
            
        out = self.move_batch(out)
        return out
   
    def __len__(self):
        size = len(self.data)
        return size
            
    def stack(self, data):
        if isinstance(data, dict):
            return apply_rec_to_dict(lambda x: torch.stack(x), data)
        else:
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
        #for b in batch:
        #    print(len(b))
        #print("len trans nemaes: ", len(self.transition_names))
        # Create dict:
        batch_dict = {trans_name: [x[idx] for x in batch] for idx, trans_name in enumerate(self.transition_names)}
        # Next states:
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

            if key not in ("action_argmax", "non_final_mask", "non_final_next_states"):
                content = self.stack(batch_dict[key])
                batch_dict[key] = content
            if key not in ("action_argmax", "non_final_mask"):
                batch_dict[key] = batch_dict[key].float()
        # Bring rewards in correct shape
        batch_dict["rewards"] = batch_dict["rewards"].unsqueeze(1)

        return batch_dict
    


class CERWrapper:
    def __init__(self, buffer):
        self.buffer = buffer
        
        self.old_loader = self.buffer.construct_loader
        self.buffer.construct_loader = lambda data, batch, collate: self.old_loader(data, batch - 1, self.collate_batch)
        
        self.update_freq = buffer.data.update_freq // buffer.batch_size * buffer.workers
        self.buffer.stop_iteration_functions.append(self.reset_idx)
        self.count = 1
        
        # Create new dataloader with a batch size reduced by 1
        #self.construct_loader(buffer.data, buffer.batch_size, self.collate_batch)
        
    def __getattr__(self, attr):
        return getattr(self.buffer, attr)
    
    def reset_idx(self):
        self.count = 1

    def collate_batch(self, batch):
        # Adds the most recent (as recent as possible) transition to the batch
        workers = self.buffer.workers
        idx = self.buffer.data.get_last_transition_idx()
        # If we have multiple workers we need to assign a different idx per worker id
        # Also we need to count until when the buffer will be updated with new transitions:
        # this influences the CER idx
        if workers > 0:
            id_ = torch.utils.data.get_worker_info().id
            idx = self.buffer.data.get_last_transition_idx()
            subtract = self.update_freq * workers
            idx = min(idx - subtract + id_ + self.count * workers, idx)
            idx = max(idx, 0)
            self.count += 1
        batch.append(self.data[idx])
        return self.buffer.collate_batch(batch)
        
    
    


        
class PrioritizedReplayBuffer(ReplayBuffer):
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
        
    def construct_sampler(self, data, batch_size):
        return torch.utils.data.sampler.WeightedRandomSampler(self.weights, batch_size, replacement=True)
    
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
        


        

class ReplayBufferOld:
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

