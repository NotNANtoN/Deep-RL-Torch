import minerl
import gym

data = minerl.data.make(
    'MineRLObtainDiamond-v0',
    data_dir='data')

# Iterate through a single epoch gathering sequences of at most 32 steps
for current_state, action, reward, next_state, done \
    in data.sarsd_iter(
        num_epochs=1, max_sequence_len=1):

        print(current_state)
        print()
        print(action)
