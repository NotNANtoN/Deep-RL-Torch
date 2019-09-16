import minerl
import gym

env = 'MineRLObtainDiamond-v0'
data = minerl.data.make(
    env,
    data_dir='data')

count = sum(1 for _ in data.sarsd_iter(max_sequence_len=1))
print("Env: ", env)
print("Count: ", count)



# Iterate through a single epoch gathering sequences of at most 32 steps
#for current_state, action, reward, next_state, done \
#    in data.sarsd_iter(
#        num_epochs=1, max_sequence_len=1):

#        print(current_state)
#        print()
#        print(action)