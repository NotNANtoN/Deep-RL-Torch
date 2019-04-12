#from gym_aigar.gym_aigar.envs.aigarPellet import aigarPelletEnv
import gym
import gym_aigar


#env = gym.make("AigarPellet-v0")
env = gym.make("AigarGreedy-v0")
#env = aigarPelletEnv()

obs = env.reset()
for i in range(2000):
  action = env.action_space.sample()
  #print("Action: ", action)
  obs, reward, done, info = env.step(action)
  if reward > 0:
    print("Reward: ", reward)
  #print("Obs: ", obs)
  #print()
  env.render(mode = 'human')
  #rgb_array = env.render(mode = "rgb_array")cd 
