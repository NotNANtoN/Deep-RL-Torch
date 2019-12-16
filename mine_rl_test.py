import gym
import minerl
from env_wrappers import SerialDiscreteActionWrapper

# Run a random agent through the environment
env = gym.make("MineRLObtainDiamondDense-v0") # A MineRLObtainDiamondDense-v0 env

env = SerialDiscreteActionWrapper(env)
print(env.action_space)
print(env.action_space.n)

obs = env.reset()
done = False

while not done:
    # Take a no-op through the environment.
    action = env.action_space.sample()
    print(action)
    obs, rew, done, _ = env.step(action)
    # Do something