import minerl
import gym
import logging
import torch
from gym.envs.classic_control import rendering
import numpy as np


logging.basicConfig(level=logging.DEBUG)

env = gym.make('MineRLNavigateDense-v0')

viewer = rendering.SimpleImageViewer()

obs, _ = env.reset()
done = False
net_reward = 0

while not done:

    #image = torch.tensor(obs['pov'], dtype=torch.float)
    #output = torch.from_numpy(obs['pov'])#dtype=torch.int)
    #scaled_image = torch.nn.functional.interpolate(output, scale_factor = 3)
    #torch.from_numpy(np.flip(obs['pov'], axis=0))
    #output = m(obs['pov'])

    action = env.action_space.noop()

    action['camera'] = [0, 0.03*obs["compassAngle"]]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 1

    obs, reward, done, info = env.step(
        action)

    net_reward += reward
    print("Total reward: ", net_reward)




    #viewer.imshow(scaled_image)


    if done:
        break



