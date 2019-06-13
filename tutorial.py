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
print(obs)

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    #image = torch.tensor(obs['pov'], dtype=torch.float)
    output = torch.from_numpy(obs['pov'])#dtype=torch.int)
    scaled_image = torch.nn.functional.interpolate(output, scale_factor = 3)
    #torch.from_numpy(np.flip(obs['pov'], axis=0))
    #output = m(obs['pov'])




    viewer.imshow(scaled_image)


    if done:
        break



