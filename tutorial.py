import minerl
import gym
import logging
import torch
from gym.envs.classic_control import rendering
import numpy as np
#from skimage.transform import resize
from torchvision.transforms.functional import resize
from torch.nn import functional as F

#logging.basicConfig(level=logging.DEBUG)



viewer = rendering.SimpleImageViewer()


#adaptive_pool = nn.AdaptiveAvgPool3d((512, 512, 3))
scale = 3

env = gym.make('MineRLNavigateDense-v0')
obs, _ = env.reset()
done = False


while not done:
    #action = env.action_space.sample()
    action = env.action_space.noop()
    obs, reward, done, _ = env.step(action)

    output = obs['pov'] #torch.from_numpy(obs['pov'])#dtype=torch.int)
    print("start shape: ", output.shape)
    
    scaled_image = torch.from_numpy(output).type('torch.DoubleTensor')
                                 
    scaled_image = F.interpolate(scaled_image.unsqueeze(0).unsqueeze(0).view((1, 3, 1, 64, 64)), size=(512, 512, 3), mode='area')
    scaled_image = scaled_image.type('torch.IntTensor').squeeze().numpy()
    #scaled_image.resize((512, 512, 3))
    #scaled_image = resize(output, (512, 512))
    #output = output.type('torch.DoubleTensor')
    #scaled_image = torch.nn.functional.upsample(output, size=(256, 256, 3))
    
    print("scaled shape: ", scaled_image.shape)
    #scaled_image = scaled_image.reshape((512, 512, 3))
    print()
    

    #viewer.imshow(scaled_image)
    env.render()


    if done:
        break



