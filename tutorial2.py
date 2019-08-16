import minerl
import gym
import logging
import torch
from gym.envs.classic_control import rendering
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

logging.basicConfig(level=logging.DEBUG)

#env = gym.make('MineRLNavigateDense-v0')

#viewer = rendering.SimpleImageViewer()

class RenderObservations(gym.Wrapper):
    def __init__(self, env, display_vector_obs=True):
        gym.Wrapper.__init__(self, env)
        self.viewer = None

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self._visual_obs = ob["pov"].copy()
        return ob, reward, done, info

    def _renderObs(self, obs):
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(obs)
        return self.viewer.isopen

    def render(self, mode='human', **kwargs):
        if mode == 'human':
            self._renderObs(self._visual_obs)
        return self._visual_obs

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def action():
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

            obs, reward, done, info = env.step(action)

            net_reward += reward
            print("Total reward: ", net_reward)




        #viewer.imshow(scaled_image)


            if done:
                break

if __name__ == "__main__":

    env = RenderObservations(gym.make('MineRLNavigateDense-v0'))
