import gym
import numpy as np
import torch.multiprocessing as mp

class vectorizeGym:
    def __init__(self, env_name):
        self.env_name = env_name
        self.local_env = gym.make(env_name)
        self.obs_size = np.array(self.local_env.observation_space.low).flatten().shape[0]
        self.num_actions = self.local_env.action_space.n
    
    def createEnv(self):
        return NormalizedEnv(gym.make(self.env_name))


class NormalizedEnv(gym.ObservationWrapper):
    '''
    Allows to apply transformation to the observation.
    A tutorial on wrapping different components of the Gym Env can be found here.
    Link : https://hub.packtpub.com/openai-gym-environments-wrappers-and-monitors-tutorial/
    
    Have a look at this file to understand Action, Reward, Observation Wrapper
    Link : https://github.com/openai/gym/blob/master/gym/core.py

    '''

    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0.0
        self.state_std = 0.0
        self.alpha = 0.9999
        self.num_stps = 0
    
    def observation(self, observation):
        self.num_stps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_stps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_stps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)