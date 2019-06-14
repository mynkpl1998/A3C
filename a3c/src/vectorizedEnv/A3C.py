import gym
import numpy as np

class vectorizeGym:

    def __init__(self, env_name, num_envs):
        self.local_env = gym.make(env_name)
        self.num_env = num_envs
        self.obs_size = np.array(self.local_env.observation_space.low).flatten().shape[0]
        self.num_actions = self.local_env.action_space.n

