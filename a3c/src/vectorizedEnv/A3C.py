from a3c.src.common.A3C import single_env_train

import gym
import numpy as np
import multiprocessing as mp

class vectorizeGym:

    def __init__(self, env_name, num_envs):
        self.local_env = gym.make(env_name)
        self.num_env = num_envs
        self.obs_size = np.array(self.local_env.observation_space.low).flatten().shape[0]
        self.num_actions = self.local_env.action_space.n

        self.processes = []

    def train(self, arguments, info, shared_model):
        
        for rank in range(self.num_env):
            proc = mp.Process(target=single_env_train, args=(arguments, rank, self.obs_size, self.num_actions, info, shared_model))
            proc.start()
            self.processes.append(proc)
        
        for p in self.processes: p.join()


