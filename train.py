from a3c.src.algorithms.A3C import A3C
from a3c.src.configParser.A3C import ConfigParser
from a3c.src.policies.A3C import MLPPolicy
from a3c.src.vectorizedEnv.A3C import vectorizeGym

import numpy as np
import gym

import torch

if __name__ == "__main__":

    config_Path = "sample_configuration.yml"
    args = ConfigParser(config_Path)
    #args.printConfig()
    
    env = vectorizeGym(args.getValue("env_name"), args.getValue("env_processes"))

    policy = MLPPolicy(env.obs_size, env.num_actions, hiddens=args.getValue("policy_hiddens"))
    
    sample_observation = torch.randn(1, env.obs_size)
    hx = torch.randn(1, args.getValue("memsize"))

    value, act_logits, hx = policy.forward(sample_observation, hx)