from a3c.src.algorithms.A3C import A3C
from a3c.src.configParser.A3C import ConfigParser
from a3c.src.policies.A3C import MLPPolicy
from a3c.src.vectorizedEnv.A3C import vectorizeGym
from a3c.src.sharedOptimizers.A3C import sharedAdam

import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import gym
import sys
import torch
import multiprocessing as mp

def buildSharedDict(keys):
    k = {}
    for key in keys:
        k[key] = torch.DoubleTensor([0]).share_memory_()
    return k

if __name__ == "__main__":

    # Set Multiprocessing to Spawn, Required for sharing cuda tensors in shared memory
    if(sys.version_info[0] > 2):
        mp.set_start_method("spawn")
    
    config_Path = "sample_configuration.yml"
    args = ConfigParser(config_Path)
    #args.printConfig()

    # Create Object to manage vectorized Environments
    vec_env = vectorizeGym(args.getValue("env_name"), args.getValue("env_processes"))
    
    # Set Seed
    torch.manual_seed(args.getValue("pySeed"));
    
    # Build Shared Model
    shared_model = MLPPolicy(vec_env.obs_size, vec_env.num_actions, hiddens=args.getValue("policy_hiddens"))
    #shared_model.getMLPInfo()

    # Shared Optimizer
    shared_optimizer = sharedAdam(shared_model.parameters(), lr=args.getValue("learning_rate"))
    
    # Shared Data Dict
    logs_keys = ['run_epr', 'run_loss', 'episodes', 'frames']
    info_dict = buildSharedDict(logs_keys)
    #print(info_dict)

    vec_env.train(args, info_dict, shared_model, shared_optimizer)


