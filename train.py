from a3c.src.algorithms.A3C import A3C
from a3c.src.configParser.A3C import ConfigParser
from a3c.src.policies.A3C import MLP
from a3c.src.vectorizedEnv.A3C import vectorizeGym
from a3c.src.sharedOptimizers.A3C import sharedAdam
from a3c.src.common.A3C import train_process, test_process

import os
import gym
import sys
import torch
import torch.multiprocessing as mp

def buildSharedDict(keys):
    k = {}
    for key in keys:
        k[key] = torch.DoubleTensor([0]).share_memory_()
    return k

if __name__ == "__main__":

    # Set Multiprocessing to Spawn, Required for sharing cuda tensors in shared memory
    if(sys.version_info[0] > 2):
        mp.set_start_method("spawn")
    
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    
    config_Path = "sample_configuration.yml"
    args = ConfigParser(config_Path)
    #args.printConfig()

    # Create Object to manage vectorized Environments
    vec_env = vectorizeGym(args.getValue("env_name"))
    
    
    # Set Torch Seed
    torch.manual_seed(args.getValue("torchSeed"))
    
    # Build Shared Model
    shared_model = MLP(vec_env.obs_size, vec_env.num_actions).share_memory()
    
    # Shared Optimizer
    shared_optimizer = sharedAdam(shared_model.parameters(), lr=args.getValue("learning_rate"))
    shared_optimizer.share_memory()

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    processes = []
    
    # Start a test Process
    
    p = mp.Process(target=test_process, args=(args.getValue("env_processes"), args, shared_model, counter, vec_env))
    p.start()
    processes.append(p)
    
    
    # Start Training
    for rank in range(0, args.getValue('env_processes')):
        p = mp.Process(target=train_process, args=(rank, args, shared_model, counter, lock, shared_optimizer, vec_env))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()