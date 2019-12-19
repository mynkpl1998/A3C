import os
import shutil
import importlib
import time
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from a3c.src.policies.A3C import cnnPolicy
from flashtorch.utils import apply_transforms, denormalize, format_for_plotting
from flashtorch.saliency import Backprop
from huepy import bold, bad, red, blue, info

def warnMessage(msgStr):
    print(bad(blue(red(msgStr))))

def infoMessage(msgStr):
    print(info(bold(blue(msgStr))))

def checkFileExists(path):
    return os.path.exists(path)
    
def logEssentials(dir_path, exp_name):

    # Create directory first
    if not os.path.exists(dir_path+exp_name):
        os.mkdir(dir_path+exp_name)
    else:
        warnMessage("Do you want to remove existing experiment logs and data ? [Y/y]")
        res = input()
        if res == 'Y' or res == 'y':
            shutil.rmtree(dir_path+exp_name)
            os.mkdir(dir_path+exp_name)
        else:
            return True
    
    os.mkdir(dir_path+exp_name+"/logs")
    os.mkdir(dir_path+exp_name+"/saved_models")
    return False

def launchTensorboard(launch_dir):
    launchCmd = "tensorboard --logdir "+launch_dir
    infoMessage("Starting tesorboard at localhost:6006")
    os.system(launchCmd)
    
def importCustomEnv(moduleName):
    importlib.import_module(moduleName)

class policy(cnnPolicy):
    def __init__(self, obs_size, action_space, hidden, memsize):
        super(policy, self).__init__(obs_size, action_space, hidden, memsize)
        self.memory = None
    
    def forward(self, inputs):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.view(1, 32 * 6 * 6)
        self.memory = self.gru(x, (self.memory))
        x = self.memory
        return self.actor_linear(x)


class saliencyMaps:

    def __init__(self, model):
        self.origModel = model
        self.model = policy(self.origModel.obs_size, self.origModel.action_space, self.origModel.hidden, self.origModel.memsize)
        self.model.load_state_dict(self.origModel.state_dict())
        self.model.train()
        self.createHooks()
        self.gradients = {}
    
    def createHooks(self,):
        #print(self.model)
        firstLayer = self.model.conv1
        firstLayer.register_backward_hook(self.hook_function)
    
    def hook_function(self, module, grad_in, grad_out):
        self.gradients['data'] = grad_in[0]
    
    def forward(self, inputImg):
        '''
        inputImg: The input image should be a torch tensor
        '''
        img, hx = inputImg
        # Should have atleast four channels (B, C, H, W)
        assert len(img.size()) == 4
        
        numpyImg = img[0].transpose(2, 0).numpy().reshape(80, 80)
        value, logit, h = self.origModel.forward(inputImg)
        targetClass = torch.argmax(logit, axis=1).item()


        backprop = Backprop(self.model)
        self.model.memory = hx
        gradients = backprop.calculate_gradients(img, targetClass, guided=True)
        #maxGradients = backprop.calculate_gradients(img, targetClass, take_max=True)
        