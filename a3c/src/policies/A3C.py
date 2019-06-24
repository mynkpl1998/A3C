import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvMLPPolicy(nn.Module):

    def __init__(self, channels, num_actions, memsize=256):
        super(ConvMLPPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize)
        self.linear_critic = nn.Linear(memsize, 1)
        self.linear_actor = nn.Linear(memsize, num_actions)


class MLPPolicy(nn.Module):

    def __init__(self, obs_size, num_actions, memsize=128, hiddens=[128]):
        self.obs_size = obs_size
        self.memsize = memsize

        super(MLPPolicy, self).__init__()
        self.fc_dict = {}

        if(len(hiddens) <= 0):
            raise ValueError("there should be at least one hidden layer in the policy")

        for i, size in enumerate(hiddens):
            layer_name = "fc_%d"%(i+1)
            if (i == 0):
                self.fc_dict[layer_name] = nn.Linear(obs_size, hiddens[i])
            else:
                self.fc_dict[layer_name] = nn.Linear(hiddens[i-1], hiddens[i])               

        self.gru = nn.GRUCell(hiddens[-1], memsize)
        self.linear_critic = nn.Linear(memsize, 1)
        self.linear_actor = nn.Linear(memsize, num_actions)

    def getMLPInfo(self):
        layer_count = 0
        
        # Fully Connected
        for layer in self.fc_dict:
            print("FC Layer %d - "%(layer_count+1), self.fc_dict[layer])
            layer_count += 1
        
        # Memory
        print("GRU - ", self.gru)
        
        # Linear Actor and Critic
        print("Critic Layer - ", self.linear_critic)
        print("Actor Layer - ", self.linear_actor)
        
    def forward(self, observation, history):
        '''
        Function : Passes the observation through fully connected layer followed by
        a recurrent layer, then calculates the value function (critic) and logits (actor)
        
        Arguments
        observation : PyTorch Tensor of type (batch, obs_size)
        history : PyTorch Tensor of type (batch, memsize)

        '''

        self.fc_out = None
        for i, layer in enumerate(self.fc_dict):
            if(i == 0):
                self.fc_out = F.elu(self.fc_dict[layer](observation))
            else:
                self.fc_out = F.elu(self.fc_dict[layer](self.fc_out))

        self.hx_out = self.gru(self.fc_out, history)
        return self.linear_critic(self.hx_out), self.linear_actor(self.hx_out), self.hx_out