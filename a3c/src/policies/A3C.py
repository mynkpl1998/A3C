import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)        

class MLP(torch.nn.Module):
    
    def __init__(self, obs_size, action_space, hidden=128, memsize=256):
        super(MLP, self).__init__()

        self.hidden = hidden
        self.obs_size = obs_size
        self.action_space = action_space
        self.memsize = memsize

        self.fc1 = nn.Linear(obs_size, hidden)
        self.gru = nn.GRUCell(hidden, memsize)
        self.critic_linear = nn.Linear(memsize, 1)
        self.actor_linear = nn.Linear(memsize, action_space)

        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()
    
    def forward(self, inputs):
        inputs, hx = inputs
        x = F.elu(self.fc1(inputs))
        x = x.view(-1, self.hidden)
        hx = self.gru(x, (hx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), hx