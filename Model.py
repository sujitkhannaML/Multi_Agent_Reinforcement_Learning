import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparams import *

def hidden_init(layer):
    wts = layer.weight.data.size()[0]
    rng = 1./np.sqrt(wts)
    return (-rng, rng)

class Actor(nn.Module):

    def __init__(self, input_dim, output_dim, seed=10, fc1_units=ACTOR_FC1_UNITS, fc2_units=ACTOR_FC2_UNITS):
        super(Actor, self).__init__()
        ##Define the entire Actor network architecture in this class
        self.seed = torch.manual_seed(seed)
        self.nonlin = NON_LIN

        #Dense Layer
        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_dim)

        #Normalization layer
        self.bn1 = nn.BatchNorm1d(fc1_units)
        #self.bn2 = nn.BatchNorm1d(fc2_units)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-(3e-3), 3e-3)

    def forward(self, state):
        '''
        This function builds up the actor network
        :return:
        '''
        if state.dim()==1:
            state = torch.unsqueeze(state, 0) #reshaaping the state to comply with batch norm
        h1 = self.nonlin(self.fc1(state))
        h1 = self.bn1(h1)
        h2 = self.nonlin(self.fc2(h1))
        return F.tanh(self.fc3(h2))


class Critic(nn.Module):

    def __init__(self, input_dim, action_size, seed=10, fcs1_units=CRITIC_FCS1_UNITS, fc2_units=CRITIC_FC2_UNITS):
        super(Critic, self).__init__()
        ##Define the entire network architecture in this class
        self.seed = torch.manual_seed(seed)
        self.nonlin = NON_LIN

        self.fcs1 = nn.Linear(input_dim + action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        #Normalization layers
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-(3e-3), 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        # Reshape the state to comply with Batch Normalization
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        # Modified DDPG architecture
        xs = torch.cat((state, action.float()), dim=1)
        x = self.nonlin(self.fcs1(xs))
        x = self.bn1(x) # Batch Normalization after Activation

        x = self.nonlin(self.fc2(x))
        return self.fc3(x)








