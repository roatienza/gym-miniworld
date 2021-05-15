import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGActor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
                super(DDPGActor, self).__init__()

                self.l1 = nn.Linear(state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, action_dim)
                
                self.max_action = max_action

        
        def forward(self, state):
                a = F.relu(self.l1(state))
                a = F.relu(self.l2(a))
                a = F.softmax(self.l3(a))
                #a = self.max_action * F.softplus(a)
                # a = self.max_action * torch.tanh(self.l3(a))
                #a = torch.argmax(a)
                return a

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
                
        self.max_action = max_action

        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    #def select_action(self, state):
    #    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    #    action = self.actor(state).cpu().data.numpy().flatten()
