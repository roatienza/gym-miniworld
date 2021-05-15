import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGActor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action, hidden_dim=32):
                super(DDPGActor, self).__init__()

                self.l1 = nn.Linear(state_dim, hidden_dim)
                self.l2 = nn.Linear(hidden_dim, hidden_dim)
                self.l3 = nn.Linear(hidden_dim, action_dim)
                
                self.max_action = max_action

        
        def forward(self, state):
                a = F.relu(self.l1(state))
                a = F.relu(self.l2(a))
                a = self.l3(a)
                #a = F.softmax(self.l3(a), dim=-1)
                return a

