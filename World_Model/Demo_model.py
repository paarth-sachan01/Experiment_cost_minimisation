import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class Demo_WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Demo_WorldModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)