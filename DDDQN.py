
import numpy as np
import torch.nn as nn
from collections import deque
import random

class DuelingDQN(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(DuelingDQN,self).__init__()
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.feature=nn.Sequential(
            nn.Linear(state_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )
        self.value_stream=nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
        self.advantage_stream=nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,action_dim)
        )
    def forward(self,state):
        features=self.feature(state)
        values=self.value_stream(features)
        advantages=self.advantage_stream(features)
        qvals=values+(advantages-advantages.mean(dim=-1,keepdim=True))
        return qvals

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)
    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    def sample(self,batch_size):
        state,action,reward,next_state,done=zip(*random.sample(self.buffer,batch_size))
        return np.array(state),action,reward,np.array(next_state),done
    def __len__(self):
        return len(self.buffer)

