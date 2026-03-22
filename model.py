import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
  
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        
        action_probs = self.net(obs)
        dist = Categorical(action_probs)
        return dist

class Critic(nn.Module):
    
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
       
        value = self.net(obs)
        return value.squeeze(-1)  

if __name__ == "__main__":
   
    dummy_obs = torch.randn(2, 18) # 2 instances, 18 features 
    act_dim = 5
    actor = Actor(18, act_dim)
    critic = Critic(18)
    
    dist = actor(dummy_obs)
    values = critic(dummy_obs)
    
    print("Actor generated actions:", dist.sample())
    print("Critic calculated values:", values.shape, "->", values.detach().numpy())
