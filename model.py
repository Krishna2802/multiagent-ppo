import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    """
    Actor Network: Parameterizes the policy \pi_theta(a | s).
    Outputs a discrete categorical probability distribution over the action space.
    """
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
        """
        Calculates action probabilities and returns a categorical distribution.
        """
        action_probs = self.net(obs)
        dist = Categorical(action_probs)
        return dist

class Critic(nn.Module):
    """
    Critic Network: Parameterizes the state value function V_phi(s).
    Estimates the expected returns from the current state.
    """
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
        """
        Outputs a scalar value estimate for the given state.
        """
        value = self.net(obs)
        return value.squeeze(-1) # Output shape: (batch_size,) 

if __name__ == "__main__":
    # Simple validation using dummy variables
    dummy_obs = torch.randn(2, 18) # 2 instances, 18 features (standard for simple_spread agent)
    act_dim = 5
    actor = Actor(18, act_dim)
    critic = Critic(18)
    
    dist = actor(dummy_obs)
    values = critic(dummy_obs)
    
    print("Actor generated actions:", dist.sample())
    print("Critic calculated values:", values.shape, "->", values.detach().numpy())
