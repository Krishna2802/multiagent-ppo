import numpy as np
import torch
import os

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()

def compute_gae(rewards, state_values, is_terminals, gamma=0.99, lam=0.95):
    """
    Computes Generalized Advantage Estimation (GAE).
    """
    advantages = []
    gae = 0
    
    # Append 0 for terminal next_value
    values = state_values + [0]
    
    for i in reversed(range(len(rewards))):
        mask = 1.0 - is_terminals[i]
        delta = rewards[i] + gamma * values[i + 1] * mask - values[i]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(state_values, dtype=torch.float32)
    
    # Normalize advantages for stabler PPO updates
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
    return advantages, returns

def get_batches(buffer_size, batch_size):
    """
    Generates shuffed mini-batch indices.
    """
    indices = np.arange(buffer_size)
    np.random.shuffle(indices)
    for start in range(0, buffer_size, batch_size):
        end = start + batch_size
        yield indices[start:end]
