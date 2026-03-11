import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from env import make_env
from model import Actor, Critic
from utils import RolloutBuffer, compute_gae, get_batches
import numpy as np
import os

# Hyperparameters
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
K_EPOCHS = 10
EPS_CLIP = 0.2
BATCH_SIZE = 64
MAX_EPISODES = 2000
MAX_CYCLES = 25
SAVE_FREQ = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class PPOAgent:
    def __init__(self, obs_dim, act_dim):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.buffer = RolloutBuffer()

    def update(self):
        # Calculate GAE
        state_values_list = [v.item() for v in self.buffer.state_values]
        advantages, returns = compute_gae(
            self.buffer.rewards, 
            state_values_list, 
            self.buffer.is_terminals, 
            GAMMA, 
            GAE_LAMBDA
        )
        
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        old_states = torch.stack(self.buffer.states).detach()
        old_actions = torch.stack(self.buffer.actions).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).detach()
        old_state_values = torch.stack(self.buffer.state_values).detach()
        
        buffer_size = len(self.buffer.states)
        total_actor_loss, total_critic_loss, total_entropy = 0, 0, 0
        updates = 0
        
        # Optimize policy for K epochs using Mini-Batches
        for _ in range(K_EPOCHS):
            for batch_indices in get_batches(buffer_size, BATCH_SIZE):
                b_states = old_states[batch_indices]
                b_actions = old_actions[batch_indices]
                b_old_logprobs = old_logprobs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]
                b_old_values = old_state_values[batch_indices]
                
                dist = self.actor(b_states)
                logprobs = dist.log_prob(b_actions)
                dist_entropy = dist.entropy()
                state_values = self.critic(b_states).squeeze(-1)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - b_old_logprobs)
                
                # Actor Surrogate Loss
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * b_advantages
                loss_actor = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
                
                # Critic Clipped Loss
                v_clipped = b_old_values + torch.clamp(state_values - b_old_values, -EPS_CLIP, EPS_CLIP)
                loss_critic_unclipped = (state_values - b_returns) ** 2
                loss_critic_clipped = (v_clipped - b_returns) ** 2
                loss_critic = 0.5 * torch.max(loss_critic_unclipped, loss_critic_clipped).mean()
                
                # Take gradient steps
                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                self.optimizer_actor.step()
                
                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                self.optimizer_critic.step()
                
                total_actor_loss += loss_actor.item()
                total_critic_loss += loss_critic.item()
                total_entropy += dist_entropy.mean().item()
                updates += 1

        self.buffer.clear()
        return total_actor_loss / updates, total_critic_loss / updates, total_entropy / updates

def train():
    set_seed(42)
    env = make_env(max_cycles=MAX_CYCLES)
    
    dummy_agent = env.possible_agents[0]
    obs_dim = env.observation_space(dummy_agent).shape[0]
    act_dim = env.action_space(dummy_agent).n
    
    ppo_agent = PPOAgent(obs_dim, act_dim)
    writer = SummaryWriter(log_dir="runs/ppo_simple_spread")
    
    os.makedirs("models", exist_ok=True)
    print("Starting Training...")
    history_rewards = []
    
    for episode in range(1, MAX_EPISODES + 1):
        obs_dict, _ = env.reset()
        episode_reward = 0
        
        while env.agents:
            active_agents = env.agents
            
            # Vectorized GPU inference
            obs_list = [obs_dict[agent] for agent in active_agents]
            obs_tensor = torch.FloatTensor(np.array(obs_list)).to(device)
            
            with torch.no_grad():
                dist = ppo_agent.actor(obs_tensor)
                values = ppo_agent.critic(obs_tensor)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
            
            action_dict = {agent: actions[i].item() for i, agent in enumerate(active_agents)}
            next_obs_dict, rewards_dict, terminations_dict, truncations_dict, _ = env.step(action_dict)
            
            # Aligned Tuple Storage by Agent
            for i, agent in enumerate(active_agents):
                reward = rewards_dict[agent]
                done = terminations_dict[agent] or truncations_dict[agent]
                
                ppo_agent.buffer.states.append(obs_tensor[i])
                ppo_agent.buffer.actions.append(actions[i])
                ppo_agent.buffer.logprobs.append(logprobs[i])
                ppo_agent.buffer.state_values.append(values[i])
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)
                
                episode_reward += reward
                
            obs_dict = next_obs_dict
        
        # PPO epoch updates
        a_loss, c_loss, entropy = ppo_agent.update()
        
        history_rewards.append(episode_reward)
        
        # TensorBoard Logging
        writer.add_scalar("Loss/Actor", a_loss, episode)
        writer.add_scalar("Loss/Critic", c_loss, episode)
        writer.add_scalar("Loss/Entropy", entropy, episode)
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        
        if episode % 100 == 0:
            avg_reward = np.mean(history_rewards[-100:])
            print(f"Episode {episode} \t Avg Reward: {avg_reward:.2f} \t Actor Loss: {a_loss:.4f} \t Critic Loss: {c_loss:.4f}")
            
        if episode % SAVE_FREQ == 0:
            torch.save(ppo_agent.actor.state_dict(), f"models/actor_ep{episode}.pth")
            torch.save(ppo_agent.critic.state_dict(), f"models/critic_ep{episode}.pth")
            
    torch.save(ppo_agent.actor.state_dict(), "models/actor_final.pth")
    torch.save(ppo_agent.critic.state_dict(), "models/critic_final.pth")
    writer.close()
    
    print("Training complete. Models saved. View logs with: tensorboard --logdir runs")
    env.close()

if __name__ == '__main__':
    train()
