import torch
import numpy as np
from env import make_env
from model import Actor
import time

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def evaluate():
    set_seed(42)
    print("Loading Evaluation Environment...")
    env = make_env(render_mode="human")
    
    dummy_agent = env.possible_agents[0]
    obs_dim = env.observation_space(dummy_agent).shape[0]
    act_dim = env.action_space(dummy_agent).n
    
    actor_net = Actor(obs_dim, act_dim)
    
    try:
        actor_net.load_state_dict(torch.load("models/actor_final.pth"))
        actor_net.eval()
        print("Successfully loaded trained models/actor_final.pth weights")
    except FileNotFoundError:
        try:
             actor_net.load_state_dict(torch.load("models/actor.pth"))
             actor_net.eval()
             print("Successfully loaded trained models/actor.pth weights")
        except FileNotFoundError:
            print("WARNING: Model weights not found. Using untrained policy.")

    print("Starting Evaluation Episode...")
    obs_dict, _ = env.reset()
    
    while env.agents:
        active_agents = env.agents
        
        obs_list = [obs_dict[agent] for agent in active_agents]
        obs_tensor = torch.FloatTensor(np.array(obs_list))
        
        with torch.no_grad():
            dist = actor_net(obs_tensor)
            actions = torch.argmax(dist.probs, dim=-1)
            
        action_dict = {agent: actions[i].item() for i, agent in enumerate(active_agents)}

        next_obs_dict, rewards_dict, terminations_dict, truncations_dict, _ = env.step(action_dict)
        
        time.sleep(0.05)
        obs_dict = next_obs_dict

    env.close()
    print("Evaluation Complete.")

if __name__ == '__main__':
    evaluate()
