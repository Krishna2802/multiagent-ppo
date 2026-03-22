from pettingzoo.mpe import simple_spread_v3

def make_env(render_mode=None, max_cycles=25):

    env = simple_spread_v3.parallel_env(
        N=3, 
        local_ratio=0.5, 
        max_cycles=max_cycles, 
        continuous_actions=False, 
        render_mode=render_mode
    )
    return env

if __name__ == "__main__":
    # Quick test to instantiate environment
    env = make_env()
    env.reset()
    print(f"Agents: {env.agents}")
    for agent in env.agents:
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)
        print(f"Agent {agent} | Obs space: {obs_space} | Act space: {act_space}")
    env.close()
