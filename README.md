## Multi-Agent Reinforcement Learning with PPO in Simple Spread

This project implements a Multi-Agent Reinforcement Learning (MARL) system where multiple agents learn to cooperate in a shared environment using Proximal Policy Optimization (PPO) with an Actor-Critic architecture.

The agents are trained in the PettingZoo Multi-Agent Particle Environment (MPE) task `simple_spread_v3`, where agents must spread out and cover landmarks efficiently while avoiding collisions.

### Requirements

- Python 3.10+
- PyTorch
- PettingZoo
- Gymnasium
- NumPy
- Matplotlib
- TQDM

You can install the dependencies via:
```bash
pip install torch pettingzoo[mpe] gymnasium numpy matplotlib tqdm
```

### Architecture

- **`env.py`**: Initializes the PettingZoo `simple_spread_v3` parallel environment.
- **`model.py`**: Defines the PyTorch Actor (policy) and Critic (value function) Neural Networks. Let Observation dimension be $O$ and Action dimension be $A$. Both networks use 2 hidden layers of 128 units.
- **`utils.py`**: Includes the PPO Rollout Buffer for storing transitions and advantage estimation. Contains plotting utilities.
- **`train.py`**: The main training loop driving environment interaction, calculating rewards, returns, and performing policy gradient updates.
- **`evaluate.py`**: Script to visually render trained agents demonstrating emergent cooperation.

### Usage

**Training**:
```bash
python train.py
```
This will train the models for a default number of episodes and save the weights and training plots.

**Evaluation**:
```bash
python evaluate.py
```
Loads the saved models and renders the multi-agent coordination within the environment.
