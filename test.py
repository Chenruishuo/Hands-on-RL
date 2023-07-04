import gym
import numpy as np
import torch
lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode="ansi")
np.random.seed(0)
env.reset(seed=0)
torch.manual_seed(0)
state=env.reset()
