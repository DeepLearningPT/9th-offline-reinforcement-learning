# %%
import argparse
import gym
import numpy as np
import os
import torch

import BCQ
import DDPG
import utils

from gym.wrappers import Monitor
# %%
ENV_NAME = "LunarLanderContinuous-v2"
SEED = 0
# %%
env = gym.make(ENV_NAME)
env = Monitor(env, 'videos/', force=True)
# %%
env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
# %%
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])
# %%
# Loading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bcq = BCQ.BCQ(
    state_dim,
    action_dim,
    max_action,
    device,
    0.99,
    0.005,
    0.75,
    0.05
)

bcq.load(f"./models/bcq_{ENV_NAME}_{SEED}")
ep = 0
# %%
MAX_EP = 10
# %%
avg_reward = 0.
for _ in range(MAX_EP):
    state, done = env.reset(), False
    while not done:
        ac = bcq.select_action(state)
        state, reward, done, _ = env.step(ac)
        avg_reward += reward
avg_reward /= MAX_EP

print(f"AVG Reward = {avg_reward}")

# %%
