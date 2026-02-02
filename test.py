"""
Testing script 
"""

import torch
import torch.optim as optim
import random as rd
import numpy as np
from networks import BirdDQN, ReplayBuffer

def test(env, device, dqn, test_num_episodes):
    for ep in range(test_num_episodes):
        observation, info = env.reset()
        done = False
        while not done:
            state = np.array([
                observation[3], 
                observation[4] - observation[9],  
                observation[5] - observation[9], 
                observation[10]
            ], dtype=np.float32) # We use relative positions to generalize better
            state_tensor = torch.as_tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = dqn(state_tensor)
            
            action = torch.argmax(q_values).item()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated