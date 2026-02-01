"""
Observation Space from env.step(action):
    0: the last pipe's horizontal position
    1: the last top pipe's vertical position
    2: the last bottom pipe's vertical position
    3: the next pipe's horizontal position
    4: the next top pipe's vertical position
    5: the next bottom pipe's vertical position
    6: the next next pipe's horizontal position
    7: the next next top pipe's vertical position
    8: the next next bottom pipe's vertical position
    9: player's vertical position
    10: player's vertical velocity
    11: player's rotation

Reward:
    +0.1 - every frame it stays alive
    +1.0 - successfully passing a pipe
    -1.0 - dying
    -0.5 - touch the top of the screen

Action Space:
    0: do nothing
    1: flap

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
from networks import BirdDQN, ReplayBuffer
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing
test_mode = False  # Set to True to run in test mode (no training)
test_model_number = 12000
test_num_episodes = 100  # Number of episodes to run in test mode

test_model_path = f"models/dqn_model_episode_{test_model_number}.pth"



# Hyperparameters

hyperparams = {
    # Learning parameters
    "batch_size": 128,
    "epsilon": 1.0, # Probability of choosing a random action
    "epsilon_decay": 0.9995,
    "epsilon_min": 0.01,
    "num_episodes": 10000,
    "learning_rate": 1e-4,
    "gamma": 0.999,
    "target_update_freq": 1000,

    # Cosmetic/Logging parameters
    "EpisodeRewardDisplayFreq": 100,
    "ModelSaveFreq": 2000
}


# Initialize DQN and Replay Buffer
input_dim = 5
output_dim = 2

dqn = BirdDQN(input_dim, output_dim).to(device)
target_dqn = BirdDQN(input_dim, output_dim).to(device)
target_dqn.load_state_dict(dqn.state_dict()) # Initializing target DQN with same weights

optimizer = optim.Adam(dqn.parameters(), lr=hyperparams["learning_rate"])

replay_buffer = ReplayBuffer(max_size=50000)

# Create the environment
render = "human" if test_mode else None
env = gym.make("FlappyBird-v0", use_lidar=False, render_mode=render)

if test_mode:
    dqn.load_state_dict(torch.load(test_model_path, map_location=device))
    dqn.eval()
    for ep in range(test_num_episodes):
        observation, info = env.reset()
        done = False
        while not done:
            state = observation[[3, 4, 5, 9, 10]]
            state_tensor = torch.as_tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = dqn(state_tensor)
            
            action = torch.argmax(q_values).item()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
else:
    train(env, device, dqn, target_dqn, replay_buffer, optimizer, hyperparams)

env.close()