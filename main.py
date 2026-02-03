"""
***Implementation of Deep Q-Network (DQN) for Flappy Bird using PyTorch and Gymnasium***

Notes:
    Remove slots=True in Experience dataclass if using Python version < 3.10


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
from test import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing

test_mode = False  # Set to True to run in test mode (no training)
test_model_number = 64000
test_num_episodes = 100  # Number of episodes to run in test mode

test_model_path = f"saved_models/dqn_model_episode_{test_model_number}.pth"

# Hyperparameters

hyperparams = {
    # Learning parameters
    "hidden_dim": 128,
    "batch_size": 64,
    "epsilon": 1.0, # Probability of choosing a random action
    "epsilon_decay": 0.999,
    "epsilon_min": 0.01,
    "num_episodes": 1000000,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "target_update_freq": 1000,
    "replay_buffer_size": 50000,

    # Custom reward parameters
    "r_death": -1.0,
    "r_top": -0.5,
    "r_alive": 0.1,
    "r_pipe": 2.0,

    # Cosmetic/Logging parameters
    "EpisodeRewardDisplayFreq": 100,
    "ModelSaveFreq": 2000
}

# Features extraction
def extract_features(observation):
    """
    Defines which features to extract from the observation space. 
    Refer to the observation space description above.
    """
    states = np.array([
                observation[3]/288.0, 
                (observation[4] - observation[9])/512.0,  
                (observation[5] - observation[9])/512.0,
                (observation[8] - observation[9])/512.0,
                observation[10]/10.0
            ], dtype=np.float32)
    return states

# Initialize DQN and Replay Buffer
input_dim = 5 # Update based on extracted features
output_dim = 2

dqn = BirdDQN(input_dim, output_dim, hidden_dim=hyperparams["hidden_dim"]).to(device)
target_dqn = BirdDQN(input_dim, output_dim, hidden_dim=hyperparams["hidden_dim"]).to(device)
target_dqn.load_state_dict(dqn.state_dict()) # Initializing target DQN with same weights

optimizer = optim.Adam(dqn.parameters(), lr=hyperparams["learning_rate"])

replay_buffer = ReplayBuffer(max_size=hyperparams["replay_buffer_size"])

# Create the environment
render = "human" if test_mode else None
env = gym.make("FlappyBird-v0", use_lidar=False, render_mode=render)

if test_mode:
    dqn.load_state_dict(torch.load(test_model_path, map_location=device))
    dqn.eval()
    test(env, device, dqn, test_num_episodes, extract_features)
else:
    train(env, device, dqn, target_dqn, replay_buffer, optimizer, extract_features, hyperparams)

env.close()