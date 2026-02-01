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
import gymnasium as gym
import flappy_bird_gymnasium
from networks import BirdDQN, ReplayBuffer
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters

hyperparams = {
    "batch_size": 64,
    "epsilon": 0.1, # Probability of choosing a random action
    "epsilon_decay": 0.999,
    "epsilon_min": 0.01,
    "num_episodes": 1000,
    "learning_rate": 1e-4,
    "gamma": 0.99
}


# Initialize DQN and Replay Buffer
input_dim = 5
output_dim = 2
dqn = BirdDQN(input_dim, output_dim).to(device)
replay_buffer = ReplayBuffer(max_size=50000)

# Create the environment
env = gym.make("FlappyBird-v0", render_mode="human")



env.close()