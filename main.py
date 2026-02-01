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

import gymnasium as gym
import flappy_bird_gymnasium

# Create the environment
env = gym.make("FlappyBird-v0", render_mode="human")

# Reset the environment
observation, info = env.reset()

for _ in range(1000):
    # this is where you would insert your policy
    # (feed the observation to your agent here)
    action = 0 #env.action_space.sample()

    # Take a step
    observation, reward, terminated, truncated, info = env.step(action)
    relevant_observation = observation[[3, 4, 5, 9, 10]]


    if terminated or truncated:
        observation, info = env.reset()



env.close()