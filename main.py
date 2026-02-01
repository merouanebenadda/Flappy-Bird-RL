import gymnasium as gym
import flappy_bird_gymnasium

# Create the environment
env = gym.make("FlappyBird-v0", render_mode="human")

# Reset the environment
observation, info = env.reset()

for _ in range(1000):
    # this is where you would insert your policy
    action = 0 #env.action_space.sample()

    # Take a step
    observation, reward, terminated, truncated, info = env.step(action)


    if terminated or truncated:
        observation, info = env.reset()



env.close()