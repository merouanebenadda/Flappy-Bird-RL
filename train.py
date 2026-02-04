"""
Training script 
"""

import os
import datetime
import torch
import torch.optim as optim
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from networks import BirdDQN, ReplayBuffer

def train(env, device, dqn, target_dqn, replay_buffer, optimizer, extract_features, hyperparams):
    # Setting up logs and model directory
    model_directory_path = f"models/models{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(model_directory_path, exist_ok=True)
    hyperparams_file = open(f"{model_directory_path}/hyperparameters.txt", "w")
    for key, value in hyperparams.items():
        hyperparams_file.write(f"{key}: {value}\n")
    hyperparams_file.close()
    logs_path = f"{model_directory_path}/logs.txt"
    logs = open(logs_path, "w")

    reward_map = {
        -1.0: hyperparams.get("r_death", -1.0),
        -0.5: hyperparams.get("r_top", -0.5),
         0.1: hyperparams.get("r_alive", 0.1),
         1.0: hyperparams.get("r_pipe", 1.0)
    }

    step = 0
    rewards_history = []
    average_reward_history = []
    average_score_history = []
    best_average_score = -float('inf')
    score_history = []

    print("Starting training...")
    # Playing loop
    for episode in range(1, hyperparams["num_episodes"]+1):
        observation, info = env.reset()
        state = extract_features(observation)

        episode_reward = 0

        while True:
            if rd.random() < hyperparams["epsilon"]:
                action = env.action_space.sample()
            else:
                state_tensor = torch.as_tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = dqn(state_tensor)
                action = torch.argmax(q_values).item()

            observation, reward, terminated, truncated, info = env.step(action)
            reward = reward_map.get(reward, reward)
            episode_reward += reward
            previous_state = state
            state = extract_features(observation) # We use relative positions to generalize better
            replay_buffer.add(previous_state, action, reward, state, terminated or truncated)

            # Learning step
            if len(replay_buffer.buffer) > hyperparams["batch_size"]:
                batch = replay_buffer.sample(hyperparams["batch_size"])

                # Extract tensors from batch
                states = torch.as_tensor(np.array([exp.state for exp in batch]), dtype=torch.float32).to(device)
                actions = torch.as_tensor(np.array([exp.action for exp in batch]), dtype=torch.int64).unsqueeze(1).to(device)
                rewards = torch.as_tensor(np.array([exp.reward for exp in batch]), dtype=torch.float32).to(device)
                next_states = torch.as_tensor(np.array([exp.next_state for exp in batch]), dtype=torch.float32).to(device)
                dones = torch.as_tensor(np.array([exp.done for exp in batch]), dtype=torch.float32).to(device)   

                # Compute current Q values
                current_q_values = dqn(states).gather(1, actions) # Get Q values for taken actions

                # Compute expected Q values
                with torch.no_grad():
                    next_q_values = target_dqn(next_states)
                    max_next_q_values, _ = torch.max(next_q_values, dim=1)
                    expected_q_values = rewards + (1 - dones) * hyperparams["gamma"] * max_next_q_values # Bellman equation

                # We use Huber loss (Smooth L1 Loss) for stability instead of MSE
                loss = torch.nn.functional.mse_loss(current_q_values, expected_q_values.unsqueeze(1))             

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            step += 1

            if step % hyperparams["target_update_freq"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            if terminated or truncated:
                break

        rewards_history.append(episode_reward)
        score_history.append(info.get("score", 0))
        average_reward = np.mean(rewards_history[-hyperparams["EpisodeRewardDisplayFreq"]:])
        average_score = np.mean(score_history[-hyperparams["EpisodeRewardDisplayFreq"]:])
        average_reward_history.append(average_reward)
        average_score_history.append(average_score)

        if episode % hyperparams["EpisodeRewardDisplayFreq"] == 0:
            output = (f"Episode: {episode}, Step {step}, Epsilon: {hyperparams['epsilon']:.4f}, " +
                  f"Average reward (last {hyperparams['EpisodeRewardDisplayFreq']} episodes): {average_reward:.2f}, " +
                  f"Average score (last {hyperparams['EpisodeRewardDisplayFreq']} episodes): {average_score:.2f}")

            # Writing logs
            logs.write(output + "\n")
            logs.flush()

            # Console output
            print(output)

        if episode % hyperparams["ModelSaveFreq"] == 0 or (average_score > best_average_score and average_score >= 1.0):
            torch.save(dqn.state_dict(), f"{model_directory_path}/dqn_model_episode_{episode}.pth")

        if average_score > best_average_score:
                best_average_score = average_score
                logs.write(f"New best average score: {best_average_score:.2f} at episode {episode}\n")
                logs.flush()

        if episode % hyperparams["PlotFreq"] == 0:
            # Plotting average reward history and score history 
            plt.figure(figsize=(12,5))
            plt.subplot(1, 2, 1)
            plt.plot(average_reward_history)
            plt.title('Average Episode Reward History')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.subplot(1, 2, 2)
            plt.plot(average_score_history)
            plt.title('Episode Score History')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.savefig(f"{model_directory_path}/training_plot_episode_{episode}.png")
            plt.close()
            

        hyperparams["epsilon"] *= hyperparams["epsilon_decay"]
        hyperparams["epsilon"] = max(hyperparams["epsilon"], hyperparams["epsilon_min"])
    
    logs.close()

    print("Training completed.")