"""
Training script 
"""

import torch
import torch.optim as optim
import random as rd
import numpy as np
from networks import BirdDQN, ReplayBuffer


def train(env, device, dqn, target_dqn, replay_buffer, optimizer, hyperparams):
    step = 0
    # Playing loop
    for _ in range(hyperparams["num_episodes"]):
        observation, info = env.reset()
        state = observation[[3, 4, 5, 9, 10]] 

        while True:
            if rd.random() < hyperparams["epsilon"]:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = dqn(state_tensor)
                action = torch.argmax(q_values).item()

            observation, reward, terminated, truncated, info = env.step(action)
            previous_state = state
            state = observation[[3, 4, 5, 9, 10]] 
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
                    expected_q_values = rewards + (1 - dones) * hyperparams["gamma"] * max_next_q_values


                loss = torch.nn.functional.mse_loss(current_q_values, expected_q_values.unsqueeze(1))             

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1

                if step % hyperparams["target_update_freq"] == 0:
                    target_dqn.load_state_dict(dqn.state_dict())


            if terminated or truncated:
                break
        
        hyperparams["epsilon"] *= hyperparams["epsilon_decay"]
        hyperparams["epsilon"] = max(hyperparams["epsilon"], hyperparams["epsilon_min"])