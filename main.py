import os
import glob
import torch
import numpy as np

import subprocess
from PPOAgents import PPOAgent, RewardFunction  # Make sure your PPOAgent is defined and importable
dir = "/home/s2686742/Cell2Fire/cell2fire/Cell2FireC/"

def read_asc_to_tensor(file_path, header_lines=6):
    with open(file_path, 'r') as f:
        for _ in range(header_lines):
            next(f)
        grid = []
        for line in f:
            grid.append(list(map(float, line.split())))
    grid_np = np.array(grid)
    if grid_np.shape != (20, 20):
        raise ValueError(f"Expected grid size of (20, 20), but got {grid_np.shape}")
    tensor = torch.tensor(grid_np).unsqueeze(0).unsqueeze(0)
    return tensor

def read_multi_channel_asc(files, header_lines=6):
    tensors = []
    for file_path in files:
        with open(file_path, 'r') as f:
            for _ in range(header_lines):
                next(f)
            grid = [list(map(float, line.split())) for line in f]
        grid_np = np.array(grid)
        if grid_np.shape != (20, 20):
            raise ValueError(f"Expected grid size of (20, 20), but got {grid_np.shape}")
        tensors.append(torch.tensor(grid_np))
    return torch.stack(tensors).unsqueeze(0)  # Shape (1, 4, 20, 20)

def main():
    # Hyperparameters
    num_epochs = 1000          # Number of PPO update cycles
    episodes_per_epoch = 3     # Number of episodes (trajectories) to collect per update

    # Initialize PPO Agent (update input channels if needed)
    agent = PPOAgent(input_channels=4, learned_reward=False)
    
    files = [
        "/home/s2686742/Cell2Fire/data/Sub20x20/Forest.asc",
        "/home/s2686742/Cell2Fire/data/Sub20x20/elevation.asc",
        "/home/s2686742/Cell2Fire/data/Sub20x20/saz.asc",
        "/home/s2686742/Cell2Fire/data/Sub20x20/slope.asc"
    ]
    tensor_input = read_multi_channel_asc(files)
    # Build a mask for valid actions from the first channel.
    mask = tensor_input[0,0,:,:] != 101
    mask = mask.view(1,400)
   # print(mask)
    for epoch in range(num_epochs):
        trajectories = {
            'states': [],
            'actions': [],       # will store tensors of shape (20,)
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': [],
            'masks': [],
            'true_rewards': []
        }
        total_reward = 0.0
        
        for episode in range(episodes_per_epoch):
            eps_greedy = False
            state = tensor_input.clone()  # Reset environment state.
            valid_actions_mask = mask
            '''
            if np.random.uniform() <= 0.05:
                eps_greedy = True
            '''
            action_indices, log_prob, value, real_action = agent.select_action(state, valid_actions_mask, eps_greedy)
            
            print("Value", value)
            
            # Simulate the fire episode to get the true reward.
            true_reward = agent.simulate_fire_episode(state[:,0:1,:,:], action_indices, eps_greedy)
            total_reward += true_reward
            
            # For a one-step episode, done is True.
            done = torch.tensor(1, dtype=torch.float32, device=agent.device)
            trajectories['states'].append(state)
            trajectories['actions'].append(action_indices)  # store the 20 selected indices
            trajectories['log_probs'].append(log_prob)
            trajectories['values'].append(value)
            trajectories['rewards'].append(torch.tensor([true_reward], dtype=torch.float32))
            trajectories['dones'].append(done)
            trajectories['masks'].append(valid_actions_mask)
            trajectories['true_rewards'].append(torch.tensor([true_reward], dtype=torch.float32))
           # print(valid_actions_mask.shape)
          
        trajectories['states'] = torch.cat(trajectories['states'], dim=0)
        trajectories['actions'] = torch.stack(trajectories['actions'])  # shape (episodes, 20)
        trajectories['log_probs'] = torch.stack(trajectories['log_probs'])
        trajectories['values'] = torch.cat(trajectories['values'], dim=0)
        trajectories['rewards'] = torch.cat(trajectories['rewards'], dim=0).squeeze(-1)
        trajectories['dones'] = torch.tensor(trajectories['dones'], dtype=torch.float32, device=agent.device)
        trajectories['masks'] = torch.cat(trajectories['masks'], dim=0)
        trajectories['true_rewards'] = torch.cat(trajectories['true_rewards'], dim=0).squeeze(-1)

        agent.update(trajectories)
        avg_reward = total_reward / episodes_per_epoch
        print(f"Epoch {epoch+1}/{num_epochs} - Average True Reward: {avg_reward:.4f}")

    test_state = torch.zeros(1, 1, 20, 20)
    test_mask = torch.ones(1, 400)
    action_indices, log_prob, value, _ = agent.select_action(test_state, mask=test_mask)
    print("\nFinal Test:")
    print(f"Chosen Action Indices: {action_indices}")
    print(f"Estimated Value: {value.item():.4f}")
    test_true_reward = agent.simulate_test_episode(test_state, action_indices[0])
    print(f"Test True Reward: {test_true_reward.item():.4f}")

if __name__ == '__main__':
    main()