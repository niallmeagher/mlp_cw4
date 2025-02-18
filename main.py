import os
import glob
import torch
import numpy as np

import subprocess
from PPOAgents import PPOAgent, RewardFunction  # Make sure your PPOAgent is defined and importable
dir = "/home/s2686742/Cell2Fire/cell2fire/Cell2FireC/"


def read_asc_to_tensor(file_path, header_lines=6):
    # Read the file, skipping the header lines
    with open(file_path, 'r') as f:
        # Skip header lines
        for _ in range(header_lines):
            next(f)
        
        # Read the remaining lines and extract the grid (assuming it's a 20x20 grid of floats)
        grid = []
        for line in f:
            # Split the line into individual values and convert to float
            grid.append(list(map(float, line.split())))

    # Convert the grid into a numpy array
    grid_np = np.array(grid)

    # Ensure the grid is of shape (20, 20)
    if grid_np.shape != (20, 20):
        raise ValueError(f"Expected grid size of (20, 20), but got {grid_np.shape}")

    # Convert the numpy array to a PyTorch tensor and add the extra dimensions (1, 1)
    tensor = torch.tensor(grid_np).unsqueeze(0).unsqueeze(0)

    return tensor

# Example usage
  # Replace with your actual file path
def read_multi_channel_asc(files, header_lines=6):
    """
    Reads multiple ASC files and stacks them as channels in a tensor.
    """
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
    episodes_per_epoch = 3    # Number of episodes (trajectories) to collect per update

    # Initialize PPO Agent (this creates the network, optimizer, etc.)
   # agent = PPOAgent(learned_reward=False)
    #csv_forest = "/home/s2686742/Cell2Fire/data/Sub20x20/Forest.asc"
    #tensor_forest = read_asc_to_tensor(csv_forest)
    agent = PPOAgent(input_channels=4, learned_reward=False)  # Update input channels
    
    files = [
        "/home/s2686742/Cell2Fire/data/Sub20x20/Forest.asc",
        "/home/s2686742/Cell2Fire/data/Sub20x20/Elevation.asc",
        "/home/s2686742/Cell2Fire/data/Sub20x20/Saz.asc",
        "/home/s2686742/Cell2Fire/data/Sub20x20/Slope.asc"
    ]
    tensor_input = read_multi_channel_asc(files)
    # Main training loop
    for epoch in range(num_epochs):
        # Containers for storing trajectories (each episode is assumed to be one step for simplicity)
        trajectories = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'returns': [],
            'masks': [],
            # This mask is used for action validity (here all actions are valid)
            'true_rewards': []
        }
        total_reward = 0.0

        for episode in range(episodes_per_epoch):
            # Environment reset: a dummy 20x20 grid state.
            state = tensor_input.clone()  # For example, an empty grid.
            # Assume all actions are valid.
            valid_actions_mask = torch.ones(1, 400)

            # Select an action.
            action, log_prob, value, real_action  = agent.select_action(
                state, mask=valid_actions_mask)

            # Use the learnable reward function to predict a reward (if desired).
            #pred_reward = agent.reward_function(state, action)

            # Simulate the fire episode to get a true reward.
            true_reward = agent.simulate_fire_episode(state[:,0:1,:,:], real_action)
            #true_reward = agent.simulate_test_episode(state, action)
            total_reward += true_reward

            # For this one-step episode, the return is the true reward.
            trajectories['states'].append(state)
            trajectories['actions'].append(
                torch.tensor(action, dtype=torch.long))
            trajectories['log_probs'].append(log_prob)
            trajectories['values'].append(value)
            trajectories['returns'].append(
                torch.tensor([true_reward], dtype=torch.float32))
            trajectories['masks'].append(valid_actions_mask)
            trajectories['true_rewards'].append(
                torch.tensor([true_reward], dtype=torch.float32))

        # Convert lists into tensors.
        trajectories['states'] = torch.cat(trajectories['states'], dim=0)
        trajectories['actions'] = torch.stack(trajectories['actions'])
        trajectories['log_probs'] = torch.stack(trajectories['log_probs'])
        trajectories['values'] = torch.cat(trajectories['values'], dim=0)
        trajectories['returns'] = torch.cat(
            trajectories['returns'], dim=0).squeeze(-1)
        trajectories['masks'] = torch.cat(trajectories['masks'], dim=0)
        trajectories['true_rewards'] = torch.cat(
            trajectories['true_rewards'], dim=0).squeeze(-1)

        # Update the policy and reward function.
        agent.update(trajectories)
        avg_reward = total_reward / episodes_per_epoch
        print(
            f"Epoch {epoch+1}/{num_epochs} - Average True Reward: {avg_reward:.4f}")

    test_state = torch.zeros(1, 1, 20, 20)
    test_mask = torch.ones(1, 400)
    action, log_prob, value = agent.select_action(test_state, mask=test_mask)
    print("\nFinal Test:")
    print(f"Chosen Action: {action}")
    print(f"Estimated Value: {value.item():.4f}")
    test_true_reward = agent.simulate_test_episode(test_state, action)
    print(f"Test True Reward: {test_true_reward.item():.4f}")
if __name__ == '__main__':
    main()
