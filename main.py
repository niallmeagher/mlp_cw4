import os
import glob
import torch
import shutil
import numpy as np
import csv

import subprocess
from PPOAgents import PPOAgent, RewardFunction  # Make sure your PPOAgent is defined and importable

HOME_DIR = '/home/s2750265/Cell2Fire/' # UPDATE THIS TO POINT TO YOUR STUDENT NUMBER
dir = f"{HOME_DIR}cell2fire/Cell2FireC/"

def save_checkpoint(agent, epoch, checkpoint_dir=f"{HOME_DIR}/data/Sub20x20_Test/Checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": agent.network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "learned_reward": agent.learned_reward
    }
    # Save the reward network if you're using one.
    if agent.learned_reward and agent.reward_net is not None:
        checkpoint["reward_net_state_dict"] = agent.reward_net.state_dict()
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(agent, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    agent.network.load_state_dict(checkpoint["model_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if agent.learned_reward and "reward_net_state_dict" in checkpoint:
        agent.reward_net.load_state_dict(checkpoint["reward_net_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"Resuming training from epoch {start_epoch}")
    return start_epoch


def load_random_csv_as_tensor(folder1, folder2, drop_first_n_cols=2, has_header=True):
    """
    Clears folder1, randomly selects a CSV from folder2, copies it to folder1,
    and returns the CSV data as a PyTorch tensor after optionally skipping the header and dropping the first N columns.

    Args:
        folder1 (str): Path to the destination folder (will be cleared).
        folder2 (str): Path to the folder containing CSV files.
        drop_first_n_cols (int): Number of columns to drop from the left (default: 2).
        has_header (bool): If True, skips the first row of the CSV.

    Returns:
        torch.Tensor: Data from the CSV as a tensor of type torch.float32.
    """
    os.makedirs(folder1, exist_ok=True)
    
    for filename in os.listdir(folder1):
        file_path = os.path.join(folder1, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    
    csv_files = glob.glob(os.path.join(folder2, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder2}")
    
    # Randomly select one CSV file
    selected_file = np.random.choice(csv_files)
    destination_file = os.path.join(folder1, os.path.basename(selected_file))
    shutil.copy(selected_file, destination_file)
    
    skip_rows = 1 if has_header else 0
    data = np.genfromtxt(destination_file, delimiter=',', skip_header=skip_rows)
    
    if drop_first_n_cols > 0:
        data = data[:, drop_first_n_cols:]
    
    data_tensor = torch.tensor(data, dtype=torch.float32)
    return data_tensor

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

def main(start_epoch=0, checkpoint_path=None):
    # Hyperparameters
    num_epochs = 1000          # Number of PPO update cycles
    episodes_per_epoch = 3    # Number of episodes (trajectories) to collect per update

    # Initialize PPO Agent (update input channels if needed)
    agent = PPOAgent(input_channels=4, learned_reward=False)
    
    csv_file = "episode_results.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Identifier", "Reward", "Value"])

    if checkpoint_path is not None:
        start_epoch = load_checkpoint(agent, checkpoint_path)
    else:
        start_epoch = 0

    files = [
        f"{HOME_DIR}data/Sub20x20/Forest.asc",
        f"{HOME_DIR}data/Sub20x20/elevation.asc",
        f"{HOME_DIR}data/Sub20x20/saz.asc",
        f"{HOME_DIR}data/Sub20x20/slope.asc"
    ]
    tensor_input = read_multi_channel_asc(files)
    # Build a mask for valid actions from the first channel.
    mask = tensor_input[0,0,:,:] != 101
    mask = mask.view(1,400)
   # print(mask)
    for epoch in range(start_epoch, num_epochs):
        trajectories = {
            'states': [],
            'actions': [],       # will store tensors of shape (20,)
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': [],
            'masks': [],
            'true_rewards': [],
            'weather': []
        }
        epoch_rewards = []
        epoch_values = []
            
        total_reward = 0.0
        folder_sample_from = os.path.join(f"{HOME_DIR}data/Sub20x20_Test/", "Weathers")
        folder_stored = os.path.join(f"{HOME_DIR}data/Sub20x20_Test/", "Weathers_Stored")
        tensor_data = load_random_csv_as_tensor(folder_sample_from, folder_stored, drop_first_n_cols=2, has_header=True)
        tabular_tensor = tensor_data.view(1, 8, 11)
        
        for episode in range(episodes_per_epoch):
            
            state = tensor_input.clone()
            valid_actions_mask = mask
            
            action_indices, log_prob, value, real_action = agent.select_action(state, tabular_tensor, valid_actions_mask)
            
            print("Value", value)
            
            # Simulate the fire episode to get the true reward.
            true_reward = agent.simulate_fire_episode(state[:,0:1,:,:], action_indices)
            total_reward += true_reward
            epoch_rewards.append(float(true_reward))
            epoch_values.append(float(value.item()))
            
            
            # For a one-step episode, done is True.
            done = torch.tensor(1, dtype=torch.float32, device=agent.device)
            trajectories['states'].append(state)
            trajectories['actions'].append(action_indices)  # store the 20 selected indices
            trajectories['log_probs'].append(log_prob)
            trajectories['values'].append(value)
            trajectories['rewards'].append(torch.tensor([true_reward], dtype=torch.float32))
            trajectories['dones'].append(done)
            trajectories['weather'].append(tabular_tensor)
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
        trajectories['weather'] = torch.cat(trajectories['weather'], dim=0)
        trajectories['true_rewards'] = torch.cat(trajectories['true_rewards'], dim=0).squeeze(-1)

        agent.update(trajectories)
        avg_reward = total_reward / episodes_per_epoch
        print(f"Epoch {epoch+1}/{num_epochs} - Average True Reward: {avg_reward:.4f}")
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            for ep in range(episodes_per_epoch):
                identifier = f"Epoch_{epoch+1}_Episode_{ep+1}"
                writer.writerow([identifier, epoch_rewards[ep], epoch_values[ep]])
        save_checkpoint(agent, epoch+1)
    

    final_path = "final_model.pt"
    torch.save(agent.network.state_dict(), final_path)
    print(f"Final model saved at {final_path}")
    '''
    test_state = torch.zeros(1, 1, 20, 20)
    test_mask = torch.ones(1, 400)
    action_indices, log_prob, value, _ = agent.select_action(test_state, mask=test_mask)
    print("\nFinal Test:")
    print(f"Chosen Action Indices: {action_indices}")
    print(f"Estimated Value: {value.item():.4f}")
    test_true_reward = agent.simulate_test_episode(test_state, action_indices[0])
    print(f"Test True Reward: {test_true_reward.item():.4f}")
    '''

if __name__ == '__main__':
    checkpoint_file = None  # Replace with your file path if needed.
    main(start_epoch=0, checkpoint_path=checkpoint_file)
