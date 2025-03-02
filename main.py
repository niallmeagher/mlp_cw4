import os
import glob
import torch
import shutil
import numpy as np
import argparse

import subprocess
from PPOAgents import PPOAgent, RewardFunction  # Make sure your PPOAgent is defined and importable

# These shouldn't be necessary as directories are cl args now
# HOME_DIR = '/home/s2750319/Cell2Fire/' # UPDATE THIS TO POINT TO YOUR STUDENT NUMBER
# dir = f"{HOME_DIR}cell2fire/Cell2FireC/"


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
    # Ensure folder1 exists
    os.makedirs(folder1, exist_ok=True)
    
    # 1. Clear folder1
    for filename in os.listdir(folder1):
        file_path = os.path.join(folder1, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    
    # 2. Get a list of CSV files in folder2
    csv_files = glob.glob(os.path.join(folder2, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder2}")
    
    # Randomly select one CSV file
    selected_file = np.random.choice(csv_files)
    destination_file = os.path.join(folder1, os.path.basename(selected_file))
    shutil.copy(selected_file, destination_file)
    
    # 3. Load CSV data using numpy
    # If there's a header, skip the first row. Note that np.genfromtxt can automatically skip the header.
    skip_rows = 1 if has_header else 0
    # Use delimiter=',' assuming CSV format.
    data = np.genfromtxt(destination_file, delimiter=',', skip_header=skip_rows)
    
    # 4. Drop the first drop_first_n_cols columns
    if drop_first_n_cols > 0:
        data = data[:, drop_first_n_cols:]
    
    # 5. Convert the NumPy array to a PyTorch tensor (assuming numeric data)
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

def main(args):
    # Data folders
    input_dir = args['input_dir'] # e.g Sub20x20
    output_dir = args['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = open(f'{output_dir}/losses.csv','w')
    output_file.write('epoch,reward\n')

    # Hyperparameters
    num_epochs = int(args['num_epochs'])          # Number of PPO update cycles
    episodes_per_epoch = int(args['episodes'])    # Number of episodes (trajectories) to collect per update

    # Initialize PPO Agent (update input channels if needed)
    agent = PPOAgent(input_folder=f'{input_dir}/', new_folder=f'{input_dir}_Test/', output_folder=f'{output_dir}',
                     input_channels=4, learned_reward=False)
    
    files = [
        f"{input_dir}/Forest.asc",
        f"{input_dir}/elevation.asc",
        f"{input_dir}/saz.asc",
        f"{input_dir}/slope.asc"
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
            'true_rewards': [],
            'weather': []
        }
        total_reward = 0.0

        # create weathers_stored folder if necessary
        folder_sample_from = os.path.join(input_dir, "Weathers")
        folder_stored = os.path.join(input_dir, "Weathers_Stored")
        if os.path.exists(folder_sample_from) and not os.path.exists(folder_stored):
            os.rename(folder_sample_from, folder_stored)

        tensor_data = load_random_csv_as_tensor(folder_sample_from, folder_stored, drop_first_n_cols=2, has_header=True)
        tabular_tensor = tensor_data.view(1, 8, 11)
        
        for episode in range(episodes_per_epoch):
            eps_greedy = False
            state = tensor_input.clone()  # Reset environment state.
            valid_actions_mask = mask
            '''
            if np.random.uniform() <= 0.05:
                eps_greedy = True
            '''
            action_indices, log_prob, value, real_action = agent.select_action(state, tabular_tensor, valid_actions_mask, eps_greedy)
            
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
        output_file.write(f"{epoch+1},{avg_reward:.4f}\n")

    test_state = torch.zeros(1, 4, 20, 20)
    test_mask = torch.ones(1, 400)
    action_indices, log_prob, value, _ = agent.select_action(test_state, mask=test_mask)
    print("\nFinal Test:")
    print(f"Chosen Action Indices: {action_indices}")
    print(f"Estimated Value: {value.item():.4f}")
    test_true_reward = agent.simulate_test_episode(test_state, action_indices[0])
    print(f"Test True Reward: {test_true_reward.item():.4f}")
    output_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num_epochs', help='Number of taining epochs to perform', required=True)
    parser.add_argument('-e','--episodes', help='Number of episodes per epoch', required=True)
    parser.add_argument('-i','--input_dir', help='Path to folder containing input data', required=True)
    parser.add_argument('-o','--output_dir', help='Path to folder where output will be stored', required=True)
    args = vars(parser.parse_args())
    main(args)