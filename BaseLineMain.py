import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import numpy as np
import argparse
import csv
import argparse
import time
import uuid
import tempfile
from concurrent.futures import ThreadPoolExecutor as TPE
from concurrent.futures import ProcessPoolExecutor as PPE
import multiprocessing as mp
from datetime import datetime

import subprocess
from BaseLineAgents import DQNAgent  # Changed to DQNAgent

username = os.getenv('USER')
HOME_DIR = os.path.join('/disk/scratch', username,'Cell2Fire', 'data') +'/'
HOME_DIR2 = os.path.join('/disk/scratch', username,'Cell2Fire', 'results') +'/'

def save_results_to_csv(results, output_dir, filename="experiment_results.csv"):
    """
    Save experiment results to a CSV file.

    Args:
        results (list of dict): List of dictionaries containing experiment results.
        output_dir (str): Directory where the CSV file will be saved.
        filename (str): Name of the CSV file (default: "experiment_results.csv").
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the full path to the CSV file
    csv_file_path = os.path.join(output_dir, filename)

    # Define the CSV fieldnames (column headers)
    fieldnames = ["Epoch", "Reward", "Burned Cells", "Loss", 'Time Elapsed']

    # Write results to the CSV file
    with open(csv_file_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Write the rows
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {csv_file_path}")

def save_checkpoint(agent, epoch, checkpoint_dir):
    """Save training checkpoint with model state"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    # Handle DataParallel wrapping when saving
    policy_state_dict = agent.policy_net.module.state_dict() if isinstance(agent.policy_net, nn.DataParallel) else agent.policy_net.state_dict()
    target_state_dict = agent.target_net.module.state_dict() if isinstance(agent.target_net, nn.DataParallel) else agent.target_net.state_dict()
    
    checkpoint = {
        "epoch": epoch,
        "policy_state_dict": policy_state_dict,
        "target_state_dict": target_state_dict,
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "learned_reward": agent.learned_reward,
        "scheduler_state_dict": agent.scheduler.state_dict() if agent.scheduler else None
    }
    
    if agent.learned_reward and hasattr(agent, "reward_net"):
        reward_state_dict = agent.reward_net.module.state_dict() if isinstance(agent.reward_net, nn.DataParallel) else agent.reward_net.state_dict()
        checkpoint["reward_net_state_dict"] = reward_state_dict
        
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(agent, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    
    # Handle DataParallel wrapping when loading
    if isinstance(agent.policy_net, nn.DataParallel):
        agent.policy_net.module.load_state_dict(checkpoint["policy_state_dict"])
    else:
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    
    if isinstance(agent.target_net, nn.DataParallel):
        agent.target_net.module.load_state_dict(checkpoint["target_state_dict"])
    else:
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if agent.learned_reward and "reward_net_state_dict" in checkpoint:
        if isinstance(agent.reward_net, nn.DataParallel):
            agent.reward_net.module.load_state_dict(checkpoint["reward_net_state_dict"])
        else:
            agent.reward_net.load_state_dict(checkpoint["reward_net_state_dict"])
    
    if agent.scheduler and "scheduler_state_dict" in checkpoint:
        agent.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
    start_epoch = checkpoint["epoch"]
    print(f"Resuming training from epoch {start_epoch}")
    return start_epoch

def load_random_csv_as_tensor(folder1, folder2, drop_first_n_cols=2, has_header=True):
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

def prepare_state_tensor(tensor_input, tabular_tensor):
    """
    Concatenate the state tensor (1, 4, 20, 20) with the tabular tensor (1, 8, 11).
    The tabular tensor is reshaped to match the spatial dimensions of the state tensor.
    """
    # Reshape tabular_tensor to (1, 8, 11, 1, 1) and repeat it to match the spatial dimensions of the state tensor
    tabular_tensor = tabular_tensor.unsqueeze(-1).unsqueeze(-1)  # Shape: (1, 8, 11, 1, 1)
    tabular_tensor = tabular_tensor.expand(-1, -1, -1, 20, 20)  # Shape: (1, 8, 11, 20, 20)
    
    tabular_tensor = tabular_tensor.reshape(1, -1, 20, 20)

    # Concatenate along the channel dimension
    combined_state = torch.cat([tensor_input, tabular_tensor], dim=1)  # Shape: (1, 12, 20, 20)
    return combined_state

def simulate_single_episode(agent, state, mask, input_folder):
    # Create a temporary working directory for this episode
    state = state.float()
    mask = mask.float()
    episode_id = uuid.uuid4().hex
    temp_work_dir = os.path.join(HOME_DIR, f"cell2fire_input_{episode_id}/")
    os.mkdir(temp_work_dir)
    temp_output_dir = tempfile.mkdtemp(prefix=f"cell2fire_output_{episode_id}", dir=HOME_DIR2)
    temp_output_base_dir = tempfile.mkdtemp(prefix=f"cell2fire_output_base_{episode_id}", dir=HOME_DIR2)
    
    try:
        shutil.copytree(input_folder, temp_work_dir, dirs_exist_ok=True)
    except Exception as e:
        print("Error during copytree:", e)
        raise
    
    try:
        # Select 20 actions (firebreak locations)
        actions = agent.select_action(state, mask)
        # Simulate the fire episode
        true_reward, average_burned_cells = agent.simulate_fire_episode(actions, work_folder=temp_work_dir, output_folder=temp_output_dir, output_folder_base=temp_output_base_dir, num_simulations=10)
        
        if true_reward is None:  # Check if reward is None
            print("Warning: Reward is None from simulate_fire_episode. Episode failed.")
            shutil.rmtree(temp_work_dir, ignore_errors=True)  # Clean up work folder if episode failed
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            shutil.rmtree(temp_output_base_dir, ignore_errors=True)
            return None  # Return None for the entire episode

    finally:
        # Clean up the temporary folder after simulation
        shutil.rmtree(temp_work_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        shutil.rmtree(temp_output_base_dir, ignore_errors=True)
    
    # Return the experience
    return {
        'state': state.detach(),
        'actions': torch.tensor(actions, dtype=torch.long),  # Shape: (20,)
        'reward': torch.tensor([true_reward], dtype=torch.float32),
        'next_state': state.detach(),  # Next state is the same since the episode ends after one step
        'done': torch.tensor(1, dtype=torch.float32),  # Episode ends after one step
        'mask': mask.detach(),
        'ABCells': average_burned_cells
    }
    

def main(args, start_epoch=0, checkpoint_path=None):
    input_dir = args['input_dir'] # e.g Sub20x20
    output_dir = args['output_dir']
    #if not os.path.exists(output_dir):
       # os.makedirs(output_dir)

    output_file = open(f'{HOME_DIR2}/Epoch_Stats.csv','w')
    output_file.write('epoch,reward,loss,policy_loss,value_loss,entropy\n')

    # Hyperparameters
    num_epochs = int(args['num_epochs'])
    episodes_per_epoch = int(args['episodes'])

    # Initialize PPO Agent (update input channels if needed)
    new_folder=f'{input_dir}_Test/'
    input_folder_final=f'{input_dir}/'
    output_folder=f'{output_dir}v2'
    output_folder_base=f'{output_dir}_base/'
    #agent = PPOAgent(input_channels=4, learned_reward=False)
    agent = DQNAgent(input_folder_final, new_folder, output_folder,output_folder_base,
                     input_channels=4, learned_reward=False)
    
    csvf = "episode_results.csv"
    csv_file = os.path.join(f"{HOME_DIR2}",csvf)
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Identifier", "Reward", "Value"])

    if checkpoint_path is not None:
        start_epoch = load_checkpoint(agent, checkpoint_path)
    else:
        start_epoch = 0

    files = [
        f"{input_dir}/Forest.asc",
        f"{input_dir}/elevation.asc",
        f"{input_dir}/saz.asc",
        f"{input_dir}/slope.asc"
    ]
    tensor_input = read_multi_channel_asc(files)
    mask = tensor_input[0,0,:,:] != 101
    mask = mask.view(1,400)

    #demonstrations = agent.generate_demonstrations(tensor_input, 500)
    #print("Demonstrations collected")
    #agent.preTraining(demonstrations, num_epochs=500)
    fullresults = []

    for epoch in range(start_epoch, num_epochs):
        total_reward = 0.0
        totalABCells = 0.0
    
        #folder_sample_from = os.path.join(input_dir, "Weathers")
        #folder_stored = os.path.join(input_dir, "Weathers_Stored")
        #if os.path.exists(folder_sample_from) and not os.path.exists(folder_stored):
            #os.rename(folder_sample_from, folder_stored)
        #tensor_data = load_random_csv_as_tensor(folder_sample_from, folder_stored, drop_first_n_cols=2, has_header=True)
        #tabular_tensor = tensor_data.view(1, 8, 11)
        #combined_state = prepare_state_tensor(tensor_input, tabular_tensor)

        start_time = time.time()
        with TPE(max_workers=mp.cpu_count()) as executor:
            
            futures = [executor.submit(simulate_single_episode, agent,
                                   tensor_input.clone(), mask, input_folder_final)
                   for _ in range(episodes_per_epoch)]
           
            results = [future.result() for future in futures]
        nones = 0
        for res in results:
            if res is None:
                nones+=1
                continue
            agent.store_transition(res['state'], res['actions'], res['reward'], res['next_state'], res['done'], res['mask'])
            totalABCells += res['ABCells']
            total_reward += res['reward'].item()

        loss = agent.update()
        if loss == None:
            loss = 0.0
        avg_reward = (total_reward) / (episodes_per_epoch -nones )
        avg_BCells = totalABCells / (episodes_per_epoch - nones)
        print(f"Epoch {epoch+1}/{num_epochs} - Average True Reward: {avg_reward:.4f}")
        print(avg_BCells)

        save_checkpoint(agent, epoch + 1, checkpoint_dir=f"{input_dir}_Test/Checkpoints")
        output_file.write(f"{epoch + 1},{avg_reward:.4f},{loss:.4f}\n")
        output_file.flush()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        fullresults.append([epoch + 1, avg_reward, avg_BCells, loss, elapsed_time])

    epochs, rewards, averageBurncells, losses, times = zip(*fullresults)
    resultDict = [
        {
            "Epoch": epoch,
            "Reward": reward,
            "Burned Cells": burned_cells,
            "Loss": loss,
            "Time Elapsed": time_elapsed
        }
        for epoch, reward, burned_cells, loss, time_elapsed in zip(epochs, rewards, averageBurncells, losses, times)
    ]


    save_results_to_csv(resultDict, '/home/s2750265/mlp_cw4/results', filename="final_results.csv")
    # Save the final model
    output_dir
    final_path = "final_model.pt"
    torch.save(agent.policy_net.state_dict(), final_path)
    print(f"Final model saved at {final_path}")
    output_file.close()
    

if __name__ == '__main__':
    #mp.set_start_method('spawn', force=True)
    checkpoint_file = None
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num_epochs', help='Number of taining epochs to perform', required=True)
    parser.add_argument('-e','--episodes', help='Number of episodes per epoch', required=True)
    parser.add_argument('-i','--input_dir', help='Path to folder containing input data', required=True)
    parser.add_argument('-o','--output_dir', help='Path to folder where output will be stored', required=True)
    parser.add_argument('-c', '--checkpoint_path', help='Path to checkpoint file if you are loading one', required=False, default=None)
    parser.add_argument('-s', '--start_epoch', help='The number of the starting epoch (if you are resuming a failed run)', required=False, default=0)
    args = vars(parser.parse_args())
    main(args)
