import os
import glob
import torch
import torch.nn as nn
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
import optuna

import subprocess
from PPOAgents import PPOAgent, RewardFunction  # Make sure your PPOAgent is defined and importable

username = os.getenv('USER')
HOME_DIR = os.path.join('/disk/scratch', username,'Cell2Fire', 'data') +'/'
HOME_DIR2 = os.path.join('/disk/scratch', username,'Cell2Fire', 'results') +'/'


def save_checkpoint(agent, epoch, checkpoint_dir):
    """Save training checkpoint with model state"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pt")
    
    # Handle DataParallel wrapping when saving
    model_state_dict = agent.network.module.state_dict() if isinstance(agent.network, nn.DataParallel) else agent.network.state_dict()
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "learned_reward": agent.learned_reward,
        "scheduler_state_dict": agent.scheduler.state_dict() if agent.scheduler else None
    }
    
    if agent.learned_reward and agent.reward_net is not None:
        reward_state_dict = agent.reward_net.module.state_dict() if isinstance(agent.reward_net, nn.DataParallel) else agent.reward_net.state_dict()
        checkpoint["reward_net_state_dict"] = reward_state_dict
        
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(agent, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    
    # Handle DataParallel wrapping when loading
    if isinstance(agent.network, nn.DataParallel):
        agent.network.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        agent.network.load_state_dict(checkpoint["model_state_dict"])
        
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


def load_best_hyperparams(db_path="/home/s2750319/shared_storage/optuna.db", study_name="ppo_20x20"):
    """
    Load the best hyperparameters from the Optuna study.
    """
    storage = f"sqlite:///{db_path}"
    
    try:
        # Load the study
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        
        # Get the best trial
        best_trial = study.best_trial
        print("Loaded best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        return best_trial.params
    
        
    except Exception as e:
        print(f"Error loading study: {e}")
        return {'lr': 0.0007112714510323,
                'T_max': 42,
                'clip_epsilon': 0.2497741529163497,
                'entropy_coef': 0.0020622640186278,
                'eta_min': 1.415085079651966e-06,
                'gae_lambda': 0.9461413974687652,
                'gamma': 0.9932907418871656,
                'scheduler': 'step',
                'value_loss_coef': 0.2642195827652803
                }

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

def simulate_single_episode(agent, state, tabular_tensor, mask, input_folder):
    # Create a temporary working directory for this episode
  
    episode_id = uuid.uuid4().hex
    testing = "/tmp/"
    #temp_work_dir = tempfile.mkdtemp(prefix=f"cell2fire_input_{episode_id} /", dir = HOME_DIR)
    temp_work_dir = os.path.join(HOME_DIR,f"cell2fire_input_{episode_id}/" )
    os.mkdir(temp_work_dir)
    temp_output_dir = tempfile.mkdtemp(prefix=f"cell2fire_output_{episode_id}", dir = HOME_DIR2)
    temp_output_base_dir = tempfile.mkdtemp(prefix=f"cell2fire_output_base_{episode_id}", dir = HOME_DIR2)
    
    
    try:
        shutil.copytree(input_folder, temp_work_dir, dirs_exist_ok = True)
    except Exception as e:
        print("Error during copytree:", e)
        raise
    
    
    try:
        action_indices, log_prob, value, continuous_action = agent.select_action(state, tabular_tensor, mask)
        true_reward = agent.simulate_fire_episode(action_indices, work_folder=temp_work_dir, output_folder = temp_output_dir, output_folder_base = temp_output_base_dir)
        
        if true_reward is None: # Check if reward is None
            print("Warning: Reward is None from simulate_fire_episode. Episode failed.")
            shutil.rmtree(temp_work_dir, ignore_errors=True) # Clean up work folder if episode failed
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            shutil.rmtree(temp_output_base_dir, ignore_errors=True)
            return None # Return None for the entire trajectory

    finally:
        # Clean up the temporary folder after simulation
        shutil.rmtree(temp_work_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        shutil.rmtree(temp_output_base_dir, ignore_errors=True)
       
        if os.path.exists(temp_work_dir):
           print(f"Warning: Work folder already exists before creation: {temp_work_dir}. This should not happen with UUIDs.")
       # print("DELETED", os.listdir(temp_work_dir))
    done = torch.tensor(1, dtype=torch.float32, device=agent.device)
    
    return {
        'state': state.detach(),
        'action': action_indices.detach(),
        'log_prob': log_prob.detach(),
        'value': value.detach(),
        'reward': torch.tensor([true_reward], dtype=torch.float32),
        'done': done,
        'weather': tabular_tensor.detach(),
        'mask': mask.detach(),
        'true_reward': torch.tensor([true_reward], dtype=torch.float32),
        'continuous_action': continuous_action.detach()
    }
    

def main(args):
    input_dir = args['input_dir'] # e.g Sub20x20
    output_dir = args['output_dir']
    checkpoint_path = args['checkpoint_path']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = open(f'{output_dir}/Epoch_Stats.csv','w')
    output_file.write('epoch,reward,loss,policy_loss,value_loss,entropy\n')

    # Hyperparameters
    num_epochs = int(args['num_epochs'])
    episodes_per_epoch = int(args['episodes'])

    # Initialize PPO Agent (update input channels if needed)
    new_folder=f'{input_dir}_Test/'
    input_folder_final=f'{input_dir}/'
    output_folder=f'{output_dir}'
    output_folder_base=f'{output_dir}_base/'

    # params = load_best_hyperparams()
    # agent = PPOAgent(input_folder_final, new_folder, output_folder,output_folder_base,
    #                  input_channels=4, learned_reward=False,
    #                  lr=params['lr'],
    #                  clip_epsilon=params["clip_epsilon"],
    #                  value_loss_coef=params["value_loss_coef"],
    #                  entropy_coef=params["entropy_coef"],
    #                  gamma=params["gamma"],
    #                  gae_lambda=params["gae_lambda"],
    #                  scheduler_type=params["scheduler"],
    #                  T_max=params["T_max"],      
    #                  eta_min=params["eta_min"],
    #                  stochastic=args['stochastic'],
    #                  normalise_rewards=args['normalise_rewards'],
    #                  single_sim=args['single_sim']
    #                  )
    
    agent = PPOAgent(input_folder_final, new_folder, output_folder,output_folder_base,
                     input_channels=4, learned_reward=False, scheduler_type='cosine')
    
    csvf = "episode_results.csv"
    csv_file = os.path.join(f"{output_dir}",csvf)
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
    for epoch in range(start_epoch, num_epochs):
        trajectories = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': [],
            'masks': [],
            'true_rewards': [],
            'weather': [],
            'continuous_action': []
        }
        epoch_rewards = []
        epoch_values = []
            
        total_reward = 0.0
    
        folder_sample_from = os.path.join(input_dir, "Weathers")
        folder_stored = os.path.join(input_dir, "Weathers_Stored")
        # if os.path.exists(folder_sample_from) and not os.path.exists(folder_stored):
        #     os.rename(folder_sample_from, folder_stored)
        # print('Populating Weathers_Stored...')
        # while not os.path.exists(folder_stored):
        #     continue
        # print('Weathers_Stored populated!')
        print(os.listdir(folder_stored))
        tensor_data = load_random_csv_as_tensor(folder_sample_from, folder_stored, drop_first_n_cols=2, has_header=True)
        tabular_tensor = tensor_data.view(1, 8, 11)
        epoch_rewards = []
        epoch_values = []
        start_time = time.time()
        with TPE(max_workers=mp.cpu_count()) as executor:
            
            futures = [executor.submit(simulate_single_episode, agent,
                                   tensor_input.clone(), tabular_tensor, mask, input_folder_final)
                   for _ in range(episodes_per_epoch)]
           
            results = [future.result() for future in futures]
        nones = 0
        for res in results:
            if res is None:
                nones+=1
                continue
            trajectories['states'].append(res['state'])
            trajectories['actions'].append(res['action'])
            trajectories['log_probs'].append(res['log_prob'])
            trajectories['values'].append(res['value'])
            trajectories['rewards'].append(res['reward'])
            trajectories['dones'].append(res['done'])
            trajectories['weather'].append(res['weather'])
            trajectories['masks'].append(res['mask'])
            trajectories['continuous_action'].append(res['continuous_action'])
            trajectories['true_rewards'].append(res['true_reward'])
            total_reward += res['reward'].item()
        trajectories['states'] = torch.cat(trajectories['states'], dim=0)
        trajectories['actions'] = torch.stack(trajectories['actions'], dim=0)
        trajectories['log_probs'] = torch.stack(trajectories['log_probs'], dim=0)
        trajectories['values'] = torch.cat(trajectories['values'], dim=0)
        trajectories['rewards'] = torch.cat(trajectories['rewards'], dim=0).squeeze(-1)
      #  rewards = (trajectories['rewards'] - trajectories['rewards'].mean()) / (trajectories['rewards'].std() + 1e-8)
      #  trajectories['rewards'] = rewards
        trajectories['dones'] = torch.tensor(trajectories['dones'], dtype=torch.float32, device=agent.device)
        trajectories['masks'] = torch.cat(trajectories['masks'], dim=0)
        trajectories['continuous_action'] = torch.cat(trajectories['continuous_action'], dim=0)
        trajectories['weather'] = torch.cat(trajectories['weather'], dim=0)
        trajectories['true_rewards'] = torch.cat(trajectories['true_rewards'], dim=0).squeeze(-1)


        avg_loss, avg_policy_loss, avg_value_loss, avg_entropy = agent.update(trajectories)
        avg_reward = (total_reward) / (episodes_per_epoch -nones )
        print(f"Epoch {epoch+1}/{num_epochs} - Average True Reward: {avg_reward:.4f}")


        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            for ep, (r, v) in enumerate(zip(epoch_rewards, epoch_values)):
                identifier = f"Epoch_{epoch+1}_Episode_{ep+1}"
                writer.writerow([identifier, r, v])
        save_checkpoint(agent, epoch+1, checkpoint_dir = f"{input_dir}_Test/Checkpoints")

        output_file.write(f"{epoch+1},{avg_reward:.4f},{avg_loss:.4f},{avg_policy_loss:.4f},{avg_value_loss:.4f},{avg_entropy:.4f}\n")
        output_file.flush() 
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    

    final_path = "final_model.pt"
    torch.save(agent.network.state_dict(), final_path)
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
    parser.add_argument('-s', '--stochastic', help='True to enable stochastic Cell2Fire sim', action='store_true', default=False)
    parser.add_argument('-nr', '--normalise_rewards', help='Whether to normalise rewards for PPOAgent', action='store_true', default=False)
    parser.add_argument('-g', '--single_sim', help='Whether to run a single or double Cell2Fire sim', action='store_true', default=False)
    args = vars(parser.parse_args())
    main(args)
