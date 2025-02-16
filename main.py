import os
import glob
import torch
import numpy as np

import subprocess
from PPOAgents import PPOAgent, RewardFunction  # Make sure your PPOAgent is defined and importable
dir = "/home/s2686742/Cell2Fire/cell2fire/Cell2FireC/"

import subprocess
import numpy as np
import glob
import os

def run_random_cell2fire_and_analyze():
    # Define the input and output directories (adjust as needed)
    input_folder = "/home/s2686742/Cell2Fire/data/Sub40x40/"
    output_folder = "/home/s2686742/Cell2Fire/results/Sub40x40v3"
    
    # Randomly assign values to numeric parameters using numpy.random:
    sim_years       = int(np.random.randint(1, 6))             # 1 to 5 years
    nsims           = int(np.random.randint(1, 11))            # 1 to 10 simulations
    grids           = int(np.random.randint(5, 21))            # 5 to 20 grids
    fire_period_len = np.round(np.random.uniform(0.5, 3.0), 2)   # float between 0.5 and 3.0
    nweathers       = int(np.random.randint(1, 6))             # 1 to 5 weather files
    ros_cv          = np.round(np.random.uniform(0.0, 1.0), 2)    # float between 0 and 1
    seed            = int(np.random.randint(1, 1001))            # seed between 1 and 1000
    ignition_rad    = int(np.random.randint(1, 6))             # ignition radius between 1 and 5
    hfactor         = np.round(np.random.uniform(0.5, 2.0), 2)
    ffactor         = np.round(np.random.uniform(0.5, 2.0), 2)
    bfactor         = np.round(np.random.uniform(0.5, 2.0), 2)
    efactor         = np.round(np.random.uniform(0.5, 2.0), 2)
    
    # Construct the command as a list (to avoid shell quoting issues)
    cmd = [
        "/home/s2686742/Cell2Fire/cell2fire/Cell2FireC/./Cell2Fire",
        "--input-instance-folder", input_folder,
        "--output-folder", output_folder,
        "--ignitions",
        "--sim-years", str(1),
        "--nsims", str(20),
        "--grids", str(10),
        "--final-grid",
        "--Fire-Period-Length", str(fire_period_len),
        "--weather", "rows",
        "--nweathers", str(nweathers),
        "--output-messages",
        "--ROS-CV", str(0.0),
        "--seed", str(1),
        "--IgnitionRad", str(ignition_rad),
        "--HFactor", str(hfactor),
        "--FFactor", str(ffactor),
        "--BFactor", str(bfactor),
        "--EFactor", str(efactor)
    ]
    
    print("Executing command:")
    print(" ".join(cmd))
    
    # Run the command and wait for completion
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running Cell2Fire:", e)
        return None
    '''
    # After execution, locate CSV files in the output folder.
    csv_files = glob.glob(os.path.join(output_folder, "*.csv"))
    if not csv_files:
        print("No CSV files found in the output folder!")
        return None

    total_zeros = 0
    total_ones = 0
    
    # Read each CSV file using NumPy and count 0s and 1s.
    for csv_file in csv_files:
        try:
            # Assume CSV files are comma-delimited and have no header.
            data = np.loadtxt(csv_file, delimiter=',')
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
        
        # Flatten the array in case it's 2D.
        flat_data = np.array(data).flatten()
        total_zeros += np.sum(flat_data == 0)
        total_ones += np.sum(flat_data == 1)
    
    total = total_zeros + total_ones
    if total == 0:
        print("No 0s or 1s found in CSV files!")
        return None
    
    prop_zeros = total_zeros / total
    prop_ones  = total_ones / total
    
    print(f"Proportion of 0s: {prop_zeros:.2f}")
    print(f"Proportion of 1s: {prop_ones:.2f}")
    '''
    csv_file = "/home/s2686742/Cell2Fire/cell2fire/results/Sub40x40v3/Grids/Grids5/ForestGrid07.csv"
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return None
    
    # Load the CSV file using NumPy (assumes comma-delimited with no header)
    try:
        data = np.loadtxt(csv_file, delimiter=',')
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None
    
    # Flatten the data (in case it's 2D) and count 0s and 1s
    flat_data = np.array(data).flatten()
    total_zeros = np.sum(flat_data == 0)
    total_ones  = np.sum(flat_data == 1)
    
    total = total_zeros + total_ones
    if total == 0:
        print("No 0s or 1s found in the CSV file!")
        return None
    
    prop_zeros = total_zeros / total
    prop_ones  = total_ones / total
    
   # print(f"Proportion of 0s: {prop_zeros:.2f}")
    #print(f"Proportion of 1s: {prop_ones:.2f}")
    
    
    return prop_ones


def simulate_fire_episode(state, action):
    """
    Dummy simulation of fires on the grid given the current state and the action
    (firebreak placements). In an actual implementation, this function would:
      1. Modify the grid state based on the action (placing firebreaks).
      2. Simulate a set of fires at random grid locations.
      3. Evaluate how effective the firebreaks were (e.g., the damage avoided).
      4. Return an average reward for the episode.
    
    For this framework example, we simply return a dummy reward.
    """
    # Dummy implementation: return a random reward between -1 and 1.
    #cmd = "./Cell2Fire --input-instance-folder /home/s2686742/Cell2Fire/data/Sub40x40/ --output-folder ../results/Sub40x40v3 --ignitions --sim-years 1 --nsims 5 --grids 10 --final-grid --Fire-Period-Length 1.9 --weather rows --nweathers 1 --output-messages --ROS-CV 0.0 --seed 1 --IgnitionRad 2 --HFactor 1.9 --FFactor 1.0 --BFactor 1.90 --EFactor 1.9"
    reward = run_random_cell2fire_and_analyze()
    #cmd_in = dir + cmd
   # os.system(cmd_in)
    return (1/reward) -1

def main():
    # Hyperparameters
    num_epochs = 1000          # Number of PPO update cycles
    episodes_per_epoch = 10    # Number of episodes (trajectories) to collect per update

    # Initialize PPO Agent (this creates the network, optimizer, etc.)
    agent = PPOAgent(learned_reward=False)
   
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
            state = torch.zeros(1, 1, 20, 20)  # For example, an empty grid.
            # Assume all actions are valid.
            valid_actions_mask = torch.ones(1, 400)

            # Select an action.
            action, log_prob, value = agent.select_action(
                state, mask=valid_actions_mask)

            # Use the learnable reward function to predict a reward (if desired).
            pred_reward = agent.reward_function(state, action)

            # Simulate the fire episode to get a true reward.
            true_reward = agent.simulate_fire_episode(state, action)
            #true_reward = agent.simulate_test_episode(state, action)
            total_reward += true_reward.item()

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
