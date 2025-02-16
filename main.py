import os
cmd = "/home/s2686742/Cell2Fire/cell2fire/Cell2FireC/./Cell2Fire --input-instance-folder /home/s2686742/Cell2Fire/data/Sub40x40/ --output-folder ../results/Sub40x40v3 --ignitions --sim-years 1 --nsims 5 --grids 10 --final-grid --Fire-Period-Length 1.9 --weather rows --nweathers 1 --output-messages --ROS-CV 0.0 --seed 1 --IgnitionRad 2 --HFactor 1.9 --FFactor 1.0 --BFactor 1.90 --EFactor 1.9"
os.system(cmd)
print("Success")

import torch
import numpy as np

import subprocess
from PPOAgents import PPOAgent, RewardFunction  # Make sure your PPOAgent is defined and importable

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
    return np.random.uniform(-1, 1)

def main():
    # Hyperparameters
    num_epochs = 1000          # Number of PPO update cycles
    episodes_per_epoch = 10    # Number of episodes (trajectories) to collect per update

    # Initialize PPO Agent (this creates the network, optimizer, etc.)
    agent = PPOAgent(learned_reward=False)
    cmd = "./Cell2Fire --input-instance-folder /home/s2686742/Cell2Fire/data/Sub40x40/ --output-folder ../results/Sub40x40v3 --ignitions --sim-years 1 --nsims 5 --grids 10 --final-grid --Fire-Period-Length 1.9 --weather rows --nweathers 1 --output-messages --ROS-CV 0.0 --seed 1 --IgnitionRad 2 --HFactor 1.9 --FFactor 1.0 --BFactor 1.90 --EFactor 1.9"
    os.system(cmd)
    print("Success")
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
            #true_reward = agent.simulate_fire_episode(state, action)
            true_reward = agent.simulate_test_episode(state, action)
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
