# Load in relevant libraries, and alias where appropriate
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import shutil
from ActorCriticNetworks import ActorCriticNetwork
import subprocess
import os
import glob


class RewardFunction(nn.Module):
    def __init__(self, state_channels=1, state_size=20, num_actions=400):
        super(RewardFunction, self).__init__()
        # A small CNN to process the state (grid)
        self.conv1 = nn.Conv2d(
            state_channels, 8, kernel_size=3, stride=1, padding=1)  # (B, 8, 20, 20)
        self.pool = nn.MaxPool2d(2, 2)  # → (B, 8, 10, 10)
        # After pooling, the state is flattened: 8 * 10 * 10 = 800.
        # We then concatenate a one-hot encoding of the action (size: num_actions).
        self.fc1 = nn.Linear(800 + num_actions, 128)
        self.fc2 = nn.Linear(128, 1)  # Outputs a single scalar reward

    def forward(self, state, action):
        """
        state: tensor of shape (B, 1, 20, 20)
        action: tensor of shape (B,) with integer actions in [0, num_actions-1]
        """
        B = state.size(0)
        x = F.relu(self.conv1(state))   # (B, 8, 20, 20)
        x = self.pool(x)                # (B, 8, 10, 10)
        x = x.view(B, -1)               # (B, 800)
        # One-hot encode the action.
        one_hot = torch.zeros(B, 400).to(state.device)
        one_hot.scatter_(1, action.unsqueeze(1), 1)
        x = torch.cat([x, one_hot], dim=1)  # (B, 800+400 = 1200)
        x = F.relu(self.fc1(x))
        reward = self.fc2(x)  # (B, 1)
        return reward



class PPOAgent:
    def __init__(self, input_channels=1, num_actions=400, lr=3e-4, clip_epsilon=0.2,
                 value_loss_coef=0.5, entropy_coef=0.01, gamma=0.99, update_epochs=4, learned_reward=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(input_channels, num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.update_epochs = update_epochs
        self.learned_reward = learned_reward

        if self.learned_reward:
            self.reward_net = RewardFunction(state_channels=input_channels, state_size=20, num_actions=num_actions).to(self.device)
            self.reward_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=lr)
        else:
            self.reward_net = None

    def run_random_cell2fire_and_analyze(self, state, topk_indices):
        input_folder = "/home/s2686742/Cell2Fire/data/Sub20x20/"
        new_folder = "/home/s2686742/Cell2Fire/data/Sub20x20_Test/"
        output_folder = "/home/s2686742/Cell2Fire/results/Sub20x20v1"

        if not os.path.exists(new_folder):
            try:
                shutil.copytree(input_folder, new_folder)
            except Exception as e:
                print(f"Error copying folder: {e}")
                return None
        
        asc_file = os.path.join(new_folder, "Forest.asc")
        try:
            with open(asc_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {asc_file}: {e}")
            return None

        num_header_lines = 6
        if len(lines) < num_header_lines:
            return None

        header_lines = lines[:num_header_lines]

        if hasattr(state, 'detach'):
            state = state.detach().cpu().numpy()
        state = state.squeeze()
        state = np.array(state)

        if state.shape != (20, 20):
            return None

        grid_lines = [" ".join(str(val) for val in row) + "\n" for row in state]
        new_file_content = header_lines + grid_lines

        try:
            with open(asc_file, 'w') as f:
                f.writelines(new_file_content)
        except Exception as e:
            return None

        try:
            cmd = [
                "/home/s2686742/Cell2Fire/cell2fire/Cell2FireC/./Cell2Fire",
                "--input-instance-folder", new_folder,
                "--output-folder", output_folder,
                "--ignitions",
                "--sim-years", str(np.random.randint(1, 6)),
                "--nsims", str(np.random.randint(1, 11)),
                "--grids", str(np.random.randint(5, 21)),
                "--final-grid",
                "--Fire-Period-Length", str(np.round(np.random.uniform(0.5, 3.0), 2)),
                "--weather", "rows",
                "--nweathers", str(np.random.randint(1, 6)),
                "--output-messages",
                "--ROS-CV", str(np.round(np.random.uniform(0.0, 1.0), 2)),
                "--seed", str(np.random.randint(1, 1001)),
                "--IgnitionRad", str(np.random.randint(1, 6)),
                "--HFactor", str(np.round(np.random.uniform(0.5, 2.0), 2)),
                "--FFactor", str(np.round(np.random.uniform(0.5, 2.0), 2)),
                "--BFactor", str(np.round(np.random.uniform(0.5, 2.0), 2)),
                "--EFactor", str(np.round(np.random.uniform(0.5, 2.0), 2))
            ]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            return None

        csv_file = "/home/s2686742/Cell2Fire/results/Sub20x20v1/Grids/Grids6/ForestGrid08.csv"
        if not os.path.exists(csv_file):
            return None

        try:
            data = np.loadtxt(csv_file, delimiter=',')
        except Exception as e:
            return None

        flat_data = np.array(data).flatten()
        total_zeros = np.sum(flat_data == 0)
        total_ones = np.sum(flat_data == 1)
        total = total_zeros + total_ones
        if total == 0:
            return None

        prop_ones = total_ones / total
        penalty_value = -.1  # Adjust penalty as needed
        rows, cols = data.shape
        penalty = 0
        for index in topk_indices:
            r, c = index // cols, index % cols  # Convert 1D index to 2D row/col

    # Get the 3x3 neighborhood
            neighbors = data[max(0, r - 1): min(rows, r + 2), max(0, c - 1): min(cols, c + 2)]

    # Check if at least one neighboring cell (excluding center) is 1
            if np.all(neighbors == 0):  
                penalty += penalty_value  # Sum the penalty

        
        
        return (1/prop_ones-1) + penalty


    def simulate_fire_episode(self, state, action):
        """
        Selects the top 20 values in 'action' and modifies 'state' accordingly.
        """
        # Get the top 20 highest action values and their indices
        topk_values, topk_indices = torch.topk(action.flatten(), k=20)

        # Convert flat indices to 2D coordinates
        rows = topk_indices // action.size(1)
        cols = topk_indices % action.size(1)

        # Update state based on the selected indices
        state[:, :, rows, cols] = 101

        # Run Cell2Fire simulation and analyze the results
        reward = self.run_random_cell2fire_and_analyze(state, topk_indices)

        # Compute final reward
        return reward if reward is not None else -1


    
    def select_action(self, state, mask=None):
        """
        Given a state (and an optional mask of valid actions), return:
          - the chosen action (as an integer),
          - its log-probability,
          - and the value estimate for the state.
        """
        state = state.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        dist, value = self.network(state, mask)
        probs = F.softmax(dist.logits, dim=-1)
        probs = probs.reshape(20, 20)
        action = dist.sample()
        log_prob = dist.log_prob(action)
       
        return action.item(), log_prob, value, probs

    def reward_function(self, state, action):
        """
        Use the learnable reward function to predict a reward given a state and action.
        """
        if self.learned_reward:
            # Use the reward network (make sure state and action are on the correct device)
            state = state.to(self.device)
            action_tensor = torch.tensor(
                [action], dtype=torch.long, device=self.device)
            pred_reward = self.reward_net(state, action_tensor)
            return pred_reward
        else:
            # Define your static reward logic here.
            # For example, assume the optimal action is 200:
            TARGET_ACTION = 200
            true_reward = 1 - (abs(action - TARGET_ACTION) / TARGET_ACTION)
            true_reward = max(0.0, true_reward)
            return torch.tensor(true_reward, dtype=torch.float32, device=self.device)


    def compute_returns(self, rewards, dones, values, next_value):
        
        returns = []
        R = next_value
        # Iterate in reverse (from last step to first)
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0  # If episode ended, reset the cumulative reward.
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(
            returns, dtype=torch.float32, device=self.device)
        return returns

    def update(self, trajectories):
       
        states = trajectories['states'].to(self.device)
        actions = trajectories['actions'].to(self.device)
        old_log_probs = trajectories['log_probs'].to(self.device).detach()
        returns = trajectories['returns'].to(self.device).detach()
        old_values = trajectories['values'].to(
            self.device).squeeze(-1).detach()
        advantages = returns - old_values
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)
        advantages = advantages.detach()
        masks = trajectories.get('masks', None)
        if masks is not None:
            masks = masks.to(self.device)

        for _ in range(self.update_epochs):
            # Re-evaluate actions & values with current policy
            dist, values = self.network(states, masks)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO ratio for clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (mean squared error)
            value_loss = F.mse_loss(values.squeeze(-1), returns)

            # Total loss with entropy bonus (to encourage exploration)
            loss = policy_loss + self.value_loss_coef * \
                value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # Assume trajectories include 'true_rewards' computed from simulation.
        if self.learned_reward and 'true_rewards' in trajectories:
            predicted_rewards = self.reward_net(
                states.detach(), actions.detach())
            reward_loss = F.mse_loss(predicted_rewards.squeeze(-1),
                                     trajectories['true_rewards'].to(self.device))
            self.reward_optimizer.zero_grad()
            reward_loss.backward()
            self.reward_optimizer.step()   
    def simulate_test_episode(self, state, action):
       
        TARGET_ACTION = 200
        # Compute a reward that is 1 at the target and decays linearly.
        true_reward = 1 - (abs(action - TARGET_ACTION) / TARGET_ACTION)
        # Clip reward to be at least 0.
        true_reward = max(0.0, true_reward)
        return torch.tensor(true_reward, dtype=torch.float32)
       

'''
    def reward_function(self, state, action, next_state):
        """
        Placeholder for the reward function.
        Implement your reward logic here. For example, it might depend on
        the current state, the action taken, and the next state.
        """
        reward = 0.0  # Replace with your reward logic
        return reward
'''
