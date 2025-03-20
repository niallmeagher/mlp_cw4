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
import difflib
import uuid
import csv
import tempfile
import concurrent.futures
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from concurrent.futures import ProcessPoolExecutor as PPE
from concurrent.futures import ThreadPoolExecutor as TPE
from torch.distributions import MultivariateNormal
import multiprocessing as mp
username = os.getenv('USER')
HOME_DIR = os.path.join('/disk/scratch', username,'Cell2Fire', 'cell2fire', 'Cell2FireC') + '/'


class RewardFunction(nn.Module):
    def __init__(self, state_channels=1, state_size=20, num_actions=400):
        super(RewardFunction, self).__init__()
        self.conv1 = nn.Conv2d(
            state_channels, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(800 + num_actions, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state, action):
        """
        state: tensor of shape (B, 1, 20, 20)
        action: tensor of shape (B,) with integer actions in [0, num_actions-1]
        """
        B = state.size(0)
        x = F.relu(self.conv1(state))
        x = self.pool(x)
        x = x.view(B, -1)
        # One-hot encode the action.
        one_hot = torch.zeros(B, 400).to(state.device)
        one_hot.scatter_(1, action.unsqueeze(1), 1)
        x = torch.cat([x, one_hot], dim=1)
        x = F.relu(self.fc1(x))
        reward = self.fc2(x)
        return reward


class PPOAgent:    
    def __init__(self, input_folder, new_folder, output_folder, output_folder_base, input_channels=1, num_actions=400, lr=3e-4, clip_epsilon=0.1,
                 value_loss_coef=0.5, entropy_coef=0.005, gamma=0.99, update_epochs=5, learned_reward=False,scheduler_type="cosine",T_max=10,eta_min=1e-5,
                 gae_lambda=0.95, stochastic=False, normalise_rewards=False, single_sim=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(input_channels, num_actions, tabular=True).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.update_epochs = update_epochs
        self.learned_reward = learned_reward
        self.input_folder = input_folder
        self.new_folder = new_folder
        self.output_folder = output_folder
        self.output_folder_base = output_folder_base

        self.num_gpus = torch.cuda.device_count()
        
        if self.num_gpus > 1:
            self.network = nn.DataParallel(self.network)
        self.network.to(self.device)

        self.gae_lambda = gae_lambda

        self.stochastic = stochastic
        self.normalise_rewards = normalise_rewards
        self.single_sim = single_sim

        self.scheduler = None
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.1
            )

        if self.learned_reward:
            self.reward_net = RewardFunction(state_channels=input_channels, state_size=20, num_actions=num_actions).to(self.device)
            self.reward_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=lr)
        else:
            self.reward_net = None

    def read_asc_file(self, filename):
        with open(filename, 'r') as f:
            header = [next(f) for _ in range(6)]
            data = np.loadtxt(f, dtype=int)
        return header, data

    def write_asc_file(self, filename, header, data):
        with open(filename, 'w') as f:
            f.writelines(header)
            np.savetxt(f, data, fmt='%d')


    def modify_csv(self, filename_input,filename_output, indices, new_value):
        with open(filename_input, 'r') as infile:
            reader = csv.reader(infile)
            rows = list(reader)

        for index in indices:
            row_idx = index - 1
            if 0 <= row_idx < len(rows):
                rows[row_idx][0] = new_value
    
        with open(filename_output, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(rows)

    def modify_first_column(self, filename_input,filename_output, topk_integers, is_csv=True):
        with open(filename_input, 'r') as f:
            lines = f.readlines()

        header = None
        start_idx = 1 if is_csv else 0
        if is_csv:
            header = lines[0]
        delimiter = ',' if is_csv else None
        data = [line.strip().split(delimiter) for line in lines[start_idx:]]

        first_col = np.array([row[0] for row in data])

    # Modify values based on indices
        if is_csv == True:
            for idx in topk_integers:
                if 0 <= idx < len(first_col):  # Ensure index is within bounds
                    first_col[idx] = "NF"  # Example: Modify by doubling the value
        else:
            for idx in topk_integers:
                if 0 <= idx < len(first_col):  # Ensure index is within bounds
                    first_col[idx] = "nf"


    # Update first column in data
        for i, row in enumerate(data):
            row[0] = str(first_col[i])  # Convert modified values back to string

    # Convert data back to text format
        modified_lines = [",".join(row) + "\n" if is_csv else " ".join(row) + "\n" for row in data]

    # Write back to the same file
        with open(filename_output, 'w') as f:
            if is_csv:
                f.write(header)  # Write header back for CSV
            f.writelines(modified_lines)  # Write modified data

    def run_random_cell2fire_and_analyze(self, topk_indices, parallel = True, stochastic = True, work_folder = None, output_folder = None, output_folder_base = None,
                                         normalise_rewards=False):
        num_grids = 10
        work_folder = work_folder or self.new_folder 
        
        self.modify_csv(os.path.join(work_folder, "Data.csv"),os.path.join(work_folder, "Data.csv"), topk_indices, 'NF')
        self.modify_first_column(os.path.join(work_folder, "Data.dat"),os.path.join(work_folder, "Data.dat"), topk_indices, is_csv=False)
        
        
        if stochastic == True:
            FPL = str(np.round(np.random.uniform(0.5, 3.0), 2))
            ROS = str(np.round(np.random.uniform(0.0, 1.0), 2))
            IR = str(np.random.randint(1, 6))
            HF = str(np.round(np.random.uniform(0.5, 2.0), 2))
            seed = str(np.random.randint(1, 7))
            FF = str(np.round(np.random.uniform(0.5, 2.0), 2))
            BF = str(np.round(np.random.uniform(0.5, 2.0), 2))
            EF = str(np.round(np.random.uniform(0.5, 2.0), 2))
        else:
            FPL = str(np.round(np.random.uniform(0.5, 3.0), 2))
            ROS = str(0.1)
            IR = str(4)
            HF = str(1.2)
            seed = str(np.random.randint(1, 7))
            FF = str(1.2)
            BF = str(1.2)
            EF = str(1.2)

        def run_command(command):
            result = subprocess.run(command, check=True,  # Set check=False to avoid raising exception immediately
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Command failed: {command}")
                print("Stdout:", result.stdout)
                print("Stderr:", result.stderr)
            return result

        try:
            cmd = [
                f"{HOME_DIR}./Cell2Fire",
                "--input-instance-folder", work_folder,
                "--output-folder", output_folder,
                "--ignitions",
                "--sim-years", str(1),
                "--nsims", str(num_grids),
                "--grids", str(32),
                "--final-grid",
                "--Fire-Period-Length", FPL,
                "--weather", "rows",
                "--nweathers", str(1),
                "--output-messages",
                "--ROS-CV", ROS,
                "--seed", seed,
                "--IgnitionRad", IR,
                "--HFactor", HF,
                "--FFactor", FF,
                "--BFactor", BF,
                "--EFactor", EF
            ]

            cmd_base = [
                f"{HOME_DIR}./Cell2Fire",
                "--input-instance-folder", self.input_folder,
                "--output-folder", output_folder_base,
                "--ignitions",
                "--sim-years", str(1),
                "--nsims", str(num_grids),
                "--grids", str(32),
                "--final-grid",
                "--Fire-Period-Length", FPL,
                "--weather", "rows",
                "--nweathers", str(1),
                "--output-messages",
                "--ROS-CV", ROS,
                "--seed", seed,
                "--IgnitionRad", IR,
                "--HFactor", HF,
                "--FFactor", FF,
                "--BFactor", BF,
                "--EFactor", EF
            ]
            if parallel == False:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(cmd_base, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                with TPE(max_workers=mp.cpu_count()) as executor:
                    future1 = executor.submit(run_command, cmd)           
                    future2 = executor.submit(run_command, cmd_base)          
                    concurrent.futures.wait([future1, future2])

        except subprocess.CalledProcessError as e:
            print("Exception raised")
        
            return None
        
        base_grids_folder = os.path.join(output_folder_base, "Grids")
        firebreak_grids_folder = os.path.join(output_folder, "Grids")
        computed_values = []
        
        for i in range(1, num_grids + 1):
            csv_file_base = os.path.join(base_grids_folder, f"Grids{i}", "ForestGrid08.csv")
            csv_file_FB = os.path.join(firebreak_grids_folder, f"Grids{i}", "ForestGrid08.csv")
            if not os.path.exists(csv_file_base):
                continue
            try:
                data_base = np.loadtxt(csv_file_base, delimiter=',')
                data_FB = np.loadtxt(csv_file_FB, delimiter=',')
            except Exception as e:
                continue

            flat_data_base = data_base.flatten()
            total_zeros_base = np.sum(flat_data_base == 0)
            total_ones_base = np.sum(flat_data_base == 1)
            total_base = total_ones_base +total_zeros_base 
            prop_ones_base = total_ones_base/total_base
            prop_base = (1/(prop_ones_base+ 1e-8)) -1

            flat_data_FB = data_FB.flatten()
            total_zeros_FB = np.sum(flat_data_FB == 0)
            total_ones_FB = np.sum(flat_data_FB == 1)
            total_FB = total_ones_FB + total_zeros_FB
            prop_ones_FB = total_ones_FB/total_FB
            prop_FB = (1/(prop_ones_FB+ 1e-8)) -1
            difference = total_ones_base - total_ones_FB
            if normalise_rewards:
                difference /= total_ones_base
            if total_FB == 0:
                continue

            prop_ones_base = total_ones_base / total_base
            penalty_value = -0
            rows, cols = data_FB.shape
            
           # penalty = -0.1
           # for index in topk_indices:
          #      r, c = index // cols, index % cols
          #      neighbors = data_FB[max(0, r - 1): min(rows, r + 2), max(0, c - 1): min(cols, c + 2)]
           #     if np.all(neighbors == 0):  
          #          penalty += penalty_value
          #  difference += penalty
            
            computed_values.append(difference)
            print("DifferenceValue:", difference)
        if not computed_values:
            return None

        final_average = np.mean(computed_values)
        
        print("FINAL", final_average)
        return final_average
    

    def simulate_fire_episode(self, action_indices, work_folder=None, output_folder = None, output_folder_base = None):
    
        header, grid = self.read_asc_file(os.path.join(work_folder, "Forest.asc"))
      
        H, W = grid.shape  # Assuming 20x20 grid
        rows = (action_indices // W).cpu().numpy()
        cols = (action_indices % W).cpu().numpy()

        #reward = self.run_random_cell2fire_and_analyze(action_indices.cpu().numpy())
        grid[rows, cols] = 101
        self.write_asc_file(os.path.join(work_folder, "Forest.asc"), header, grid)
        if self.single_sim:
            reward = self.run_single_cell2fire_and_analyze(action_indices,parallel=True,
                                                            stochastic=self.stochastic,
                                                            normalise_rewards=self.normalise_rewards,
                                                            work_folder=work_folder, output_folder = output_folder)
        else:
            reward = self.run_random_cell2fire_and_analyze(action_indices,
                                                            parallel=True,
                                                            stochastic=self.stochastic,
                                                            normalise_rewards=self.normalise_rewards,
                                                            work_folder=work_folder, output_folder = output_folder, output_folder_base= output_folder_base)
        
        return reward
    
    def select_action(self, state, weather=None, mask=None):
        state = state.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if weather is not None:
            weather = weather.to(self.device)

    # Forward pass to get actor logits and value
        actor_logits, value = self.network(state, tabular=weather, mask=mask)

        probs = F.softmax(actor_logits, dim=1)
        num_samples = 20
        topk_indices = torch.multinomial(probs, num_samples=num_samples, replacement=False)
        topk_indices = topk_indices.squeeze(0)
        dist = Categorical(probs=probs)
        log_prob = dist.log_prob(topk_indices).sum()
     
        #return topk_indices, log_prob, value, continuous_action
        return topk_indices, log_prob, value, actor_logits
        
        

    def reward_function(self, state, action):
        if self.learned_reward:
            state = state.to(self.device)
            action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
            pred_reward = self.reward_net(state, action_tensor)
            return pred_reward
        else:
            TARGET_ACTION = 200
            true_reward = 1 - (abs(action - TARGET_ACTION) / TARGET_ACTION)
            true_reward = max(0.0, true_reward)
            return torch.tensor(true_reward, dtype=torch.float32, device=self.device)

    def compute_gae(self, rewards, dones, values, next_value):
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).
        Args:
            rewards (Tensor): shape (T,)
            dones (Tensor): shape (T,) with 1 if episode terminated, 0 otherwise.
            values (Tensor): shape (T,) with value estimates.
            next_value (Tensor): scalar, the value for the state following the last time step.
        Returns:
            advantages (Tensor): shape (T,)
            returns (Tensor): shape (T,)
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=self.device)
        for t in reversed(range(T)):
            mask = 1 - dones[t]
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns
    
    def update(self, trajectories):
        states = trajectories['states'].to(self.device)
        masks = trajectories['masks'].to(self.device)
        weather = trajectories['weather'].to(self.device)
        actions = trajectories['actions'].to(self.device)
        continuous_actions = trajectories['continuous_action'].to(self.device)
        old_log_probs = trajectories['log_probs'].to(self.device).detach()
        rewards = trajectories['rewards'].to(self.device)
        dones = trajectories['dones'].to(self.device)
        old_values = trajectories['values'].to(self.device).squeeze(-1).detach()

        with torch.no_grad():
            next_value = self.network(states[-1:], tabular=weather[-1:], mask=masks[-1:])[1].detach().squeeze()

        advantages, returns = self.compute_gae(rewards, dones, old_values, next_value)
      
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = returns.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        scaler = GradScaler()
        policy_losses = []
        value_losses = []
        entropies = []
        losses = []

        for _ in range(self.update_epochs):
            actor_logits, values = self.network(states, tabular=weather, mask=masks)
            dist_softmax = F.softmax(actor_logits,dim=1)
            dist = Categorical(probs = dist_softmax)

            new_log_probs = []
            entropies2 = []
            for i in range(states.size(0)):
                state_logits, _ = self.network(states[i:i+1], tabular=weather[i:i+1],mask=masks[i:i+1] if masks is not None else None)
                new_probs = F.softmax(state_logits, dim=1)
                new_dist = Categorical(probs=new_probs)
        
                new_log_probs.append(new_dist.log_prob(actions[i]).sum())
                #For new entropy
                entropies2.append(new_dist.entropy())
        
            new_log_probs = torch.stack(new_log_probs)
            #New Entropy
            entropy = torch.stack(entropies2).mean()
            
            
            batch_size = states.size(0)
            flat_actions = actions.view(-1)
            new_log_probs = []
            for i in range(batch_size):
                batch_actions = actions[i]  # Get all actions for this state
                state_probs = dist_softmax[i:i+1]  # Get probabilities for this state
                state_dist = Categorical(probs=state_probs)
            # Calculate log prob for each action and sum them
                state_log_probs = state_dist.log_prob(batch_actions).sum()
                new_log_probs.append(state_log_probs)
        
            new_log_probs = torch.stack(new_log_probs)
        
            entropy = dist.entropy().mean()
            
            #For old entropy
           # entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
        
            value_loss = F.mse_loss(values.squeeze(-1), returns)
        
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        
        # Optimize
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            scaler.step(self.optimizer)
            scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()  # Update LR after each epoch
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            losses.append(loss.item())

        
            print("LOSS", policy_loss.item(), self.value_loss_coef, value_loss.item(), self.entropy_coef, entropy.item())
        avg_loss = np.mean(losses)
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy = np.mean(entropies)
        return avg_loss, avg_policy_loss, avg_value_loss, avg_entropy

    def simulate_test_episode(self, state, action):
        TARGET_ACTION = 200
        true_reward = 1 - (abs(action - TARGET_ACTION) / TARGET_ACTION)
        true_reward = max(0.0, true_reward)
        return torch.tensor(true_reward, dtype=torch.float32)
    
    def run_single_cell2fire_and_analyze(self, topk_indices, parallel = True, stochastic = True, work_folder = None, output_folder = None, output_folder_base = None,
                                         normalise_rewards=False):
        num_grids = 10
        work_folder = work_folder or self.new_folder 
        
        self.modify_csv(os.path.join(work_folder, "Data.csv"),os.path.join(work_folder, "Data.csv"), topk_indices, 'NF')
        self.modify_first_column(os.path.join(work_folder, "Data.dat"),os.path.join(work_folder, "Data.dat"), topk_indices, is_csv=False)
        
        
        if stochastic == True:
            FPL = str(np.round(np.random.uniform(0.5, 3.0), 2))
            ROS = str(np.round(np.random.uniform(0.0, 1.0), 2))
            IR = str(np.random.randint(1, 6))
            HF = str(np.round(np.random.uniform(0.5, 2.0), 2))
            seed = str(np.random.randint(1, 7))
            FF = str(np.round(np.random.uniform(0.5, 2.0), 2))
            BF = str(np.round(np.random.uniform(0.5, 2.0), 2))
            EF = str(np.round(np.random.uniform(0.5, 2.0), 2))
        else:
            FPL = str(np.round(np.random.uniform(0.5, 3.0), 2))
            ROS = str(0.1)
            IR = str(4)
            HF = str(1.2)
            seed = str(np.random.randint(1, 7))
            FF = str(1.2)
            BF = str(1.2)
            EF = str(1.2)

        def run_command(command):
            result = subprocess.run(command, check=True,  # Set check=False to avoid raising exception immediately
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Command failed: {command}")
                print("Stdout:", result.stdout)
                print("Stderr:", result.stderr)
            return result

        try:
            cmd = [
                f"{HOME_DIR}./Cell2Fire",
                "--input-instance-folder", work_folder,
                "--output-folder", output_folder,
                "--ignitions",
                "--sim-years", str(1),
                "--nsims", str(num_grids),
                "--grids", str(32),
                "--final-grid",
                "--Fire-Period-Length", FPL,
                "--weather", "rows",
                "--nweathers", str(1),
                "--output-messages",
                "--ROS-CV", ROS,
                "--seed", seed,
                "--IgnitionRad", IR,
                "--HFactor", HF,
                "--FFactor", FF,
                "--BFactor", BF,
                "--EFactor", EF
            ]

            if parallel == False:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                with TPE(max_workers=mp.cpu_count()) as executor:
                    future1 = executor.submit(run_command, cmd)           
                    concurrent.futures.wait([future1])

        except subprocess.CalledProcessError as e:
            print("Exception raised")
        
            return None
        
        firebreak_grids_folder = os.path.join(output_folder, "Grids")
        computed_values = []
        
        for i in range(1, num_grids + 1):
            csv_file_FB = os.path.join(firebreak_grids_folder, f"Grids{i}", "ForestGrid08.csv")
            try:
                data_FB = np.loadtxt(csv_file_FB, delimiter=',')
            except Exception as e:
                continue

            flat_data_FB = data_FB.flatten()
            total_zeros_FB = np.sum(flat_data_FB == 0)
            total_ones_FB = np.sum(flat_data_FB == 1)
            total_FB = total_ones_FB + total_zeros_FB
            prop_ones_FB = total_ones_FB/total_FB
            prop_FB = (1/(prop_ones_FB+ 1e-8)) -1
            if total_FB == 0:
                continue

            penalty_value = -0
            rows, cols = data_FB.shape
            
            difference = 0
            penalty = -0.1
            for index in topk_indices:
               r, c = index // cols, index % cols
               neighbors = data_FB[max(0, r - 1): min(rows, r + 2), max(0, c - 1): min(cols, c + 2)]
               if np.all(neighbors == 0):  
                   penalty += penalty_value
            difference += penalty
            
            computed_values.append(difference)
            print("DifferenceValue:", difference)
        if not computed_values:
            return None

        final_average = np.mean(computed_values)
        
        print("FINAL", final_average)
        return final_average

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

