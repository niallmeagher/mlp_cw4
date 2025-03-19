from collections import deque
import random
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import shutil
import subprocess
import os
import glob
import difflib
import uuid
import csv
import tempfile
from torch.distributions import Categorical
from ActorCriticNetworks import ActorCriticNetwork
import concurrent.futures
import multiprocessing as mp
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
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

import NNetworks
username = os.getenv('USER')
HOME_DIR = os.path.join('/disk/scratch', username,'Cell2Fire', 'cell2fire', 'Cell2FireC') + '/'

class DQNAgent:
    
    def __init__(self, input_folder, new_folder, output_folder, output_folder_base, input_channels=1, num_actions=400, lr=3e-4, clip_epsilon=0.1,
                 value_loss_coef=0.5, entropy_coef=0.005, gamma=0.99, update_epochs=5, learned_reward=False,scheduler_type="cosine",T_max=10,eta_min=1e-5,
                 network = "SmallNet"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        #Network
        if network == "SmallNet":
            self.policy_net = NNetworks.smallNet(input_channels, num_actions=num_actions).to(self.device)
            self.target_net = NNetworks.smallNet(input_channels, num_actions=num_actions).to(self.device)
        else:
            self.policy_net = NNetworks.bigNet(input_channels, num_actions=num_actions).to(self.device)
            self.target_net = NNetworks.bigNet(input_channels, num_actions=num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        #Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        #Hyperparameters
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.update_epochs = update_epochs
        self.learned_reward = learned_reward

        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995 

        self.update_step = 0  # Counter for target network updates
        self.target_update = 1000  # Update target network every 1000 steps
        
        #Folders
        self.input_folder = input_folder
        self.new_folder = new_folder
        self.output_folder = output_folder
        self.output_folder_base = output_folder_base

        #History
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64


        #GPU
        self.num_gpus = torch.cuda.device_count()        
        if self.num_gpus > 1:
            self.policy_net = nn.DataParallel(self.policy_net)
        self.policy_net.to(self.device)


        self.scheduler = None
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.1
            )

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
    
    #Simple DPV Alternative, based on the number of times a cell catches fire
    '''
    def calculate_dpv(self, work_folder, num_simulations = 10):
        dpv_values = np.zeros((20,20))
        for n in range(num_simulations):
            burned_cells = self.run_Cell2FireOnce_ReturnBurnMap(work_folder)
            for i in range(20):
                for j in range(20):
                    if burned_cells[i,j] == 1:
                        dpv_values[i,j] += 1
            print("n:", n)
        return dpv_values / num_simulations
    '''

    #Naive DPV calculation - based on the number of cells protected by each cell
    '''
    def calculate_dpv(self, work_folder, num_simulations=10):
        dpv_values = np.zeros((20, 20))
        for n in range(num_simulations):
            # Run a fire simulation
            burned_cells = self.run_Cell2FireOnce_ReturnBurnMap(work_folder)
            
            # Calculate DPV for each cell
            for i in range(20):
                for j in range(20):
                    if burned_cells[i, j] == 1:
                        # Count the number of downstream cells protected by this cell
                        protected_cells = 0
                        for di in range(-1, 2):  # Check neighboring cells
                            for dj in range(-1, 2):
                                ni, nj = i + di, j + dj
                                if 0 <= ni < 20 and 0 <= nj < 20 and burned_cells[ni, nj] == 0:
                                    protected_cells += 1
                        dpv_values[i, j] += protected_cells
        return dpv_values / num_simulations
    '''

    #Full DPV calculation - using graphs and subgraphs to calculate risk and then the DPV
    def create_forest_graph(self, grid_size):
        num_nodes = grid_size * grid_size
        adjacency_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(grid_size):
            for j in range(grid_size):
                cell_idx = i * grid_size + j
                if j < grid_size - 1:
                    adjacency_matrix[cell_idx, cell_idx + 1] = 1
                if i < grid_size - 1:
                    adjacency_matrix[cell_idx, cell_idx + grid_size] = 1
        
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        return adjacency_matrix
    
    def calculate_mst(self, adjacency_matrix, burned_cells):
        subgraph = adjacency_matrix[burned_cells, :][:, burned_cells]
        sparce_graph = csr_matrix(subgraph)
        mst = minimum_spanning_tree(sparce_graph)
        return mst
    

    def calculate_dpv(self, work_folder, num_simulations=10):
        dpv_values = np.zeros((20, 20))
        adjacency_matrix = self.create_forest_graph(20)

        for _ in range(num_simulations):
            burned_cells = self.run_Cell2FireOnce_ReturnBurnMap(work_folder)
            burned_indices = np.where(burned_cells.flatten() == 1)[0]
            
            # Create a mapping from original indices to subgraph indices
            index_mapping = {original_idx: subgraph_idx for subgraph_idx, original_idx in enumerate(burned_indices)}
            
            # Calculate MST on the subgraph of burned cells
            subgraph = adjacency_matrix[burned_indices, :][:, burned_indices]
            sparse_graph = csr_matrix(subgraph)
            mst = minimum_spanning_tree(sparse_graph)

            # Calculate DPV for each burned cell
            for original_idx in burned_indices:
                subgraph_idx = index_mapping[original_idx]
                mst_rooted = mst[subgraph_idx, :]
                dpv_values[original_idx // 20, original_idx % 20] += np.sum(mst_rooted)
        
        return dpv_values / num_simulations
    
    def select_firebreaks_dpv(self, dpv_values, num_firebreaks=20):
        # Normalize DPV values (optional)
        dpv_values_normalized = (dpv_values - np.min(dpv_values)) / (np.max(dpv_values) - np.min(dpv_values))
        # Select top-k cells with the highest DPV values
        topk_indices = np.argsort(dpv_values_normalized.flatten())[-num_firebreaks:]
        return topk_indices
    
    def generate_demonstrations(self, inputTensors, num_demos = 1000):
        demonstrations = []
        for n in range(num_demos):
            state = inputTensors 
            dpv_values = self.calculate_dpv(self.new_folder)
            topk_indices = self.select_firebreaks_dpv(dpv_values)

            demonstrations.append((state, topk_indices))
            print("Demonstration:", n)
        print("demonstrations generated")
        return demonstrations

    def calculate_burnedArea(self, grid):
        flat_data = grid.flatten()
        total_zeros = np.sum(flat_data == 0)
        total_ones = np.sum(flat_data == 1)
        total_base = total_ones + total_zeros
        return total_ones

    def run_Cell2FireOnceWithBreaks(self, topk_indices, work_folder = None):
        work_folder = work_folder or self.new_folder
        if not os.path.exists(work_folder):
            try:
                shutil.copytree(self.input_folder, work_folder)
            except Exception as e:
                print(f"Error copying folder: {e}")
                return None
        
        self.modify_csv(os.path.join(self.input_folder, "Data.csv"),os.path.join(work_folder, "Data.csv"), topk_indices, 'NF')
        self.modify_first_column(os.path.join(self.input_folder, "Data.dat"),os.path.join(work_folder, "Data.dat"), topk_indices, is_csv=False)
        
        try:
            cmd = [
                f"{HOME_DIR}./Cell2Fire",
                "--input-instance-folder", self.new_folder,
                "--output-folder", self.output_folder,
                "--ignitions",
                "--sim-years", str(1),
                "--nsims", str(1),
                "--grids", str(32),
                "--final-grid",
                "--Fire-Period-Length", str(1.0),
                "--weather", "rows",
                "--nweathers", str(1),
                "--output-messages",
                "--ROS-CV", str(0.0),
                "--seed", str(1),
                "--IgnitionRad", str(4),
                "--HFactor", str(1.2),
                "--FFactor", str(1.2),
                "--BFactor", str(1.2),
                "--EFactor", str(1.2)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            return None
        
        firebreak_grids_folder = os.path.join(self.output_folder, "Grids")
        csv_file = os.path.join(firebreak_grids_folder, f"Grids{1}", "ForestGrid08.csv")
        if os.path.exists(csv_file):
            try:
                data = np.loadtxt(csv_file, delimiter=',')
            except Exception as e:
                return None, None
        return self.calculate_burnedArea(data), data

    def run_Cell2FireOnce_ReturnBurnMap(self, work_folder = None, stochastic = True):
        work_folder = work_folder or self.new_folder
        if not os.path.exists(work_folder):
            try:
                shutil.copytree(self.input_folder, work_folder)
            except Exception as e:
                print(f"Error copying folder: {e}")
                return None
        
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
        
        try:
            cmd = [
                f"{HOME_DIR}./Cell2Fire",
                "--input-instance-folder", self.new_folder,
                "--output-folder", self.output_folder,
                "--ignitions",
                "--sim-years", str(1),
                "--nsims", str(1),
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
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            return None
        
        firebreak_grids_folder = os.path.join(self.output_folder, "Grids")
        csv_file = os.path.join(firebreak_grids_folder, f"Grids{1}", "ForestGrid08.csv")
        if os.path.exists(csv_file):
            try:
                data = np.loadtxt(csv_file, delimiter=',')
            except Exception as e:
                return None
        return data


    def run_random_cell2fire_and_analyze(self, topk_indices, parallel = True, stochastic = True, work_folder = None, output_folder = None, output_folder_base = None):
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
        
        print(action_indices)
        if isinstance(action_indices, list):
            action_indices = torch.tensor(action_indices, dtype=torch.long)

        H, W = grid.shape  # Assuming 20x20 grid
        rows = (action_indices // W).cpu().numpy()
        cols = (action_indices % W).cpu().numpy()

        #reward = self.run_random_cell2fire_and_analyze(action_indices.cpu().numpy())
        grid[rows, cols] = 101
        self.write_asc_file(os.path.join(work_folder, "Forest.asc"), header, grid)
        reward = self.run_random_cell2fire_and_analyze(action_indices,
                                                       parallel=True,
                                                       stochastic=False,
                                                       work_folder=work_folder, 
                                                       output_folder = output_folder, 
                                                       output_folder_base= output_folder_base)
        
        return reward
  
    def select_action(self, state, mask=None):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)

        if random.random() < self.epsilon:
            # Randomly select 20 actions (firebreak locations)
            return random.sample(range(self.policy_net.advantage_fc.out_features), 20)
        else:
            with torch.no_grad():
                # Get Q-values for all actions
                q_values = self.policy_net(state, mask=mask)
                # Select the top 20 actions with the highest Q-values
                topk_actions = torch.topk(q_values, k=20, dim=1).indices.squeeze(0).tolist()
            return topk_actions
        
    def store_transition(self, state, actions, reward, next_state, done, mask=None):
        self.replay_buffer.append((state, actions, reward, next_state, done, mask))
        

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
    
    def preTraining(self, demonstrations, num_epochs=100, margin = 0.1, l2_weight = 0.01):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.optimizer.param_groups[0]['lr'])
        for epoch in range(num_epochs):
            epochLoss = 0.0
            for state, action in demonstrations:
                state = torch.tensor(state, dtype = torch.float32).to(self.device)
                action = torch.tensor(action, dtype=torch.long).to(self.device)
                tabular = torch.zeros(1, 8, 11).to(self.device)
                action_logits, value =  self.network(state, tabular = tabular)

                logits_expanded = action_logits.repeat(action.size(0), 1)
                action_loss = F.cross_entropy(logits_expanded, action)

                demonstrator_logits = logits_expanded.gather(1, action.unsqueeze(1))
                otherLogits = logits_expanded.clone()
                otherLogits.scatter_(1, action.unsqueeze(1), -1e10)

                maxOtherLogits = otherLogits.max(dim=1)[0]

                marginLoss = F.relu(maxOtherLogits - demonstrator_logits + margin).mean()

                l2_loss = 0.0
                for param in self.network.parameters():
                    l2_loss += torch.norm(param, p=2)
                loss = action_loss + marginLoss + l2_weight * l2_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epochLoss += loss.item()
            print(epoch)
            avgLoss = epochLoss / len(demonstrations)
            print(f"Epoch {epoch}, Loss: {avgLoss}")

    def update(self):
        # Check if there are enough samples in the replay buffer
        if len(self.replay_buffer) < self.batch_size:
            print("Replay buffer has fewer samples than batch size. Skipping update.")
            return
        
        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).squeeze(1).to(self.device)  # Shape: [batch_size, channels, height, width]
        print("States shape:", states.shape)  # Debug: Should be [batch_size, channels, height, width]
        
        actions = torch.LongTensor(np.array(actions)).to(self.device)  # Shape: [batch_size, 20]
        print("Actions shape:", actions.shape)  # Debug: Should be [batch_size, 20]
        
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)  # Shape: [batch_size]
        print("Rewards shape:", rewards.shape)  # Debug: Should be [batch_size]
        
        next_states = torch.FloatTensor(np.array(next_states)).squeeze(1).to(self.device)  # Shape: [batch_size, channels, height, width]
        print("Next states shape:", next_states.shape)  # Debug: Should be [batch_size, channels, height, width]
        
        dones = torch.FloatTensor(np.array(dones)).to(self.device)  # Shape: [batch_size]
        print("Dones shape:", dones.shape)  # Debug: Should be [batch_size]
        
        masks = torch.FloatTensor(np.array(masks)).to(self.device) if masks[0] is not None else None  # Shape: [batch_size, num_actions]
        print("Masks shape:", masks.shape if masks is not None else "None")  # Debug: Should be [batch_size, num_actions] or None
        
        # Compute Q-values for the current states using the policy network
        current_q_values = self.policy_net(states, mask=masks)  # Shape: [batch_size, num_actions]
        print("Current Q-values shape:", current_q_values.shape)  # Debug: Should be [batch_size, num_actions]
        
        # Gather Q-values for the actions that were actually taken
        gathered_q_values = torch.gather(current_q_values, 1, actions)  # Shape: [batch_size, 20]
        print("Gathered Q-values shape:", gathered_q_values.shape)  # Debug: Should be [batch_size, 20]
        
        # Compute Q-values for the next states using the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states, mask=masks)  # Shape: [batch_size, num_actions]
            print("Next Q-values shape:", next_q_values.shape)  # Debug: Should be [batch_size, num_actions]
            
            # Use the maximum Q-value for the next state to compute the target
            max_next_q_values = next_q_values.max(1)[0]  # Shape: [batch_size]
            print("Max next Q-values shape:", max_next_q_values.shape)  # Debug: Should be [batch_size]
            
            # Compute the target Q-values using the Bellman equation
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values  # Shape: [batch_size]
            print("Target Q-values shape (before reshape):", target_q_values.shape)  # Debug: Should be [batch_size]
        
        # Reshape target_q_values to match gathered_q_values
        target_q_values = target_q_values.unsqueeze(1).expand(-1, 20)  # Shape: [batch_size, 20]
        print("Target Q-values shape (after reshape):", target_q_values.shape)  # Debug: Should be [batch_size, 20]
        
        # Compute the loss (mean squared error between gathered_q_values and target_q_values)
        loss = F.mse_loss(gathered_q_values, target_q_values)
        print("Loss:", loss.item())  # Debug: Print the computed loss
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay the exploration rate (epsilon)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print("Epsilon:", self.epsilon)  # Debug: Print the updated epsilon value
        
        # Update the target network periodically
        if self.update_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("Target network updated.")  # Debug: Indicate that the target network was updated
        
        # Increment the update step counter
        self.update_step += 1
        print("Update step:", self.update_step)  # Debug: Print the current update step
        
        # Return the loss for logging or monitoring
        return loss.item()

    def simulate_test_episode(self, state, action):
        TARGET_ACTION = 200
        true_reward = 1 - (abs(action - TARGET_ACTION) / TARGET_ACTION)
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

