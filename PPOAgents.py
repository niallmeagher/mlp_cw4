# Load in relevant libraries, and alias where appropriate
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ActorCriticNetworks import ActorCriticNetwork


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
                 value_loss_coef=0.5, entropy_coef=0.01, gamma=0.99, update_epochs=4):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(
            input_channels, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.update_epochs = update_epochs

        self.reward_net = RewardFunction(state_channels=input_channels, state_size=20, num_actions=num_actions).to(self.device)
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=lr)

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
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def reward_function(self, state, action):
        """
        Use the learnable reward function to predict a reward given a state and action.
        """
        state = state.to(self.device)
        action_tensor = torch.tensor(
            [action], dtype=torch.long).to(self.device)
        pred_reward = self.reward_net(state, action_tensor)
        return pred_reward

    def compute_returns(self, rewards, dones, values, next_value):
        """
        Compute discounted returns.
          - rewards: list of rewards for the trajectory.
          - dones: list of booleans indicating episode termination.
          - values: list of value estimates.
          - next_value: value estimate for the state following the last state.
        """
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
        """
        Update the policy using PPO.
        Expects a dictionary `trajectories` with:
          'states': tensor of shape (batch, 1, 20, 20)
          'actions': tensor of shape (batch,) of chosen action indices
          'log_probs': tensor of shape (batch,) of log probabilities (from old policy)
          'returns': tensor of shape (batch,) computed discounted returns
          'values': tensor of shape (batch, 1) of value estimates (from old policy)
          Optionally, 'masks': tensor of shape (batch, num_actions) for action masking.
        """
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
        if 'true_rewards' in trajectories:
            # Detach states and actions so that the reward network update does not try to backpropagate through
            # the graph that was used in the policy update.
            predicted_rewards = self.reward_net(states.detach(), actions.detach())
            reward_loss = F.mse_loss(
                predicted_rewards.squeeze(-1), trajectories['true_rewards'].to(self.device))
            self.reward_optimizer.zero_grad()
            reward_loss.backward()
            self.reward_optimizer.step()
           
          
    def simulate_test_episode(self, state, action):
        """
        A test simulation function.
        In this test, we assume that the optimal action is to choose index 200.
        We define the true reward as higher when the chosen action is near 200.
        For example, we compute:
              true_reward = 1 - (abs(action - 200) / 200)
        so that an action of 200 yields a reward of 1, and an action of 0 or 400 yields 0.
        """
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