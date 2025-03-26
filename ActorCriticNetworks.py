import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# -----------------------------
# 1. Define the Actor-Critic Network
# -----------------------------


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_channels=1, num_actions=1600, tabular = False):
        super(ActorCriticNetwork, self).__init__()
        # Shared CNN Backbone
        self.tabular = tabular
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=3, kernel_size=2, stride=1, padding=0)
        # Update pool to use ceil_mode=True
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        combined_feature_size = 16*9*9
        if self.tabular == True:

            self.tab_fc1 = nn.Linear(8 * 11, 128)
            self.tab_fc2 = nn.Linear(128, 64)
        
        # Combined feature size will be cnn_feature_size + tab_feature_size
            combined_feature_size = 16 * 9 * 9 + 64

        # Actor branch: expects flattened feature size of 16 * 4 * 4 = 256
        self.actor_fc1 = nn.Linear(combined_feature_size, 512)
        self.bn1 = nn.LayerNorm(512)
        self.actor_fc2 = nn.Linear(512, 128)
        self.bn2 = nn.LayerNorm(128)
        self.actor_out = nn.Linear(128, num_actions)

        # Critic branch: same input dimensions
        self.critic_fc1 = nn.Linear(combined_feature_size, 512)
        self.bn1_2 = nn.LayerNorm(512)
        self.critic_fc2 = nn.Linear(512, 128)
        self.bn2_2 = nn.LayerNorm(128)
        self.critic_out = nn.Linear(128, 1)

    def forward(self, x, tabular =None, mask=None):
        # Shared CNN forward pass
        x = x.float()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Now x should have shape (B, 256)
        if self.tabular == True:

            tab = tabular.float()
            tab = tab.view(tab.size(0), -1)  # Flatten to (B, 88)
            tab = F.relu(self.tab_fc1(tab))
            tab = F.relu(self.tab_fc2(tab))
            combined = torch.cat([x, tab], dim=1)
        else:
            combined = x
        # Actor branch
        actor_hidden = F.relu(self.actor_fc1(combined))
        actor_hidden = self.bn1(actor_hidden)
        actor_hidden = F.relu(self.actor_fc2(actor_hidden))
        actor_logits = self.actor_out(actor_hidden)  # (B, num_actions)
        actor_logits = actor_logits
        if mask is not None:
            actor_logits = actor_logits.masked_fill(mask == 0, -1e10)
        #dist = Categorical(logits=actor_logits)

        # Critic branch
        critic_hidden = F.relu(self.critic_fc1(combined))
        critic_hidden = self.bn1_2(critic_hidden)
        critic_hidden = F.relu(self.critic_fc2(critic_hidden))
        value = self.critic_out(critic_hidden)  # (B, 1)

        #return dist, value
        return actor_logits, value
