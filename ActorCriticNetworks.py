import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# -----------------------------
# 1. Define the Actor-Critic Network
# -----------------------------


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_channels=1, num_actions=400):
        super(ActorCriticNetwork, self).__init__()
        # Shared CNN Backbone
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=3, kernel_size=2, stride=1, padding=0)
        # Update pool to use ceil_mode=True
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)

        # Actor branch: expects flattened feature size of 16 * 4 * 4 = 256
        self.actor_fc1 = nn.Linear(16 * 4 * 4, 512)
        self.actor_fc2 = nn.Linear(512, 128)
        self.actor_out = nn.Linear(128, num_actions)

        # Critic branch: same input dimensions
        self.critic_fc1 = nn.Linear(16 * 4 * 4, 512)
        self.critic_fc2 = nn.Linear(512, 128)
        self.critic_out = nn.Linear(128, 1)

    def forward(self, x, mask=None):
        # Shared CNN forward pass
        x = x.float()
       # print(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Now x should have shape (B, 256)
       # print(x)
        # Actor branch
        actor_hidden = F.relu(self.actor_fc1(x))
       # print(actor_hidden)
        actor_hidden = F.relu(self.actor_fc2(actor_hidden))
        actor_logits = self.actor_out(actor_hidden)  # (B, num_actions)
#        print("ACTOR1",actor_logits)
        if mask is not None:
            actor_logits = actor_logits.masked_fill(mask == 0, -1e10)
 #       print("ACTOR", actor_logits)
        dist = Categorical(logits=actor_logits)

        # Critic branch
        critic_hidden = F.relu(self.critic_fc1(x))
        critic_hidden = F.relu(self.critic_fc2(critic_hidden))
        value = self.critic_out(critic_hidden)  # (B, 1)

        return dist, value
