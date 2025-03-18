import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class smallNet(nn.Module):
    def __init__(self, input_channels=1, input_width=20, input_height=20, num_actions=400):
        # Implementing small-net architecture
        super(smallNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(32 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)

        self.value_fc = nn.Linear(128, 1)
        self.advantage_fc = nn.Linear(128, num_actions)

    def forward(self, x, mask=None):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_fc(x)
        advantage = self.advantage_fc(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        if mask is not None:
            q_values = q_values * mask

        #probabilties = F.softmax(q_values, dim=1)

        #gridProbs = probabilties.view(20, 20, -1)

        return q_values
    
class bigNet(nn.Module):
    def __init__(self, input_channels=1, input_width=20, input_height=20, num_actions=400):
        super(bigNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=1)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=1)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.dropout3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=1)
        self.dropout4 = nn.Dropout(0.2)

        fc_input_size = 32 * 8 * 8

        self.fc1 = nn.Linear(fc_input_size, 2048)
        self.fc2 = nn.Linear(2048, 48)
        self.fc3 = nn.Linear(48, 32)

        self.value_fc = nn.Linear(32, 1)
        self.advantage_fc = nn.Linear(32, num_actions)

    def forward(self, x, mask=None):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        value = self.value_fc(x)
        advantage = self.advantage_fc(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        if mask is not None:
            q_values = q_values * mask

        return q_values