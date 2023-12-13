import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(input_shape), 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_actions)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * input_shape[0] * input_shape[1], num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

