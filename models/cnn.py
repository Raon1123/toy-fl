import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes, in_channel=3):
        super().__init__()
        
        # conv1
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

        # conv2
        self.conv2 = nn.Conv2d(6, 16, 5)

        # FC
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x