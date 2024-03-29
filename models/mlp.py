import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, hidden_layers=[32*32*3, 1024, 10], activation='relu'):
        # TODO: May add activation functions
        super(MLP, self).__init__()

        self.fcs = []
        self.layer_depth = len(hidden_layers)

        for depth in range(self.layer_depth-1):
            self.fcs.append(nn.Linear(hidden_layers[depth], hidden_layers[depth+1]))

        self.fcs = nn.ModuleList(self.fcs)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        for depth in range(self.layer_depth-2):
            x = self.activation(self.fcs[depth](x))
        x = self.fcs[-1](x)
        return x