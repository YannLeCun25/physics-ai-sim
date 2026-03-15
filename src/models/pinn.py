import torch
import torch.nn as nn

class PhysicsInformedNN(nn.Module):
    """Deep Neural Network architecture for Physics-Informed Learning."""
    def __init__(self, layers: list):
        super().__init__()
        self.depth = len(layers) - 1
        self.activation = nn.Tanh()
        
        layer_list = []
        for i in range(self.depth):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        for i in range(self.depth - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x)
