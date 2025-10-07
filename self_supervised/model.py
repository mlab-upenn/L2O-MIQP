import torch
import numpy as np
import torch
import torch.nn as nn

class STE_Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FFNet(nn.Module):
    """Feed-forward neural network with optional activation and STE rounding at the output."""
    
    def __init__(self, shape, activation=None):
        """
        Arguments:
            shape: list of ints describing network shape, including input & output size.
            activation: a nn function specifying the network activation (e.g., nn.ReLU()).
        """
        super(FFNet, self).__init__()
        self.shape = shape
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(shape) - 1):
            self.layers.append(nn.Linear(shape[i], shape[i + 1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.activation is not None:
                x = self.activation(x)
        
        x = self.layers[-1](x)
        x = torch.sigmoid(x) 
        return STE_Round.apply(x)