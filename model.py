import torch
import torch.nn as nn
from typing import List, Sequence

################################################################################
# Model
################################################################################

class GaussianActivation(nn.Module):
    """
    Custom Gaussian activation function.
    f(x) = exp(-x^2)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.exp(-torch.pow(x, 2))

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 4,
        out_dim: int = 3,
    ):
        super().__init__()
        layers: List[nn.Module] = [256, 256, 256, 256]
        self.activation = nn.GELU
        self.linear_layers = nn.ModuleList([nn.Linear(in_dim, layers[0])])
        for i in range(1, len(layers)):
            self.linear_layers.append(nn.Linear(layers[i-1], layers[i]))
            self.linear_layers.append(self.activation())
        self.output_layer = nn.Linear(layers[-1], out_dim)
    def forward(self, x):  # (B, 4) → (B, 3)
        for layer in self.linear_layers:
            x = layer(x)
        return nn.ReLU()(self.output_layer(x))
