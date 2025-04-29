import torch
import torch.nn as nn
from typing import List, Sequence

################################################################################
# Model
################################################################################

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 4,
        out_dim: int = 3,
        hidden_sizes: Sequence[int] = (256, 256, 256, 256),
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), activation()])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # (B, 4) → (B, 3)
        return self.net(x)