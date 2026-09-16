import math
import torch
import torch.nn as nn
from copy import deepcopy

from .config import KARTConfig
from .layers import KARTLayer

class KARTNet(nn.Module):
    def __init__(
            self, 
            input_dim: int,
            output_dim: int,
            d_model: int,
            num_layers: int,
            base_config: KARTConfig
        ):
        super().__init__()
        
        self.num_layers = num_layers
        self.stem = nn.Linear(input_dim, d_model)
        
        # Lock in the inner config for KART layers with in/out features = d_model
        inner_config = deepcopy(base_config)
        inner_config.in_features = d_model
        inner_config.out_features = d_model
        
        self.layers = nn.ModuleList([
            KARTLayer(inner_config) for _ in range(num_layers)
        ])
        self.beta_L = 1.0 / math.sqrt(num_layers) if num_layers > 0 else 1.0

        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (Batch, input_dim)
        Output shape:  (Batch, output_dim)
        """
        x = self.stem(x)
        for layer in self.layers:
            x = x + self.beta_L * layer(x)
            
        x = self.head_norm(x)
        out = self.head(x)
        
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class KARTRegressor(nn.Module):
    def __init__(
            self, 
            input_dim: int,
            num_layers: int,
            base_config: KARTConfig
        ):
        super().__init__()
        self.num_layers = num_layers
        self.stem = nn.Identity()
        
        inner_config = deepcopy(base_config)
        inner_config.in_features = input_dim
        inner_config.out_features = input_dim
        
        self.layers = nn.ModuleList([
            KARTLayer(inner_config) for _ in range(num_layers)
        ])
        self.beta_L = 1.0 / math.sqrt(num_layers) if num_layers > 0 else 1.0

        self.head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.layers:
            x = x + self.beta_L * layer(x)
        return self.head(x)