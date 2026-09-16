import torch
import torch.nn as nn
from .config import KARTConfig

class DomainBounding(nn.Module):
    def __init__(self, config: KARTConfig):
        super().__init__()
        self.use_bound = config.use_domain_bound
        
        if self.use_bound:
            self.norm = nn.LayerNorm(config.in_features, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape:  (Batch, in_features)
        Output shape:   (Batch, in_features) - Value in (0, 1)
        """
        if not self.use_bound:
            return x
        x_norm = self.norm(x)
        x_tilde = (torch.tanh(x_norm) + 1.0) / 2.0
        return x_tilde

class BranchShifter(nn.Module):
    shift_array: torch.Tensor 

    def __init__(self, config: KARTConfig):
        super().__init__()
        self.Q = config.Q
        self.shift_type = config.shift_type

        if self.shift_type == 'deterministic':
            eta_val = 1.0 / self.Q
            shift_tensor = torch.arange(self.Q, dtype=torch.float32) * eta_val
            self.register_buffer('shift_array', shift_tensor)
            
        elif self.shift_type == 'learned':
            self.learned_eta = nn.Parameter(torch.tensor(1.0 / self.Q))
            self.register_buffer('shift_array', torch.empty(self.Q)) 
            
        else:
            raise ValueError(f"Shift {self.shift_type} is not supported!")

    def forward(self, x_tilde: torch.Tensor) -> torch.Tensor:
        """
        Input x_tilde shape: (Batch, in_features)
        Output z shape:      (Batch, in_features, Q)
        """
        x_expanded = x_tilde.unsqueeze(-1)
        
        if self.shift_type == 'learned':
            q_indices = torch.arange(self.Q, dtype=x_tilde.dtype, device=x_tilde.device)
            current_shifts = q_indices * self.learned_eta
        else:
            current_shifts = self.shift_array
            
        z = x_expanded + current_shifts
        return z