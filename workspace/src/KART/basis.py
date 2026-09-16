import math
import torch
import torch.nn as nn
from .config import KARTConfig

class BaseBasis(nn.Module):
    def __init__(self, K: int):
        super().__init__()
        self.K = K

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Classes inheriting from BaseBasis must implement the forward method.")

class FourierBasis(BaseBasis):
    freqs: torch.Tensor

    def __init__(self, K: int):
        super().__init__(K)
        assert K % 2 == 0, "For FourierBasis, K must be an even number."
        num_freqs = K // 2
        freqs_array = torch.arange(1, num_freqs + 1, dtype=torch.float32) * math.pi
        self.register_buffer('freqs', freqs_array)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_expanded = z.unsqueeze(-1)
        args = z_expanded * self.freqs
        sin_part = torch.sin(args)
        cos_part = torch.cos(args)
        out = torch.cat([sin_part, cos_part], dim=-1)
        return out

class BSplineBasis(BaseBasis):
    knots: torch.Tensor

    def __init__(self, K: int, degree: int = 3):
        super().__init__(K)
        self.degree = degree
        assert K > degree, f"K ({K}) must be greater than degree ({degree}) for B-Spline."
        grid_size = K - degree
        
        grid_min, grid_max = 0.0, 2.0
        step = (grid_max - grid_min) / grid_size
        

        start_knot = grid_min - degree * step
        end_knot = grid_max + degree * step
        num_knots = grid_size + 2 * degree + 1
        
        knots = torch.linspace(start_knot, end_knot, num_knots, dtype=torch.float32)
        self.register_buffer('knots', knots)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(-1)
        bases = ((z >= self.knots[:-1]) & (z < self.knots[1:])).to(z.dtype)

        for d in range(1, self.degree + 1):
            left_den = self.knots[d:-1] - self.knots[:-d-1]
            right_den = self.knots[d+1:] - self.knots[1:-d]
            
            left_den = torch.where(left_den == 0, torch.ones_like(left_den), left_den)
            right_den = torch.where(right_den == 0, torch.ones_like(right_den), right_den)
            
            left_term = ((z - self.knots[:-d-1]) / left_den) * bases[..., :-1]
            right_term = ((self.knots[d+1:] - z) / right_den) * bases[..., 1:]
            
            bases = left_term + right_term

        return bases

class LinearBasis(BaseBasis):
    def __init__(self, K: int):
        super().__init__(K)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.unsqueeze(-1).expand(-1, -1, -1, self.K)

class BasisExpansion(nn.Module):
    def __init__(self, config: KARTConfig):
        super().__init__()
        self.basis_type = config.basis_type
        
        if self.basis_type == 'fourier':
            self.basis_fn = FourierBasis(config.K)
        elif self.basis_type == 'b_spline':
            degree = getattr(config, 'degree', 3) 
            self.basis_fn = BSplineBasis(config.K, degree=degree)
        elif self.basis_type == 'linear':
            self.basis_fn = LinearBasis(config.K)
        else:
            raise ValueError(f"Invalid basis type {self.basis_type}!")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.basis_fn(z)