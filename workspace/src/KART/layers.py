import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KARTConfig
from .modules import DomainBounding, BranchShifter
from .basis import BasisExpansion

class KARTLayer(nn.Module):
    def __init__(self, config: KARTConfig):
        super().__init__()
        self.config = config
        
        self.domain_bound = DomainBounding(config)
        self.shifter = BranchShifter(config)
        self.basis = BasisExpansion(config)
        self.act_fn = self._get_activation(config.act_type)
        
        self.c = nn.Parameter(torch.empty(config.in_features, config.K, config.D))
        self.v = nn.Parameter(torch.empty(config.Q, config.M, config.D))
        self.b = nn.Parameter(torch.empty(config.Q, config.M))
        self.r = nn.Parameter(torch.empty(config.Q, config.M, config.R))
        self.W_o = nn.Parameter(torch.empty(config.out_features, config.R))
        
        self.reset_parameters()

    def _get_activation(self, act_type: str):
        if act_type == 'silu':
            return F.silu
        elif act_type == 'tanh':
            return torch.tanh
        elif act_type == 'relu':
            return F.relu
        elif act_type == 'gelu':
            return F.gelu
        else:
            raise ValueError(f"Activation function {act_type} is not supported.")

    def reset_parameters(self):
        std = self.config.init_std
        n, K, Q, M = self.config.in_features, self.config.K, self.config.Q, self.config.M
        D, R = self.config.D, self.config.R
        
        with torch.no_grad():
            dummy_x = torch.randn(1024, n)
            x_tilde = self.domain_bound(dummy_x)
            z = self.shifter(x_tilde)
            B_dummy = self.basis(z)
            # Second raw moment mu_B^(2) = E[B^2]
            mu_B_sq = (B_dummy ** 2).mean().item()

        mu_g_sq = 0.356
        
        # Var(c) = 1 / (n * K * mu_B^2)
        std_c = std * math.sqrt(1.0 / (n * K * mu_B_sq))
        nn.init.normal_(self.c, mean=0.0, std=std_c)
        
        # Var(v) = 1 / D
        std_v = std * math.sqrt(1.0 / D)
        nn.init.normal_(self.v, mean=0.0, std=std_v)
        nn.init.zeros_(self.b) # Bias always initialized to 0
        
        # Var(r) = 1 / (M * mu_g^2)
        std_r = std * math.sqrt(1.0 / (M * mu_g_sq))
        nn.init.normal_(self.r, mean=0.0, std=std_r)
        
        # Var(W_o) = 1 / R
        std_Wo = std * math.sqrt(1.0 / R)
        nn.init.normal_(self.W_o, mean=0.0, std=std_Wo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x has shape (Batch, n)
        Output: f has shape (Batch, d_out)
        """
        x_tilde = self.domain_bound(x)
        z = self.shifter(x_tilde)
        B_val = self.basis(z)

        B_dim, n, Q, K = B_val.shape
        D, M, R = self.config.D, self.config.M, self.config.R
        B_val_reshaped = B_val.transpose(1, 2).reshape(B_dim, Q, n * K)
        c_reshaped = self.c.view(n * K, D)
        u = torch.matmul(B_val_reshaped, c_reshaped)

        u_t = u.transpose(0, 1) 
        v_t = self.v.transpose(1, 2)
        s = torch.bmm(u_t, v_t).transpose(0, 1) + self.b 

        a = self.act_fn(s)
        a_t = a.transpose(0, 1)
        h = torch.bmm(a_t, self.r).sum(dim=0) / math.sqrt(self.config.Q)
        
        out = torch.matmul(h, self.W_o.t())
        return out