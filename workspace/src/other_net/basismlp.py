import torch.nn as nn
from KART.basis import FourierBasis

class BasisMLP(nn.Module):
    def __init__(self, d_model, K=8):
        super().__init__()
        self.basis = FourierBasis(K)
        self.fc1 = nn.Linear(d_model * K, d_model * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        b_val = self.basis(x)
        b_flat = b_val.flatten(start_dim=-2)
        return self.fc2(self.act(self.fc1(b_flat)))