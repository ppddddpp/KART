import math
import torch.nn as nn

class ResMLPLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.SiLU() 
        self.fc2 = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class ResMLPNet(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_layers):
        super().__init__()
        self.stem = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([ResMLPLayer(d_model) for _ in range(num_layers)])
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)
        self.beta_L = 1.0 / math.sqrt(num_layers) if num_layers > 0 else 1.0
        
    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = x + self.beta_L * layer(x) 
        x = self.head_norm(x)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)