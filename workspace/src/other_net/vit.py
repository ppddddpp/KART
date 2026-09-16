import torch
import torch.nn as nn
from .efficient_kan import KANLinear
from KART.layers import KARTLayer
from .basismlp import BasisMLP

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, d_model=128, img_size=32):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class ViTBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_type='mlp', kart_config=None, ls_init_value=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ls_init_value = ls_init_value
        if ls_init_value is not None:
            self.gamma = nn.Parameter(ls_init_value * torch.ones(d_model))
        else:
            self.gamma = None

        if ffn_type == 'mlp':
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            )
        elif ffn_type == 'kart':
            assert kart_config is not None, "kart_config should not be None when ffn_type='kart'!"
            self.ffn = KARTLayer(kart_config)
        elif ffn_type == 'kan':
            self.ffn = KANLinear(in_features=d_model, out_features=d_model, grid_size=5, spline_order=3)
        elif ffn_type == 'basis_mlp':
            self.ffn = BasisMLP(d_model, K=8)
        else:
            raise ValueError("Error: only support 'mlp', 'kart', 'kan', or 'basis_mlp' for ffn_type")

    def forward(self, x):
        x_norm1 = self.norm1(x)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + attn_out
        
        B, N, D = x.shape
        x_norm2 = self.norm2(x)
        
        if isinstance(self.ffn, nn.Sequential) or isinstance(self.ffn, BasisMLP):
            ffn_out = self.ffn(x_norm2)
        else: 
            x_flat = x_norm2.reshape(B * N, D)
            ffn_out = self.ffn(x_flat).reshape(B, N, D)
            
        if self.gamma is not None:
            ffn_out = self.gamma * ffn_out
            
        x = x + ffn_out
            
        return x

class MiniViT(nn.Module):
    def __init__(self, ffn_type='mlp', d_model=128, n_heads=4, depth=4, num_classes=100, kart_config=None, ls_init_value=None):
        super().__init__()
        self.patch_embed = PatchEmbedding(d_model=d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, d_model))
        
        self.blocks = nn.ModuleList([
            ViTBlock(d_model, n_heads, ffn_type, kart_config, ls_init_value) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        return self.head(self.norm(x[:, 0]))