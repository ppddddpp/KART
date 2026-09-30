import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class LowRankMLPLayer(nn.Module):
    """
    Low-Rank MLP Residual Branch.
    Implements:
        x_tilde = DomainBounding(x) if use_domain_bound else x
        z = x_tilde @ V               (in_features -> D)
        a = act(z @ U^T + b)          (D -> H)
        h = a @ R_mat^T               (H -> R)
        F_LR(x) = h @ W_o^T           (R -> out_features)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        D: int,
        R: int,
        H: int,
        act_type: str = 'silu',
        use_domain_bound: bool = True,
        init_std: float = 1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.D = D
        self.R = R
        self.H = H
        self.act_type = act_type
        self.use_domain_bound = use_domain_bound
        self.init_std = init_std

        if self.use_domain_bound:
            self.norm = nn.LayerNorm(in_features, elementwise_affine=False)

        self.V = nn.Parameter(torch.empty(in_features, D))
        self.U = nn.Parameter(torch.empty(H, D))
        self.b = nn.Parameter(torch.empty(H))
        self.R_mat = nn.Parameter(torch.empty(R, H))
        self.W_o = nn.Parameter(torch.empty(out_features, R))

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

    def domain_bound(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_domain_bound:
            return x
        x_norm = self.norm(x)
        return (torch.tanh(x_norm) + 1.0) / 2.0

    def reset_parameters(self):
        std = self.init_std
        with torch.no_grad():
            dummy_x = torch.randn(4096, self.in_features)
            x_tilde = self.domain_bound(dummy_x)
            mu_x2 = x_tilde.square().mean().item()

        # Empirical second moment of SiLU under standard Gaussian is approx 0.356
        mu_g_sq = 0.356

        std_V = std * math.sqrt(1.0 / (self.in_features * mu_x2))
        std_U = std * math.sqrt(1.0 / self.D)
        std_R = std * math.sqrt(1.0 / (self.H * mu_g_sq))
        std_Wo = std * math.sqrt(1.0 / self.R)

        nn.init.normal_(self.V, mean=0.0, std=std_V)
        nn.init.normal_(self.U, mean=0.0, std=std_U)
        nn.init.zeros_(self.b)
        nn.init.normal_(self.R_mat, mean=0.0, std=std_R)
        nn.init.normal_(self.W_o, mean=0.0, std=std_Wo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tilde = self.domain_bound(x)
        z = torch.matmul(x_tilde, self.V)                     # (Batch, D)
        act_fn = self._get_activation(self.act_type)
        a = act_fn(F.linear(z, self.U, self.b))               # (Batch, H)
        h = F.linear(a, self.R_mat)                           # (Batch, R)
        out = F.linear(h, self.W_o)                           # (Batch, out_features)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LowRankMLPNet(nn.Module):
    """
    Residual Low-Rank MLP Network for Classification.
    Matches KARTNet / ResMLPNet macro-backbone:
        x_0 = stem(x)
        x_{l+1} = x_l + (1 / sqrt(L)) * F_LR(x_l)
        out = head(LayerNorm(x_L))
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        num_layers: int,
        D: int,
        R: int,
        H: int,
        act_type: str = 'silu',
        use_domain_bound: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.D = D
        self.R = R
        self.H = H
        self.use_domain_bound = use_domain_bound

        self.stem = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            LowRankMLPLayer(
                in_features=d_model,
                out_features=d_model,
                D=D,
                R=R,
                H=H,
                act_type=act_type,
                use_domain_bound=use_domain_bound
            ) for _ in range(num_layers)
        ])
        self.beta_L = 1.0 / math.sqrt(num_layers) if num_layers > 0 else 1.0
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.layers:
            x = x + self.beta_L * layer(x)
        x = self.head_norm(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_branch_parameters(self) -> int:
        return sum(p.numel() for layer in self.layers for p in layer.parameters() if p.requires_grad)


class LowRankMLPRegressor(nn.Module):
    """
    Residual Low-Rank MLP Network for Regression.
    Matches KARTRegressor macro-backbone:
        stem = Identity()
        x_{l+1} = x_l + (1 / sqrt(L)) * F_LR(x_l)
        head = Linear(input_dim, 1)
    """
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        D: int,
        R: int,
        H: int,
        act_type: str = 'silu',
        use_domain_bound: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.D = D
        self.R = R
        self.H = H
        self.use_domain_bound = use_domain_bound

        self.stem = nn.Identity()
        self.layers = nn.ModuleList([
            LowRankMLPLayer(
                in_features=input_dim,
                out_features=input_dim,
                D=D,
                R=R,
                H=H,
                act_type=act_type,
                use_domain_bound=use_domain_bound
            ) for _ in range(num_layers)
        ])
        self.beta_L = 1.0 / math.sqrt(num_layers) if num_layers > 0 else 1.0
        self.head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.layers:
            x = x + self.beta_L * layer(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_branch_parameters(self) -> int:
        return sum(p.numel() for layer in self.layers for p in layer.parameters() if p.requires_grad)


def solve_param_matched_lr_config(
    n: int,
    d_out: int,
    target_branch_params: int,
    D_target: int,
    R_target: int,
    H_target: int,
    num_layers: int = 3,
    stem_head_params: int = 0,
    enforce_symmetric: bool = True
) -> Tuple[int, int, int, Dict[str, Any]]:
    """
    Deterministic bounded solver to find (D, R, H) of LR-MLP matching target branch & network params.
    Constraints:
        1 <= D <= min(n, H)
        1 <= R <= min(H, d_out)
        1 <= H <= H_max = max(4 * H_target, 2 * n)
        If enforce_symmetric and D_target == R_target: D == R
    Search:
        With S = D + R:
        P_branch = n * S + H * (S + 1)
        H* = (target_branch_params - n * S) / (S + 1)
        Only check H in {floor(H*), ceil(H*)} satisfying constraints.
    Tie-breaking:
        1. Smallest whole-network parameter discrepancy
        2. Smallest H
        3. Smallest D
        4. Smallest R
    """
    H_max = max(4 * H_target, 2 * n)
    target_total = stem_head_params + num_layers * target_branch_params

    best_candidate = None
    best_key = None
    best_info = None

    match_symmetric = enforce_symmetric and (D_target == R_target)

    for D in range(1, n + 1):
        r_range = [D] if match_symmetric else range(1, d_out + 1)
        for R in r_range:
            S = D + R
            numerator = target_branch_params - n * S
            denominator = S + 1
            if numerator <= 0:
                continue

            H_star = numerator / denominator
            candidates_H = {math.floor(H_star), math.ceil(H_star)}

            for H in candidates_H:
                if H < 1 or H > H_max:
                    continue
                if D > H or R > H:
                    continue

                branch_params = n * S + H * (S + 1)
                total_params = stem_head_params + num_layers * branch_params

                disc_total = abs(total_params - target_total) / target_total if target_total > 0 else 0.0
                disc_branch = abs(branch_params - target_branch_params) / target_branch_params

                # Key: (rounded discrepancy to avoid 1e-16 float noise, H, D, R)
                key = (round(disc_total, 5), H, D, R)

                if best_key is None or key < best_key:
                    best_key = key
                    best_candidate = (D, R, H)
                    best_info = {
                        "D": D,
                        "R": R,
                        "H": H,
                        "branch_params": branch_params,
                        "total_params": total_params,
                        "target_branch_params": target_branch_params,
                        "target_total_params": target_total,
                        "branch_discrepancy": disc_branch,
                        "total_discrepancy": disc_total
                    }

    if best_candidate is None or best_info is None:
        raise RuntimeError(f"Solver failed to find a valid LR configuration for n={n}, target={target_branch_params}")

    return best_candidate[0], best_candidate[1], best_candidate[2], best_info

