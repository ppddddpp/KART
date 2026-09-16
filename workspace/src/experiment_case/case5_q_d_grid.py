import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from benchmark import set_seed
from KART import KARTRegressor, KARTConfig

def generate_interaction_data(num_samples=10000, seed=42):
    set_seed(seed)
    X = torch.rand(num_samples, 10) 
    y = torch.sin(math.pi * X[:, 0] * X[:, 1]) + (X[:, 2] - X[:, 3])**2
    return TensorDataset(X, y.unsqueeze(1))

def case5_q_d_grid(dataset_seed=42, seeds_list=[1, 2, 3, 4, 5]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(BASE_DIR / 'results', exist_ok=True)

    dataset = generate_interaction_data(10000, seed=dataset_seed)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [8000, 2000],
        generator=torch.Generator().manual_seed(dataset_seed)
    )
    
    q_list = [1, 2, 4, 8]
    d_list = [1, 2, 4, 8, 16]
    basis_types = ['fourier', 'b_spline']
    
    all_results = []

    for basis in basis_types:
        print(f"\n=== Running Experiment for Basis: {basis.upper()} ===")
        grid_matrix = np.zeros((len(q_list), len(d_list)))
        
        for i, q_val in enumerate(q_list):
            for j, d_val in enumerate(d_list):
                print(f"  -> Q={q_val}, D={d_val} ", end="")
                seed_rmses = []
                
                for seed in seeds_list:
                    set_seed(seed)
                    
                    g_data = torch.Generator()
                    g_data.manual_seed(seed)
                    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=g_data)
                    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
                    
                    config = KARTConfig(
                        in_features=10, out_features=10, Q=q_val, D=d_val, R=8, K=8, M=8, 
                        basis_type=basis, act_type='silu', use_domain_bound=False
                    )
                    model = KARTRegressor(input_dim=10, num_layers=2, base_config=config).to(device)
                    
                    optimizer = optim.AdamW(model.parameters(), lr=0.01)
                    criterion = nn.MSELoss()
                    
                    model.train()
                    for epoch in range(15):
                        for inputs, targets in train_loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            optimizer.zero_grad()
                            loss = criterion(model(inputs), targets)
                            loss.backward()
                            optimizer.step()
                            
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            val_loss += criterion(model(inputs), targets).item() * inputs.size(0)
                    rmse = math.sqrt(val_loss / len(val_ds))
                    seed_rmses.append(rmse)
                
                mean_rmse = np.mean(seed_rmses)
                std_rmse = np.std(seed_rmses, ddof=1) if len(seed_rmses) > 1 else np.std(seed_rmses)
                print(f"| RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
                
                all_results.append({"Basis": basis, "Q": q_val, "D": d_val, "RMSE (Mean)": mean_rmse, "RMSE (Std)": std_rmse})
                grid_matrix[i, j] = mean_rmse

        plt.figure(figsize=(8, 6))
        sns.heatmap(grid_matrix, annot=True, fmt=".4f", cmap="viridis_r", 
                    xticklabels=[str(d) for d in d_list],
                    yticklabels=[str(q) for q in q_list])
        plt.title(f"Synthetic Interaction Capacity ({basis.upper()})", fontweight='bold')
        plt.xlabel("D (Inner Latent Dimension)")
        plt.ylabel("Q (Shifts)")
        plt.tight_layout()
        plt.savefig(BASE_DIR / 'results' / f'xai_q_d_heatmap_{basis}.png', dpi=300)
        plt.close()

    pd.DataFrame(all_results).to_csv(BASE_DIR / 'results' / 'xai_q_d_tradeoff_full.csv', index=False)
    print("\n Results saved to xai_q_d_tradeoff_full.csv.")