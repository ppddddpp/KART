import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from benchmark import set_seed
from KART import KARTRegressor, KARTConfig

def generate_randomized_synthetic_data(num_samples=10000, seed=42, true_indices=[0, 1, 2]):
    set_seed(seed)
    X = torch.rand(num_samples, 20) 
    idx0, idx1, idx2 = true_indices
    y = torch.sin(2 * math.pi * X[:, idx0]) + 0.5 * (X[:, idx1] ** 2) + torch.exp(-3 * X[:, idx2])
    return TensorDataset(X, y.unsqueeze(1))

def train_regression_model(model, train_loader, epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def calculate_feature_importance(model, val_loader, device):
    layer = model.layers[0] 
    input_dim = layer.config.in_features
    energy_scores = torch.zeros(input_dim).to(device)

    model.eval()
    
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        inputs.requires_grad_(True)
        
        z = layer.shifter(inputs)
        B_val = layer.basis(z) 
        u_p_all = torch.einsum('bnqk,nkd->bnqd', B_val, layer.c)

        u = u_p_all.sum(dim=1)
        u.retain_grad()
        u_t = u.transpose(0, 1) 
        v_t = layer.v.transpose(1, 2)
        s = torch.bmm(u_t, v_t).transpose(0, 1) + layer.b 
        a = layer.act_fn(s)
        a_t = a.transpose(0, 1)
        h = torch.bmm(a_t, layer.r).sum(dim=0) / math.sqrt(layer.config.Q)
        kart_out = torch.matmul(h, layer.W_o.t())
        
        x_out = inputs + model.beta_L * kart_out
        outputs = model.head(x_out)

        model.zero_grad()
        outputs.sum().backward()
        
        with torch.no_grad():
            assert u.grad is not None, "Gradient is None"
            grad_u = u.grad.unsqueeze(1)
            attribution_p = grad_u * u_p_all
            
            # I_p = E[ || (dy/du) concat u_p ||^2 ]
            score_p = (attribution_p ** 2).sum(dim=(2, 3)).sum(dim=0)
            energy_scores += score_p
                
    energy_scores = energy_scores / energy_scores.sum() * 100
    return energy_scores.cpu().numpy()

def evaluate_mse(model, val_loader, device):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item() * inputs.size(0)
    return total_loss / len(val_loader.dataset)

def evaluate_with_masking(model, val_loader, mask_indices, device):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            masked_inputs = inputs.clone()
            # Mask out the masked indices
            masked_inputs[:, mask_indices] = 0.5 
            
            outputs = model(masked_inputs)
            total_loss += criterion(outputs, targets).item() * inputs.size(0)
            
    return total_loss / len(val_loader.dataset)

def plot_functional_recovery(model, device, save_path, true_indices):
    model.eval()
    x_sweep = torch.linspace(0, 1, 100).to(device)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle("Partial Output Response versus True Generating Function", fontsize=16, fontweight='bold')
    
    titles = [f"x{true_indices[0]}: sin(2.pi.x)", f"x{true_indices[1]}: 0.5 * x^2", f"x{true_indices[2]}: exp(-3x)"]
    gt_funcs = [
        lambda x: torch.sin(2 * math.pi * x),
        lambda x: 0.5 * (x ** 2),
        lambda x: torch.exp(-3 * x)
    ]
    
    with torch.no_grad():
        for p in range(3):
            X_dummy = torch.full((100, 20), 0.5).to(device)
            X_dummy[:, true_indices[p]] = x_sweep
            
            y_pred = model(X_dummy).squeeze().cpu()
            y_true = gt_funcs[p](x_sweep.cpu())
            
            y_pred_centered = y_pred - y_pred.mean()
            y_true_centered = y_true - y_true.mean()
            
            axes[p].plot(x_sweep.cpu(), y_true_centered, 'k--', linewidth=2, label="Ground Truth")
            axes[p].plot(x_sweep.cpu(), y_pred_centered, 'b-', linewidth=2, label="KART Learned")
            axes[p].set_title(titles[p], fontweight='bold')
            axes[p].legend()
            axes[p].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=300)
    plt.close()

def case4_synthetic_xai(dataset_seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(BASE_DIR / 'results', exist_ok=True)
    
    NUM_TRIALS = 15

    all_trial_results = []
    all_feature_results = []
    recalls = []
    spearmans = []

    for trial in range(NUM_TRIALS):
        current_seed = dataset_seed + trial
        set_seed(current_seed)
        true_indices = np.random.choice(20, 3, replace=False).tolist()
        
        dataset = generate_randomized_synthetic_data(num_samples=10000, seed=current_seed, true_indices=true_indices)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [8000, 2000], 
            generator=torch.Generator().manual_seed(current_seed)
        )
        
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        
        config = KARTConfig(
            in_features=20, out_features=20, 
            Q=4, D=16, R=16, K=8, M=8, 
            basis_type='b_spline', degree=3, act_type='silu',
            use_domain_bound=False 
        )
        model = KARTRegressor(input_dim=20, num_layers=1, base_config=config).to(device)
        train_regression_model(model, train_loader, epochs=20, device=device)
        
        energy_scores = calculate_feature_importance(model, val_loader, device)
        predicted_top3 = np.argsort(energy_scores)[::-1][:3].tolist()

        baseline_mse = evaluate_mse(model, val_loader, device)
        delta_mses = np.zeros(20)
        for p in range(20):
            masked_mse = evaluate_with_masking(model, val_loader, [p], device)
            delta_mses[p] = masked_mse - baseline_mse
            
        # Top-3 Recall & Spearman Correlation
        recall = len(set(predicted_top3).intersection(set(true_indices))) / 3.0
        spearman_corr, _ = spearmanr(energy_scores, delta_mses)
        
        recalls.append(recall)
        spearmans.append(spearman_corr)
        
        print(f"Trial {trial+1:02d} | True: {str(true_indices):<12} | Pred Top-3: {str(predicted_top3):<12} | Recall: {recall:.2f} | Spearman Rho: {spearman_corr:.4f}")
        
        all_trial_results.append({
            "Trial": trial + 1,
            "True_Indices": str(true_indices),
            "Pred_Top3": str(predicted_top3),
            "Top3_Recall": recall,
            "Spearman_Rho": spearman_corr
        })

        true_set = set(true_indices)
        for p in range(20):
            all_feature_results.append({
                "Trial": trial + 1,
                "Feature": p,
                "Is_True_Feature": int(p in true_set),
                "Energy_Score": float(energy_scores[p]),
                "Delta_MSE": float(delta_mses[p])
            })
        
        if trial == 0:
            img_path = BASE_DIR / 'results' / 'xai_functional_recovery_trial.png'
            plot_functional_recovery(model, device, img_path, true_indices)

    RANDOM_RECALL_BASELINE = 3 / 20

    print("-" * 80)
    print(f"Random Top-3 Recall baseline: {RANDOM_RECALL_BASELINE:.4f}")
    print(f"Observed Top-3 Recall: {np.mean(recalls):.4f} ± {np.std(recalls, ddof=1):.4f}")
    print(f"Average Spearman Rho: {np.mean(spearmans):.4f} ± {np.std(spearmans, ddof=1):.4f}")
    print("-" * 80)

    df_trials = pd.DataFrame(all_trial_results)
    df_trials.to_csv(BASE_DIR / 'results' / 'xai_robustness_trials.csv', index=False)
    
    pd.DataFrame(all_feature_results).to_csv(BASE_DIR / 'results' / 'xai_robustness_per_feature.csv', index=False)
    
    print(f"\n -> Saved robust XAI metrics to '{BASE_DIR / 'results' / 'xai_robustness_trials.csv'}'")
    print(f" -> Saved per-feature results to '{BASE_DIR / 'results' / 'xai_robustness_per_feature.csv'}'")