import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from benchmark import set_seed
from KART import KARTRegressor, KARTConfig
from experiment_case.case4_synthetic_xai import calculate_feature_importance

def generate_equalized_data(num_samples=10000, seed=42, true_indices=[0, 1, 2]):
    set_seed(seed)
    X = torch.rand(num_samples, 20) 
    idx0, idx1, idx2 = true_indices
    
    # sin(2pix)
    mu1 = 0.0
    std1 = math.sqrt(0.5)

    # 0.5*x^2
    mu2 = 1.0 / 6.0
    var2 = 1.0 / 45.0
    std2 = math.sqrt(var2)

    # exp(-3x)
    mu3 = (1.0 - math.exp(-3.0)) / 3.0
    second3 = (1.0 - math.exp(-6.0)) / 6.0
    var3 = second3 - mu3 ** 2
    std3 = math.sqrt(var3)

    y1 = (torch.sin(2 * math.pi * X[:, idx0]) - mu1) / std1
    y2 = (0.5 * X[:, idx1] ** 2 - mu2) / std2
    y3 = (torch.exp(-3 * X[:, idx2]) - mu3) / std3

    y = y1 + y2 + y3
    return TensorDataset(X, y.unsqueeze(1))

def case12_equalized_xai(dataset_seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(BASE_DIR / 'results', exist_ok=True)
    
    NUM_TRIALS = 15
    recalls = []
    trial_results = []
    sine_hits = []
    quad_hits = []
    exp_hits = []

    for trial in range(NUM_TRIALS):
        current_seed = dataset_seed + trial
        set_seed(current_seed)
        true_indices = np.random.choice(20, 3, replace=False).tolist()
        
        dataset = generate_equalized_data(num_samples=10000, seed=current_seed, true_indices=true_indices)
        train_ds, val_ds = torch.utils.data.random_split(dataset, [8000, 2000], generator=torch.Generator().manual_seed(current_seed))
        
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        
        config = KARTConfig(in_features=20, out_features=20, Q=4, D=16, R=16, K=8, M=8, basis_type='b_spline', degree=3, act_type='silu', use_domain_bound=False)
        model = KARTRegressor(input_dim=20, num_layers=1, base_config=config).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        model.train()
        for epoch in range(20):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                
        energy_scores = calculate_feature_importance(model, val_loader, device)
        predicted_top3 = np.argsort(energy_scores)[::-1][:3].tolist()
        
        recall = len(set(predicted_top3).intersection(set(true_indices))) / 3.0
        recalls.append(recall)
        
        # top 3
        sine_hit = int(true_indices[0] in predicted_top3)
        quad_hit = int(true_indices[1] in predicted_top3)
        exp_hit = int(true_indices[2] in predicted_top3)
        
        sine_hits.append(sine_hit)
        quad_hits.append(quad_hit)
        exp_hits.append(exp_hit)

        trial_results.append({
            "Trial": trial + 1,
            "True_Indices": str(true_indices),
            "Pred_Top3": str(predicted_top3),
            "Top3_Recall": recall,
            "Sine_Hit": sine_hit,
            "Quad_Hit": quad_hit,
            "Exp_Hit": exp_hit
        })
        print(f"Trial {trial+1:02d} | True: {str(true_indices):<12} | Pred Top-3: {str(predicted_top3):<12} | Recall: {recall:.2f}")

    pd.DataFrame(trial_results).to_csv(
        BASE_DIR / 'results' / 'xai_equalized_trials.csv',
        index=False
    )

    print("-" * 60)
    print(f"Equalized Top-3 Recall: {np.mean(recalls):.4f} ± {np.std(recalls, ddof=1):.4f} (Baseline: 0.1500)")
    print(f"Sine recovery: {np.mean(sine_hits):.3f}")
    print(f"Quadratic recovery: {np.mean(quad_hits):.3f}")
    print(f"Exponential recovery: {np.mean(exp_hits):.3f}")
    print("-" * 60)