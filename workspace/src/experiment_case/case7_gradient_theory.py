import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from benchmark import set_seed
from KART import KARTLayer, KARTConfig

def estimate_rho_silu(num_samples=1_000_000, device=torch.device("cpu")):
    z = torch.randn(num_samples, device=device)
    sig = torch.sigmoid(z)
    g = z * sig
    g_prime = sig + z * sig * (1.0 - sig)

    mu_g = (g ** 2).mean()
    nu_g = (g_prime ** 2).mean()

    return (nu_g / mu_g).item()

def run_gradient_experiment(config_kwargs, seed, device, rho_g, rho_B):
    set_seed(seed)
    config = KARTConfig(**config_kwargs)
    layer = KARTLayer(config).to(device)
    
    x = torch.rand(256, config.in_features, device=device)
    x.requires_grad_(True)
    
    f_x = layer(x)
    delta_f = torch.randn_like(f_x)
    
    L = torch.sum(f_x * delta_f)
    L.backward()
    
    assert x.grad is not None, "Error: x.grad is None"
    
    # G_emp = Var(x.grad) / Var(delta_f)
    G_emp = x.grad.var().item() / delta_f.var().item()
    # G_pred = (d_out / n) * rho_g * rho_B
    G_pred = (config.out_features / config.in_features) * rho_g * rho_B
    
    return G_emp, G_pred

def case7_gradient_theory():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(BASE_DIR / 'results', exist_ok=True)

    set_seed(42)
    rho_g = estimate_rho_silu(device=device)
    m_freq = 4
    rho_B = (math.pi ** 2) * (m_freq + 1) * (2 * m_freq + 1) / 6.0 

    print(f"rho_g = {rho_g:.6f}")
    print(f"rho_B = {rho_B:.6f}")
    print(f"rho_g * rho_B = {rho_g * rho_B:.6f}\n")

    seeds = list(range(20))
    widths = [64, 128, 256, 512, 1024]
    
    all_results = []
    parity_emp_means, parity_emp_stds, parity_preds = [], [], []

    print("=== MODE A: d_out = 64 (Prediction: G ~ 1/n) ===")
    emp_vars_mode_A = []
    pred_mode_A = []
    for n in widths:
        emp_list = []
        pred = 0.0
        for seed in seeds:
            kwargs = dict(in_features=n, out_features=64, Q=4, D=8, R=8, K=8, M=8, basis_type='fourier', use_domain_bound=False)
            emp, pred = run_gradient_experiment(kwargs, seed, device, rho_g, rho_B)
            emp_list.append(emp)
        
        mean_emp, std_emp = np.mean(emp_list), np.std(emp_list, ddof=1)
        emp_vars_mode_A.append(mean_emp)
        pred_mode_A.append(pred)
        parity_emp_means.append(mean_emp); parity_emp_stds.append(std_emp); parity_preds.append(pred)
        all_results.append({
            "Experiment": "Mode A (d_out=64)", 
            "Variable": "n", "Value": n, 
            "G_Empirical": mean_emp, 
            "G_Empirical_Std": std_emp,
            "G_Predicted": pred,
            "Theory_Ratio": mean_emp / pred if pred != 0 else 0,
            "rho_g": rho_g,
            "rho_B": rho_B
            })
        print(f"  -> n={n:<4} | G_emp: {mean_emp:.4f} ± {std_emp:.4f} | G_pred: {pred:.4f}")

    print("\n=== MODE B: d_out = n (Prediction: G ~ constant) ===")
    emp_vars_mode_B = []
    pred_mode_B = []
    for n in widths:
        emp_list = []
        pred = 0.0
        for seed in seeds:
            kwargs = dict(in_features=n, out_features=n, Q=4, D=8, R=8, K=8, M=8, basis_type='fourier', use_domain_bound=False)
            emp, pred = run_gradient_experiment(kwargs, seed, device, rho_g, rho_B)
            emp_list.append(emp)
        
        mean_emp, std_emp = np.mean(emp_list), np.std(emp_list, ddof=1)
        emp_vars_mode_B.append(mean_emp)
        pred_mode_B.append(pred)
        parity_emp_means.append(mean_emp); parity_emp_stds.append(std_emp); parity_preds.append(pred)
        all_results.append({
            "Experiment": "Mode B (d_out=n)", 
            "Variable": "n", "Value": n, 
            "G_Empirical": mean_emp, 
            "G_Empirical_Std": std_emp,
            "G_Predicted": pred,
            "Theory_Ratio": mean_emp / pred if pred != 0 else 0,
            "rho_g": rho_g,
            "rho_B": rho_B
            })
        print(f"  -> n={n:<4} | G_emp: {mean_emp:.4f} ± {std_emp:.4f} | G_pred: {pred:.4f}")

    sweep_dict = {
        'Q': [1, 2, 4, 8],
        'D': [4, 8, 16, 32],
        'M': [4, 8, 16, 32],
        'R': [4, 8, 16, 32],
        'K': [2, 4, 8, 16]
    }
    
    for var_name, values in sweep_dict.items():
        print(f"\n -- Sweeping {var_name} --")
        for val in values:
            emp_list = []
            pred = 0.0
            current_rho_B = 0.0
            for seed in seeds:
                kwargs = dict(in_features=256, out_features=64, Q=4, D=8, R=8, K=8, M=8, basis_type='fourier', use_domain_bound=False)
                kwargs[var_name] = val
                
                current_rho_B = rho_B
                if var_name == 'K':
                    current_m_freq = val // 2
                    current_rho_B = (math.pi ** 2) * (current_m_freq + 1) * (2 * current_m_freq + 1) / 6.0
                    
                emp, pred = run_gradient_experiment(kwargs, seed, device, rho_g, current_rho_B)
                emp_list.append(emp)
                
            mean_emp, std_emp = np.mean(emp_list), np.std(emp_list, ddof=1)
            parity_emp_means.append(mean_emp); parity_emp_stds.append(std_emp); parity_preds.append(pred)
            all_results.append({
                "Experiment": "Independence Sweep", 
                "Variable": var_name, 
                "Value": val, 
                "G_Empirical": mean_emp, 
                "G_Empirical_Std": std_emp,
                "G_Predicted": pred,
                "Theory_Ratio": mean_emp / pred if pred != 0 else 0,
                "rho_g": rho_g,
                "rho_B": current_rho_B
                })
            print(f"  -> {var_name}={val:<2} | G_emp: {mean_emp:.4f} ± {std_emp:.4f} | G_pred: {pred:.4f}")

    pd.DataFrame(all_results).to_csv(BASE_DIR / 'results' / 'gradient_gain_full_theory.csv', index=False)

    # dual scaling plot: G_empirical vs n for Mode A and Mode B
    plt.figure(figsize=(8, 6))
    plt.plot(widths, emp_vars_mode_B, marker='s', color='darkred', linewidth=2, label="Mode B Emp: $d_{out} = n$")
    plt.plot(widths, emp_vars_mode_A, marker='o', color='darkblue', linewidth=2, label="Mode A Emp: $d_{out} = 64$")
    plt.plot(widths, pred_mode_B, 'r--', linewidth=2, label=r"Theory B: $\rho_g\rho_B$")
    plt.plot(widths, pred_mode_A, 'b--', linewidth=2, label=r"Theory A: $(d_{out}/n)\rho_g\rho_B$")

    plt.title("Gradient Gain Scaling Regimes", fontweight='bold')
    plt.xlabel("Network Width (n)")
    plt.ylabel(r"Gradient Gain $G_{\mathrm{empirical}}$")
    plt.xscale('log'); plt.yscale('log')
    plt.xticks(widths, labels=[str(w) for w in widths])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'results' / 'gradient_gain_dual_scaling.png', dpi=300)
    plt.close()

    # parity plot: G_empirical vs G_predicted
    plt.figure(figsize=(7, 7))
    plt.errorbar(
        parity_preds, 
        parity_emp_means, 
        yerr=parity_emp_stds, 
        fmt='o', 
        capsize=3, 
        alpha=0.7, 
        label="Configurations"
    )
    
    min_val = min(min(parity_preds), min(parity_emp_means)) * 0.7
    max_val = max(max(parity_preds), max(parity_emp_means)) * 1.3
    
    plt.plot(
        [min_val, max_val], 
        [min_val, max_val], 
        'k--', 
        label="Perfect match: y=x"
    )
    
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r"Predicted gain $G_{\mathrm{pred}}$")
    plt.ylabel(r"Empirical gain $G_{\mathrm{emp}}$")
    plt.title("Gradient Gain: Predicted vs Empirical", fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'results' / 'gradient_gain_parity.png', dpi=300)
    plt.close()