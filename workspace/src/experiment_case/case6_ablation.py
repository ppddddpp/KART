import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from benchmark import train_model_with_time, set_seed, get_dataloaders
from KART import KARTNet, KARTConfig

def case6_ablation(dataset_seed=42, seeds_list=[1, 2, 3, 4, 5]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(BASE_DIR / 'results', exist_ok=True)

    variants = [
        {"name": "Full KART (Q=4, D=8)", "config": KARTConfig(64, 64, Q=4, D=8, R=8, K=8, M=8, basis_type='fourier', use_domain_bound=True)},
        {"name": "No Shifts (Q=1)", "config": KARTConfig(64, 64, Q=1, D=8, R=8, K=8, M=8, basis_type='fourier', use_domain_bound=True)},
        {"name": "Scalar Latent (D=1)", "config": KARTConfig(64, 64, Q=4, D=1, R=8, K=8, M=8, basis_type='fourier', use_domain_bound=True)},
        {"name": "Linear Basis", "config": KARTConfig(64, 64, Q=4, D=8, R=8, K=8, M=8, basis_type='linear', use_domain_bound=True)},
        {"name": "No Domain Bound", "config": KARTConfig(64, 64, Q=4, D=8, R=8, K=8, M=8, basis_type='fourier', use_domain_bound=False)}
    ]
    
    summary_results = []
    
    for var in variants:
        print(f"=== Training Variant: {var['name']} ===")
        seed_accs, seed_epochs = [], []
        params = 0
        
        for idx, seed in enumerate(seeds_list):
            set_seed(seed)
            train_loader, val_loader, test_loader = get_dataloaders(
                dataset_name="FashionMNIST", base_dir=str(BASE_DIR / 'data'), batch_size=256, seed=dataset_seed
            )
            
            model = KARTNet(784, 10, 64, 3, var['config']).to(device)
            if idx == 0:
                params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            _, _, _, best_epoch = train_model_with_time(var['name'], model, train_loader, val_loader, epochs=30, device=device)
            
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
                    correct += (torch.max(model(inputs).data, 1)[1] == labels).sum().item()
                    total += labels.size(0)
            acc = 100 * correct / total
            
            seed_accs.append(acc)
            seed_epochs.append(best_epoch)
            
        summary_results.append({
            "Ablation Variant": var['name'],
            "Params": params,
            "Best Epoch": f"{np.mean(seed_epochs):.1f} ± {np.std(seed_epochs, ddof=1) if len(seed_epochs) > 1 else np.std(seed_epochs):.1f}",
            "Accuracy": f"{np.mean(seed_accs):.2f}% ± {np.std(seed_accs, ddof=1) if len(seed_accs) > 1 else np.std(seed_accs):.2f}%"
        })
        
    df = pd.DataFrame(summary_results)
    df.to_csv(BASE_DIR / 'results' / 'ablation_ingredients_summary.csv', index=False)
    
    print("\n" + "="*85)
    print(f"{'Ablation Variant':<25} | {'Params':<10} | {'Best Epoch':<15} | {'Accuracy (Mean±Std)'}")
    print("-" * 85)
    for r in summary_results:
        print(f"{r['Ablation Variant']:<25} | {r['Params']:<10,} | {r['Best Epoch']:<15} | {r['Accuracy']}")
    print("="*85)