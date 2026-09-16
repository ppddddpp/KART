import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from benchmark import set_seed, get_dataloaders, train_vit_model
from KART import KARTConfig
from other_net.vit import MiniViT

def case11_vit_final_tuning(dataset_seed=42, seeds_list=[1, 2, 3]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(BASE_DIR / 'results', exist_ok=True)
    
    BATCH_SIZE = 256; EPOCHS = 50; D_MODEL = 128

    architectures = [
        {"name": "1. ViT-KART (Q=1, R=64) + LS 0.25", "ffn": "kart", "config": KARTConfig(D_MODEL, D_MODEL, Q=1, D=16, M=8, R=64, K=8, basis_type='fourier', use_domain_bound=True), "ls_val": 0.25},
        {"name": "2. ViT-KART (Q=1, R=64) + LS 0.1", "ffn": "kart", "config": KARTConfig(D_MODEL, D_MODEL, Q=1, D=16, M=8, R=64, K=8, basis_type='fourier', use_domain_bound=True), "ls_val": 0.1},
        {"name": "3. ViT-KART (Q=1, R=128) + LS 0.1", "ffn": "kart", "config": KARTConfig(D_MODEL, D_MODEL, Q=1, D=16, M=8, R=128, K=8, basis_type='fourier', use_domain_bound=True), "ls_val": 0.1}
    ]
    
    summary_results = []
    
    for arch in architectures:
        print(f"\n=== Training: {arch['name']} ===")
        seed_accs = []
        seed_val_accs = []
        seed_times = []
        seed_epochs = []
        all_gamma_means = []
        all_gamma_stds = []
        
        for idx, seed in enumerate(seeds_list):
            set_seed(seed)
            train_loader, val_loader, test_loader = get_dataloaders("CIFAR100", str(BASE_DIR / 'data'), BATCH_SIZE, seed=dataset_seed)
            model = MiniViT(ffn_type=arch['ffn'], d_model=D_MODEL, kart_config=arch['config'], ls_init_value=arch['ls_val']).to(device)
            train_time, best_epoch, best_val_acc = train_vit_model(model, train_loader, val_loader, EPOCHS, device)
            model.eval(); correct = 0; total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    correct += (torch.max(model(inputs).data, 1)[1] == labels).sum().item()
                    total += labels.size(0)
            
            acc = 100 * correct / total
            seed_accs.append(acc)
            seed_val_accs.append(best_val_acc)
            seed_times.append(train_time)
            seed_epochs.append(best_epoch)
            
            print(f"    Seed {seed}: Val Acc = {best_val_acc:.2f}% | Test Acc = {acc:.2f}% | Best Epoch = {best_epoch}")
            
            if arch['ls_val'] is not None:
                gamma_means = []
                gamma_stds = []
                for block in model.blocks:
                    if hasattr(block, 'gamma') and block.gamma is not None:
                        gamma_detach = block.gamma.detach()
                        gamma_means.append(round(gamma_detach.mean().item(), 4))
                        gamma_stds.append(round(gamma_detach.std().item(), 4))
                        
                all_gamma_means.append(gamma_means)
                all_gamma_stds.append(gamma_stds)
                print(f"    -> Final Gamma Means per block: {gamma_means}")
                print(f"    -> Final Gamma Stds per block : {gamma_stds}")
            
        summary_results.append({
            "Architecture": arch['name'],
            "Validation Accuracy": f"{np.mean(seed_val_accs):.2f}% ± {np.std(seed_val_accs, ddof=1):.2f}%",
            "Test Accuracy": f"{np.mean(seed_accs):.2f}% ± {np.std(seed_accs, ddof=1):.2f}%",
            "Best Epoch": f"{np.mean(seed_epochs):.1f} ± {np.std(seed_epochs, ddof=1):.1f}",
            "Avg Learned Gamma": str(np.mean(all_gamma_means, axis=0).round(4).tolist()) if all_gamma_means else "N/A",
            "Gamma Channel Std": str(np.mean(all_gamma_stds, axis=0).round(4).tolist()) if all_gamma_stds else "N/A"
        })
        
    pd.DataFrame(summary_results).to_csv(BASE_DIR / 'results' / 'vit_final_tuning.csv', index=False)
    
    print("\n" + "="*145)
    print(f"{'Architecture':<35} | {'Validation Acc':<18} | {'Test Acc':<16} | {'Best Epoch':<12} | {'Avg Gamma':<25} | {'Gamma Std'}")
    print("-" * 145)
    for r in summary_results: 
        print(f"{r['Architecture']:<35} | {r['Validation Accuracy']:<18} | {r['Test Accuracy']:<16} | {r['Best Epoch']:<12} | {r['Avg Learned Gamma']:<25} | {r['Gamma Channel Std']}")
    print("="*145)