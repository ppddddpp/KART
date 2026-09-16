import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from benchmark import train_model_with_time, set_seed, get_dataloaders

def case8_d_m_grid(dataset_seed=42, seeds_list=[1, 2, 3, 4, 5]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(BASE_DIR / 'results', exist_ok=True)

    INPUT_DIM = 3 * 32 * 32
    OUTPUT_DIM = 100
    NUM_LAYERS = 3
    EPOCHS = 100
    BATCH_SIZE = 256
    D_MODEL = 256
    
    d_m_configs = [(8, 8), (16, 8), (32, 8), (8, 16), (16, 16), (32, 16), (8, 32), (16, 32), (32, 32)]
    summary_results = []
    for (d_val, m_val) in d_m_configs:
        print(f"\n=== Training KART (Fourier) | D={d_val}, M={m_val} (Fixed R=8) ===")
        seed_accs, seed_epochs = [], []
        params = 0
        theo_flops = 0

        for idx, seed in enumerate(seeds_list):
            set_seed(seed)
            train_loader, val_loader, test_loader = get_dataloaders(
                dataset_name="CIFAR100", base_dir=str(BASE_DIR / 'data'), batch_size=BATCH_SIZE, seed=dataset_seed
            )

            from KART import KARTNet, KARTConfig
            config = KARTConfig(D_MODEL, D_MODEL, Q=4, D=d_val, R=8, K=8, M=m_val, basis_type='fourier')
            kart_model = KARTNet(INPUT_DIM, OUTPUT_DIM, D_MODEL, NUM_LAYERS, config).to(device)
            
            if idx == 0:
                params = sum(p.numel() for p in kart_model.parameters() if p.requires_grad)
                from benchmark import get_theoretical_flops_kart
                theo_flops = get_theoretical_flops_kart(config, D_MODEL, NUM_LAYERS, BATCH_SIZE, INPUT_DIM, OUTPUT_DIM)
                
            _, _, _, best_epoch = train_model_with_time("KART", kart_model, train_loader, val_loader, EPOCHS, device)
            
            kart_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
                    correct += (torch.max(kart_model(inputs).data, 1)[1] == labels).sum().item()
                    total += labels.size(0)
            acc = 100 * correct / total
            
            seed_accs.append(acc)
            seed_epochs.append(best_epoch)
            print(f"    Seed {seed}: Acc = {acc:.2f}%")
            
        summary_results.append({
            "D": d_val, "M": m_val, "Params": params,
            "Best Epoch": f"{np.mean(seed_epochs):.1f} ± {np.std(seed_epochs, ddof=1) if len(seed_epochs) > 1 else np.std(seed_epochs):.1f}",
            "Accuracy": f"{np.mean(seed_accs):.2f}% ± {np.std(seed_accs, ddof=1) if len(seed_accs) > 1 else np.std(seed_accs):.2f}%",
            "Theo_Flops": theo_flops,
            "Mean_Acc_Raw": np.mean(seed_accs)
        })
        
    df = pd.DataFrame(summary_results)
    df.to_csv(BASE_DIR / 'results' / 'cifar100_d_m_capacity.csv', index=False)
    
    print("\n" + "="*80)
    print(f"{'D':<5} | {'M':<5} | {'Params':<10} | {'Best Epoch':<15} | {'Accuracy'} | {'Theo_Flops'}")
    print("-" * 80)
    for r in summary_results:
        print(f"{r['D']:<5} | {r['M']:<5} | {r['Params']:<10,} | {r['Best Epoch']:<15} | {r['Accuracy']} | {r['Theo_Flops']:,}")
    print("="*80)