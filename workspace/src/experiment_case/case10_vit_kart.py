import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from benchmark import set_seed, get_dataloaders, run_hardware_profiling, train_vit_model
from KART import KARTConfig
from other_net.vit import MiniViT

def case10_vit_kart(dataset_seed=42, seeds_list=[1, 2, 3]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(BASE_DIR / 'results', exist_ok=True)
    
    BATCH_SIZE = 256
    EPOCHS = 50
    D_MODEL = 128

    architectures = [
        {"name": "0. ViT-MLP (Baseline)", "ffn": "mlp", "config": None, "ls_val": None},
        {"name": "1. ViT-EfficientKAN", "ffn": "kan", "config": None, "ls_val": None},
        {"name": "2. ViT-BasisMLP", "ffn": "basis_mlp", "config": None, "ls_val": None}, 
        
        {"name": "3. ViT-KART (Q=4, R=16)", "ffn": "kart", "config": KARTConfig(D_MODEL, D_MODEL, Q=4, D=16, M=8, R=16, K=8, basis_type='fourier', use_domain_bound=True), "ls_val": None},
        {"name": "4. ViT-KART (Q=1, R=16)", "ffn": "kart", "config": KARTConfig(D_MODEL, D_MODEL, Q=1, D=16, M=8, R=16, K=8, basis_type='fourier', use_domain_bound=True), "ls_val": None},
        
        {"name": "5. ViT-KART (Q=1, R=32)", "ffn": "kart", "config": KARTConfig(D_MODEL, D_MODEL, Q=1, D=16, M=8, R=32, K=8, basis_type='fourier', use_domain_bound=True), "ls_val": None},
        {"name": "6. ViT-KART (Q=1, R=64)", "ffn": "kart", "config": KARTConfig(D_MODEL, D_MODEL, Q=1, D=16, M=8, R=64, K=8, basis_type='fourier', use_domain_bound=True), "ls_val": None},
        {"name": "7. ViT-KART (Q=1, R=128)", "ffn": "kart", "config": KARTConfig(D_MODEL, D_MODEL, Q=1, D=16, M=8, R=128, K=8, basis_type='fourier', use_domain_bound=True), "ls_val": None},
        
        {"name": "8. ViT-KART (Q=1, R=128) + LS 0.5", "ffn": "kart", "config": KARTConfig(D_MODEL, D_MODEL, Q=1, D=16, M=8, R=128, K=8, basis_type='fourier', use_domain_bound=True), "ls_val": 0.5},
        {"name": "9. ViT-KART (Q=1, R=128) + LS 0.25", "ffn": "kart", "config": KARTConfig(D_MODEL, D_MODEL, Q=1, D=16, M=8, R=128, K=8, basis_type='fourier', use_domain_bound=True), "ls_val": 0.25}
    ]
    
    summary_results = []
    
    for arch in architectures:
        print(f"=== Training: {arch['name']} ===")
        seed_accs, seed_times, seed_epochs = [], [], []
        params, peak_vram, throughput, prof_flops = 0, 0, 0, 0
        
        for idx, seed in enumerate(seeds_list):
            set_seed(seed)
            train_loader, val_loader, test_loader = get_dataloaders(
                dataset_name="CIFAR100", base_dir=str(BASE_DIR / 'data'), batch_size=BATCH_SIZE, seed=dataset_seed
            )
            
            model = MiniViT(ffn_type=arch['ffn'], d_model=D_MODEL, kart_config=arch['config'], ls_init_value=arch['ls_val']).to(device)
            
            if idx == 0:
                params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                peak_vram, throughput, prof_flops = run_hardware_profiling(model, device, input_shape=(BATCH_SIZE, 3, 32, 32))
            
            train_time, best_epoch, _ = train_vit_model(model, train_loader, val_loader, EPOCHS, device)
            
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    correct += (torch.max(model(inputs).data, 1)[1] == labels).sum().item()
                    total += labels.size(0)
            test_acc = 100 * correct / total
            
            seed_accs.append(test_acc)
            seed_times.append(train_time)
            seed_epochs.append(best_epoch)
            print(f"    Seed {seed}: Test Acc = {test_acc:.2f}%")
            
        summary_results.append({
            "Architecture": arch['name'],
            "Params": params,
            "Prof FLOPs": prof_flops,
            "Throughput": round(throughput, 0),
            "VRAM (MB)": round(peak_vram, 2),
            "Accuracy": f"{np.mean(seed_accs):.2f}% ± {np.std(seed_accs, ddof=1):.2f}%"
        })
        
    df = pd.DataFrame(summary_results)
    df.to_csv(BASE_DIR / 'results' / 'vit_final_diagnostic.csv', index=False)
    
    print("\n" + "="*115)
    print(f"{'Architecture':<40} | {'Params':<10} | {'Throughput':<12} | {'VRAM':<8} | {'Accuracy'}")
    print("-" * 115)
    for r in summary_results:
        print(f"{r['Architecture']:<40} | {r['Params']:<10,} | {r['Throughput']:<12.0f} | {r['VRAM (MB)']:<8} | {r['Accuracy']}")
    print("="*115)