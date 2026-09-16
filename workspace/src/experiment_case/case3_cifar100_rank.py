import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from benchmark import (
    train_model_with_time, set_seed, get_model, get_dataloaders,
    run_hardware_profiling, get_theoretical_flops_kart, get_theoretical_flops_mlp
)

def case3_cifar100_rank(dataset_seed=42, seeds_list=[1, 2, 3, 4, 5]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(BASE_DIR / 'results', exist_ok=True)
    
    INPUT_DIM = 3 * 32 * 32
    OUTPUT_DIM = 100
    NUM_LAYERS = 3
    EPOCHS = 100
    BATCH_SIZE = 256
    SEEDS = seeds_list
    
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name="CIFAR100", base_dir=str(BASE_DIR / 'data'), batch_size=BATCH_SIZE, seed=dataset_seed
    )

    D_MODEL = 256 
    rank_configs = [ (8, 8), (16, 8), (8, 16), (16, 16), (32, 16), (32, 32) ]
    
    summary_results = []
    raw_results = []

    print("=== Model: ResMLP ===")
    seed_accs, seed_times, seed_epochs = [], [], []
    params, mlp_thru, mlp_flops = 0, 0, 0
    
    for idx, seed in enumerate(SEEDS):
        set_seed(seed)
        mlp_model = get_model("ResMLP", INPUT_DIM, OUTPUT_DIM, D_MODEL, NUM_LAYERS, device)
        
        if idx == 0:
            params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
            _, mlp_thru, _ = run_hardware_profiling(mlp_model, device, (BATCH_SIZE, INPUT_DIM))
            mlp_flops = get_theoretical_flops_mlp(D_MODEL, NUM_LAYERS, BATCH_SIZE, INPUT_DIM, OUTPUT_DIM)
        
        _, _, train_time, best_epoch = train_model_with_time("ResMLP", mlp_model, train_loader, val_loader, EPOCHS, device)
        
        mlp_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
                correct += (torch.max(mlp_model(inputs).data, 1)[1] == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        
        seed_accs.append(acc); seed_times.append(train_time); seed_epochs.append(best_epoch)
        raw_results.append({"Model": "ResMLP", "D, R": "N/A", "Seed": seed, "Accuracy": acc, "Train Time": train_time, "Best Epoch": best_epoch})
        
    summary_results.append({
        "Model": "ResMLP", "D, R": "N/A", "Params": params, "Theo FLOPs": mlp_flops, "Throughput": mlp_thru, 
        "Train Time (s)": round(np.mean(seed_times), 2), 
        "Best Epoch": f"{np.mean(seed_epochs):.1f} ± {np.std(seed_epochs, ddof=1) if len(seed_epochs) > 1 else np.std(seed_epochs):.1f}",
        "Accuracy": f"{np.mean(seed_accs):.2f} ± {np.std(seed_accs, ddof=1) if len(seed_accs) > 1 else np.std(seed_accs):.2f}"
    })

    for (d_val, r_val) in rank_configs:
        print(f"\n=== Training KART (Fourier) | D={d_val}, R={r_val} ===")
        seed_accs, seed_times, seed_epochs = [], [], []
        params, kart_thru, kart_flops = 0, 0, 0
        
        for idx, seed in enumerate(SEEDS):
            set_seed(seed)
            kart_model = get_model("KART (Fourier)", INPUT_DIM, OUTPUT_DIM, D_MODEL, NUM_LAYERS, device, D_rank=d_val, R_rank=r_val)
            
            if idx == 0:
                params = sum(p.numel() for p in kart_model.parameters() if p.requires_grad)
                _, kart_thru, _ = run_hardware_profiling(kart_model, device, (BATCH_SIZE, INPUT_DIM))
                kart_flops = get_theoretical_flops_kart(kart_model.layers[0].config, D_MODEL, NUM_LAYERS, BATCH_SIZE, INPUT_DIM, OUTPUT_DIM)
            
            _, _, train_time, best_epoch = train_model_with_time("KART", kart_model, train_loader, val_loader, EPOCHS, device)
            
            kart_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
                    correct += (torch.max(kart_model(inputs).data, 1)[1] == labels).sum().item()
                    total += labels.size(0)
            acc = 100 * correct / total
            
            seed_accs.append(acc); seed_times.append(train_time); seed_epochs.append(best_epoch)
            raw_results.append({"Model": "KART (Fourier)", "D, R": f"({d_val}, {r_val})", "Seed": seed, "Accuracy": acc, "Train Time": train_time, "Best Epoch": best_epoch})
            
        summary_results.append({
            "Model": "KART (Fourier)", "D, R": f"({d_val}, {r_val})", "Params": params, "Theo FLOPs": kart_flops, "Throughput": kart_thru, 
            "Train Time (s)": round(np.mean(seed_times), 2), 
            "Best Epoch": f"{np.mean(seed_epochs):.1f} ± {np.std(seed_epochs, ddof=1) if len(seed_epochs) > 1 else np.std(seed_epochs):.1f}",
            "Accuracy": f"{np.mean(seed_accs):.2f} ± {np.std(seed_accs, ddof=1) if len(seed_accs) > 1 else np.std(seed_accs):.2f}"
        })
        
    pd.DataFrame(raw_results).to_csv(BASE_DIR / 'results' / 'cifar100_rank_tradeoff_raw.csv', index=False)
    pd.DataFrame(summary_results).to_csv(BASE_DIR / 'results' / 'cifar100_rank_tradeoff_summary.csv', index=False)
    
    print("\n" + "="*135)
    print(f"{'Model':<16} | {'(D, R)':<10} | {'Params':<10} | {'FLOPs (Theo)':<15} | {'Throughput':<12} | {'Train Time':<12} | {'Best Epoch':<15} | {'Accuracy (Mean±Std)'}")
    print("-" * 135)
    for r in summary_results:
        print(f"{r['Model']:<16} | {r['D, R']:<10} | {r['Params']:<10,} | {r['Theo FLOPs']:<15,} | {r['Throughput']:<12.0f} | {r['Train Time (s)']:<10.2f} s | {r['Best Epoch']:<15} | {r['Accuracy']}")
    print("="*135)
