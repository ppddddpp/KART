import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from benchmark import (
    train_model_with_time, set_seed, get_model, 
    get_dataloaders, run_hardware_profiling, 
    get_theoretical_flops_kart, get_theoretical_flops_mlp
)
from benchmark.measure import extract_1d_learned_function

def case1_fmnist(dataset_seed=42, seeds_list=[1, 2, 3]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(BASE_DIR / 'results', exist_ok=True)
    
    INPUT_DIM = 784
    OUTPUT_DIM = 10
    D_MODEL = 64
    NUM_LAYERS = 3
    EPOCHS = 50
    BATCH_SIZE = 256
    SEEDS = seeds_list 
    
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name="FashionMNIST", 
        base_dir=str(BASE_DIR / 'data'),
        batch_size=BATCH_SIZE, 
        seed=dataset_seed 
    )
    
    model_names = [
        "1. ResMLP (Baseline)", "2. KART (Fourier)", "3. KART (B-Spline Deg 3)", 
        "4. KART (B-Spline Deg 1)", "5. KART (Linear Inner Basis)"
    ]
    
    raw_results = []
    summary_results = []
    summary_json = {}
    final_models_for_plot = {} 
    
    for name in model_names:
        print(f"=== Training: [{name}] with {len(SEEDS)} Seeds ===")
        seed_test_accs = []
        seed_train_times = []
        seed_curvatures = []
        seed_best_epochs = []
        params, peak_vram, throughput, theo_flops, prof_flops = 0, 0, 0, 0, 0
        
        for idx, seed in enumerate(SEEDS):
            set_seed(seed) 
            model = get_model(name, INPUT_DIM, OUTPUT_DIM, D_MODEL, NUM_LAYERS, device)
            
            if idx == 0:
                params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                peak_vram, throughput, prof_flops = run_hardware_profiling(model, device, (BATCH_SIZE, INPUT_DIM))
                theo_flops = get_theoretical_flops_mlp(D_MODEL, NUM_LAYERS, BATCH_SIZE, INPUT_DIM, OUTPUT_DIM) if "ResMLP" in name else get_theoretical_flops_kart(model.layers[0].config, D_MODEL, NUM_LAYERS, BATCH_SIZE, INPUT_DIM, OUTPUT_DIM)

            _, _, train_time, best_epoch = train_model_with_time(name, model, train_loader, val_loader, EPOCHS, device)
            
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100 * correct / total
            
            if "ResMLP" not in name:
                _, _, cur_curvature = extract_1d_learned_function(model, device)
            else:
                cur_curvature = None

            seed_test_accs.append(test_acc)
            seed_train_times.append(train_time)
            seed_curvatures.append(cur_curvature)
            seed_best_epochs.append(best_epoch)
            
            raw_results.append({
                "Model": name,
                "Seed": seed,
                "Accuracy": test_acc,
                "Train Time": train_time,
                "Best Epoch": best_epoch,
                "Curvature": cur_curvature
            })

            if idx == len(SEEDS) - 1:
                final_models_for_plot[name] = model 
        
        mean_acc = np.mean(seed_test_accs)
        std_acc = np.std(seed_test_accs, ddof=1) if len(seed_test_accs) > 1 else np.std(seed_test_accs)
        mean_time = np.mean(seed_train_times)
        mean_epoch = np.mean(seed_best_epochs)
        std_epoch = np.std(seed_best_epochs, ddof=1) if len(seed_best_epochs) > 1 else np.std(seed_best_epochs)
        epoch_str = f"{mean_epoch:.1f} ± {std_epoch:.1f}" if len(SEEDS) > 1 else f"{mean_epoch:.0f}"
        
        if "ResMLP" not in name:
            mean_curv = np.mean(seed_curvatures)
            std_curv = np.std(seed_curvatures, ddof=1) if len(seed_curvatures) > 1 else np.std(seed_curvatures)
            curv_str = f"{mean_curv:.4f} ± {std_curv:.4f}"
            plot_curv_val = mean_curv
        else:
            mean_curv = None
            std_curv = None
            curv_str = "N/A"
            plot_curv_val = None
        
        summary_results.append({
            "Model": name,
            "Params": params,
            "Theo FLOPs": theo_flops,
            "Prof FLOPs": prof_flops,
            "VRAM (MB)": round(peak_vram, 2),
            "Throughput": round(throughput, 0),
            "Train Time (s)": round(mean_time, 2),
            "Best Epoch": epoch_str,
            "Accuracy": f"{mean_acc:.2f} ± {std_acc:.2f}", 
            "Curvature": curv_str,
            "Plot_Curv": plot_curv_val 
        })
        
        summary_json[name] = {
            "Accuracy": {"mean": round(float(mean_acc), 4), "std": round(float(std_acc), 4), "raw": seed_test_accs},
            "Curvature": {"mean": round(float(mean_curv), 4) if mean_curv else None, "std": round(float(std_curv), 4) if std_curv else None, "raw": seed_curvatures},
            "Train Time (s)": {"mean": round(float(mean_time), 4), "raw": seed_train_times},
            "Best Epoch": {"mean": round(float(mean_epoch), 4), "std": round(float(std_epoch), 4), "raw": seed_best_epochs},
            "Hardware": {
                "Params": params,
                "Theo FLOPs": theo_flops,
                "Prof FLOPs": prof_flops,
                "VRAM (MB)": round(peak_vram, 2),
                "Throughput": round(throughput, 0)
            }
        }
        
        print(f"   => Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"   => Curvature: {curv_str}\n")
        
    pd.DataFrame(raw_results).to_csv(BASE_DIR / 'results' / 'fmnist_raw_seeds.csv', index=False)
    df_summary = pd.DataFrame(summary_results)
    df_summary.drop(columns=['Plot_Curv']).to_csv(BASE_DIR / 'results' / 'fmnist_summary.csv', index=False)
    
    with open(BASE_DIR / 'results' / 'fmnist_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)
    
    print("="*155)
    print(f"{'Model':<25} | {'Params':<9} | {'VRAM':<6} | {'Throughput':<10} | {'Train Time':<12} | {'Best Epoch':<15} | {'Accuracy (Mean±Std)':<20} | {'Curvature'}")
    print("-" * 155)
    for r in summary_results:
        print(f"{r['Model']:<25} | {r['Params']:<9,} | {r['VRAM (MB)']:<6.2f} | {r['Throughput']:<10.0f} | {r['Train Time (s)']:<10.2f} s | {r['Best Epoch']:<15} | {r['Accuracy']:<20} | {r['Curvature']}")
    print("="*155)
    
    plt.figure(figsize=(16, 10))
    plt.suptitle("Example of KART 1D nonlinear function learning how it looks likes", fontsize=16)
    
    plot_idx = 1
    for name, model in final_models_for_plot.items():
        if "ResMLP" in name: continue
            
        x_vals, y_vals, _ = extract_1d_learned_function(model, device)
        
        plt.subplot(2, 2, plot_idx)
        plt.plot(x_vals, y_vals, linewidth=2, color='darkblue')
        plt.title(f"{name}", fontsize=12, fontweight='bold')
        plt.xlabel("Input Feature (z)")
        plt.ylabel("Output Feature (y)")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        curv_val = next(item['Plot_Curv'] for item in summary_results if item["Model"] == name)
        acc_str = next(item['Accuracy'] for item in summary_results if item["Model"] == name)
        
        plt.text(0.05, 0.9, f"Curvature (Mean): {curv_val:.4f}\nAccuracy: {acc_str}%", 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        plot_idx += 1
        
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.savefig(BASE_DIR / 'results' / 'learned_1d_functions.png', dpi=300)
    plt.close()
    
    print(f"\n File results saved to '{BASE_DIR / 'results'}':")
    print("  - fmnist_raw_seeds.csv (Record of all seeds)")
    print("  - fmnist_summary.csv   (Summary of all seeds, Mean ± Std)")
    print("  - fmnist_summary.json  (JSON format for quick inspection)")
    print("  - learned_1d_functions.png")