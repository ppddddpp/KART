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
    train_model_with_time, set_seed, get_model,get_dataloaders,
    run_hardware_profiling, get_theoretical_flops_kart, get_theoretical_flops_mlp
)

def case2_cifar100(dataset_seed=42, seeds_list=[1, 2, 3]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(BASE_DIR / 'results', exist_ok=True)
    
    INPUT_DIM = 3 * 32 * 32
    OUTPUT_DIM = 100
    NUM_LAYERS = 3
    EPOCHS = 100
    BATCH_SIZE = 256
    
    SEEDS = seeds_list 
    
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name="CIFAR100", 
        base_dir=str(BASE_DIR / 'data'), 
        batch_size=BATCH_SIZE, 
        seed=dataset_seed
    )

    widths = [64, 128, 256, 512]
    model_names = ["ResMLP", "KART (Fourier)"] 

    raw_results = []
    summary_results = []
    summary_json = {}

    for d_model in widths:
        print(f"=== Training models with D_MODEL = {d_model} ===")
        summary_json[str(d_model)] = {}
        
        for name in model_names:
            print(f"  -> Training: [{name}] with {len(SEEDS)} seeds...")
            seed_test_accs = []
            seed_train_times = []
            seed_best_epochs = []
            params, peak_vram, throughput, theo_flops, prof_flops = 0, 0, 0, 0, 0
            
            for idx, seed in enumerate(SEEDS):
                set_seed(seed)
                model = get_model(name, INPUT_DIM, OUTPUT_DIM, d_model, NUM_LAYERS, device)
                
                if idx == 0:
                    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    peak_vram, throughput, prof_flops = run_hardware_profiling(model, device, (BATCH_SIZE, INPUT_DIM))
                    
                    if "ResMLP" in name: 
                        theo_flops = get_theoretical_flops_mlp(d_model, NUM_LAYERS, BATCH_SIZE, INPUT_DIM, OUTPUT_DIM)
                    else: 
                        theo_flops = get_theoretical_flops_kart(model.layers[0].config, d_model, NUM_LAYERS, BATCH_SIZE, INPUT_DIM, OUTPUT_DIM)
                
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
                
                seed_test_accs.append(test_acc)
                seed_train_times.append(train_time)
                seed_best_epochs.append(best_epoch)

                raw_results.append({
                    "Width": d_model,
                    "Model": name,
                    "Seed": seed,
                    "Accuracy": test_acc,
                    "Train Time": train_time,
                    "Best Epoch": best_epoch
                })
            
            mean_acc = np.mean(seed_test_accs)
            std_acc = np.std(seed_test_accs, ddof=1) if len(seed_test_accs) > 1 else np.std(seed_test_accs)
            mean_time = np.mean(seed_train_times)
            mean_epoch = np.mean(seed_best_epochs)
            std_epoch = np.std(seed_best_epochs, ddof=1) if len(seed_best_epochs) > 1 else np.std(seed_best_epochs)
            epoch_str = f"{mean_epoch:.1f} ± {std_epoch:.1f}" if len(SEEDS) > 1 else f"{mean_epoch:.0f}"
            acc_str = f"{mean_acc:.2f} ± {std_acc:.2f}" if len(SEEDS) > 1 else f"{mean_acc:.2f}"
            
            summary_results.append({
                "Width": d_model,
                "Model": name,
                "Params": params,
                "Theo FLOPs": theo_flops,
                "Prof FLOPs": prof_flops,
                "Throughput": round(throughput, 0),
                "VRAM": round(peak_vram, 2),
                "Train Time (s)": round(mean_time, 2),
                "Best Epoch": epoch_str,
                "Accuracy": acc_str
            })
            
            summary_json[str(d_model)][name] = {
                "Accuracy": {"mean": round(float(mean_acc), 4), "std": round(float(std_acc), 4), "raw": seed_test_accs},
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
            
            print(f"     => Acc: {acc_str}% | Throughput: {throughput:.0f} | VRAM: {peak_vram:.2f} MB\n")

    pd.DataFrame(raw_results).to_csv(BASE_DIR / 'results' / 'cifar100_raw_seeds.csv', index=False)
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(BASE_DIR / 'results' / 'cifar100_summary.csv', index=False)
    
    with open(BASE_DIR / 'results' / 'cifar100_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)

    print("\n" + "="*135)
    print(f"{'Width':<6} | {'Model':<15} | {'Params':<10} | {'FLOPs (Theo)':<15} | {'Throughput':<12} | {'Train Time':<14} | {'Best Epoch':<15} | {'Accuracy'}")
    print("-" * 135)
    for r in summary_results:
        print(f"{r['Width']:<6} | {r['Model']:<15} | {r['Params']:<10,} | {r['Theo FLOPs']:<15,} | {r['Throughput']:<12.0f} | {r['Train Time (s)']:<11.2f} s | {r['Best Epoch']:<15} | {r['Accuracy']}%")
    print("="*135)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    mlp_flops = df_summary[df_summary['Model'] == 'ResMLP']['Theo FLOPs'].tolist()
    kart_flops = df_summary[df_summary['Model'] == 'KART (Fourier)']['Theo FLOPs'].tolist()
    plt.plot(widths, mlp_flops, marker='s', color='red', linestyle='--', label='ResMLP (asymptotic O(n^2) block)')
    plt.plot(widths, kart_flops, marker='o', color='blue', label='KART (fixed-rank O(n) block)')
    plt.title('Estimated Arithmetic Cost Scaling')
    plt.xlabel('Network Width (d_model)')
    plt.ylabel('Estimated FLOPs per Batch Forward Pass (B=256)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    mlp_thru = df_summary[df_summary['Model'] == 'ResMLP']['Throughput'].tolist()
    kart_thru = df_summary[df_summary['Model'] == 'KART (Fourier)']['Throughput'].tolist()
    plt.plot(widths, mlp_thru, marker='s', color='red', linestyle='--', label='ResMLP')
    plt.plot(widths, kart_thru, marker='o', color='blue', label='KART')
    plt.title('Measured Inference Throughput')
    plt.xlabel('Network Width (d_model)')
    plt.ylabel('Samples / Second (Higher is better)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'results' / 'cifar100_crossover.png', dpi=300)
    plt.close()
    
    print(f"\n Saved results to '{BASE_DIR / 'results'}':")
    print("  - cifar100_raw_seeds.csv")
    print("  - cifar100_summary.csv")
    print("  - cifar100_summary.json")
    print("  - cifar100_crossover.png")