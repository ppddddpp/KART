import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import pandas as pd
from typing import Optional, cast, Sized
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import torchvision
import torchvision.transforms as T
import medmnist
from medmnist import INFO

from KART.config import KARTConfig
from KART.network import KARTNet
from other_net.resmlp import ResMLPNet
from other_net.low_rank_mlp import LowRankMLPNet

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

RAW_COLUMNS = [
    "Dataset",
    "Model",
    "Seed",
    "Status",
    "Total Params",
    "Branch Params",
    "Best Epoch",
    "Train Time",
    "Test Loss",
    "Accuracy",
    "Balanced Accuracy",
    "Macro F1",
    "AUROC",
    "Peak VRAM (MB)",
    "Initial Gradient Norm",
    "Maximum Gradient Norm",
    "Failure Reason",
]

class SqueezeTargetWrapper(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(cast(Sized, self.ds))
    def __getitem__(self, idx):
        img, target = self.ds[idx]
        img = img.view(-1) # flatten
        # PneumoniaMNIST target is e.g. [1], squeeze to scalar
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).squeeze().long()
        elif isinstance(target, torch.Tensor):
            target = target.squeeze().long()
        else:
            target = torch.tensor(target).squeeze().long()
        return img, target

class FlattenImageWrapper(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(cast(Sized, self.ds))
    def __getitem__(self, idx):
        img, target = self.ds[idx]
        return img.view(-1), target

def get_case14_dataset(dataset_name, base_dir):
    if dataset_name == "cifar10":
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        full_train = torchvision.datasets.CIFAR10(root=base_dir, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(root=base_dir, train=False, download=True, transform=test_transform)
        
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(cast(Sized, full_train)), generator=g).tolist()
        train_idx, val_idx = indices[:-5000], indices[-5000:]
        train_ds = Subset(full_train, train_idx)
        
        val_full = torchvision.datasets.CIFAR10(root=base_dir, train=True, download=True, transform=test_transform)
        val_ds = Subset(val_full, val_idx)
        
        return FlattenImageWrapper(train_ds), FlattenImageWrapper(val_ds), FlattenImageWrapper(test_ds), 3072, 10
        
    elif dataset_name in ["pathmnist", "pneumoniamnist"]:
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        
        pass
        
        mean = (0.5,) if info['n_channels'] == 1 else (0.5, 0.5, 0.5)
        std = (0.5,) if info['n_channels'] == 1 else (0.5, 0.5, 0.5)
        
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        train_ds = DataClass(split='train', download=True, root=base_dir, transform=transform, as_rgb=False)
        val_ds = DataClass(split='val', download=True, root=base_dir, transform=transform, as_rgb=False)
        test_ds = DataClass(split='test', download=True, root=base_dir, transform=transform, as_rgb=False)
        
        train_ds = SqueezeTargetWrapper(train_ds)
        val_ds = SqueezeTargetWrapper(val_ds)
        test_ds = SqueezeTargetWrapper(test_ds)
        
        print(f"MedMNIST version: {medmnist.__version__}, root: {base_dir}")
        print(f"Split sizes: train={len(cast(Sized, train_ds))}, val={len(cast(Sized, val_ds))}, test={len(cast(Sized, test_ds))}")
        
        d_in = 28 * 28 * info['n_channels']
        num_classes = len(info['label'])
        return train_ds, val_ds, test_ds, d_in, num_classes
    
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

def create_model(model_name, d_in, num_classes):
    d_model = 64
    L = 3
    if model_name == "Dense ResMLP":
        return ResMLPNet(d_in, num_classes, d_model, L)
    elif model_name == "LR-MLP Exact-Parity":
        return LowRankMLPNet(d_in, num_classes, d_model, L, D=15, R=16, H=99, use_domain_bound=True)
    elif model_name == "KART Fourier":
        cfg = KARTConfig(
            in_features=d_in, out_features=num_classes,
            Q=4, D=8, R=8, K=8, M=8, basis_type='fourier', use_domain_bound=True
        )
        return KARTNet(d_in, num_classes, d_model, L, cfg)
    elif model_name == "KART B-Spline":
        cfg = KARTConfig(
            in_features=d_in, out_features=num_classes,
            Q=4, D=8, R=8, K=8, M=8, basis_type='b_spline', degree=3, use_domain_bound=True
        )
        return KARTNet(d_in, num_classes, d_model, L, cfg)
    else:
        raise ValueError(f"Unknown model {model_name}")

def run_preflight_validation(dataset_name, device, base_dir):
    print("Running preflight validation...")
    _, _, _, d_in, num_classes = get_case14_dataset(dataset_name, base_dir)
    
    kart = create_model("KART Fourier", d_in, num_classes)
    lr = create_model("LR-MLP Exact-Parity", d_in, num_classes)
    
    if hasattr(kart, "count_branch_parameters"):
        kart_branch = getattr(kart, "count_branch_parameters")()
    else:
        kart_layers = getattr(kart, "layers", [])
        kart_branch = sum(p.numel() for l in kart_layers for p in l.parameters() if p.requires_grad)
        
    if hasattr(lr, "count_branch_parameters"):
        lr_branch = getattr(lr, "count_branch_parameters")()
    else:
        lr_layers = getattr(lr, "layers", [])
        lr_branch = sum(p.numel() for l in lr_layers for p in l.parameters() if p.requires_grad)
        
    assert kart_branch == lr_branch, f"Parity mismatch! KART: {kart_branch}, LR: {lr_branch}"
    
    for m in [kart, lr]:
        m.to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        x = torch.randn(2, d_in, device=device)
        y = torch.randint(0, num_classes, (2,), device=device)
        
        out = m(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "NaN/Inf gradient found in preflight"
        opt.step()
        del m
    print("Preflight validation passed.")

def worker_init_fn(worker_id, seed):
    set_seed(seed + worker_id)

def train_one_model(model_name, dataset_name, seed, suite, device, base_dir):
    set_seed(seed)
    train_ds, val_ds, test_ds, d_in, num_classes = get_case14_dataset(dataset_name, base_dir)
    
    g_train = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, 
        generator=g_train, num_workers=2, pin_memory=True,
        worker_init_fn=seed_worker
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    model = create_model(model_name, d_in, num_classes).to(device)
    total_params = getattr(model, "count_parameters")() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters() if p.requires_grad)
    if hasattr(model, "count_branch_parameters"):
        branch_params = getattr(model, "count_branch_parameters")()
    else:
        layers = getattr(model, "layers", [])
        branch_params = sum(p.numel() for l in layers for p in l.parameters() if p.requires_grad)
        
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    
    epochs = 1 if suite == "smoke" else 50
    patience = 15
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0
    
    history = []
    initial_grad_norm = None
    max_grad_norm = 0.0
    
    t0 = time.time()
    
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            if not torch.isfinite(loss):
                raise ValueError(f"Loss is {loss.item()} at epoch {ep}")
            loss.backward()
            
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
            if not torch.isfinite(torch.tensor(gnorm)):
                raise ValueError(f"Gradient norm is {gnorm} at epoch {ep}")
            if initial_grad_norm is None: initial_grad_norm = gnorm
            max_grad_norm = max(max_grad_norm, gnorm)
            
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(cast(Sized, train_ds))
        
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += F.cross_entropy(out, y).item() * x.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
        val_loss /= len(cast(Sized, val_ds))
        val_acc = correct / len(cast(Sized, val_ds))
        
        history.append({
            "Dataset": dataset_name, "Model": model_name, "Seed": seed,
            "Epoch": ep+1, "Train Loss": train_loss, "Val Loss": val_loss,
            "Val Accuracy": val_acc, "Gradient Norm": max_grad_norm
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and suite == "full":
                break
                
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_loss = 0.0
    all_y, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_loss += F.cross_entropy(out, y).item() * x.size(0)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_y.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    test_loss /= len(cast(Sized, test_ds))
    acc = accuracy_score(all_y, all_preds)
    bal_acc = balanced_accuracy_score(all_y, all_preds)
    f1 = f1_score(all_y, all_preds, average='macro')
    
    if num_classes == 2:
        auroc = roc_auc_score(all_y, [p[1] for p in all_probs])
    else:
        auroc = roc_auc_score(all_y, all_probs, multi_class='ovr', average='macro')
        
    peak_vram = torch.cuda.max_memory_allocated(device)/1024**2 if torch.cuda.is_available() else 0
    
    result = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Seed": seed,
        "Status": "COMPLETED",
        "Total Params": total_params,
        "Branch Params": branch_params,
        "Best Epoch": best_epoch,
        "Train Time": train_time,
        "Test Loss": test_loss,
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "Macro F1": f1,
        "AUROC": auroc,
        "Peak VRAM (MB)": peak_vram,
        "Initial Gradient Norm": initial_grad_norm,
        "Maximum Gradient Norm": max_grad_norm,
        "Failure Reason": ""
    }
    return result, history

from typing import Optional

def run_case14(dataset_name: str, suite: str = "full", max_hours: Optional[float] = None):
    print(f"Starting Case 14 for {dataset_name} | Suite: {suite}")
    
    base_dir = Path(__file__).resolve().parent.parent.parent / "data"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    out_dir = Path(__file__).resolve().parent.parent.parent / "results" / suite / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_csv = out_dir / "raw.csv"
    hist_csv = out_dir / "history.csv"
    sum_csv = out_dir / "summary.csv"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    run_info = {
        "dataset": dataset_name,
        "suite": suite,
        "device": str(device)
    }
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=4)
    
    run_preflight_validation(dataset_name, device, base_dir)
    
    models = ["Dense ResMLP", "LR-MLP Exact-Parity", "KART Fourier", "KART B-Spline"]
    seeds = [1, 2, 3] if suite == "full" else [1]
    
    completed = set()
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        df = (
            df.sort_values(["Dataset", "Model", "Seed"])
              .drop_duplicates(subset=["Dataset", "Model", "Seed"], keep="last")
        )
        df.to_csv(raw_csv, index=False)
        comp_df = df[df["Status"] == "COMPLETED"]
        completed = set(zip(comp_df["Model"], comp_df["Seed"]))
        
    if hist_csv.exists():
        hdf = pd.read_csv(hist_csv)
        hdf = (
            hdf.sort_values(["Dataset", "Model", "Seed", "Epoch"])
               .drop_duplicates(subset=["Dataset", "Model", "Seed", "Epoch"], keep="last")
        )
        hdf.to_csv(hist_csv, index=False)
        
    start_time = time.time()
    time_up = False
    
    for seed in seeds:
        for m_name in models:
            if max_hours and (time.time() - start_time) / 3600.0 > max_hours:
                print(f"[TIME LIMIT REACHED] Stopping before {m_name} Seed {seed}")
                time_up = True
                break
                
            if (m_name, seed) in completed:
                print(f"[SKIP] {m_name} Seed {seed} already completed.")
                continue
                
            print(f"--- Training {m_name} | Seed {seed} ---")
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            
            # Remove any failed prior attempts from CSVs
            if raw_csv.exists():
                df = pd.read_csv(raw_csv)
                df = df[~((df["Model"] == m_name) & (df["Seed"] == seed))]
                df.to_csv(raw_csv, index=False)
            if hist_csv.exists():
                hdf = pd.read_csv(hist_csv)
                hdf = hdf[~((hdf["Model"] == m_name) & (hdf["Seed"] == seed))]
                hdf.to_csv(hist_csv, index=False)
                
            try:
                res, hist = train_one_model(m_name, dataset_name, seed, suite, device, base_dir)
                
                res_df = pd.DataFrame([res]).reindex(columns=RAW_COLUMNS)
                res_df.to_csv(raw_csv, mode='a', header=not raw_csv.exists(), index=False)
                
                hist_df = pd.DataFrame(hist)
                hist_df.to_csv(hist_csv, mode='a', header=not hist_csv.exists(), index=False)
                
                if suite == "smoke" and m_name == models[-1]:
                    # Estimate full time
                    smoke_time = time.time() - start_time
                    est_ratio = 50 # assuming 50 epochs max
                    print(f"[Forecast] Full suite estimated time: {smoke_time * 3 * est_ratio / 3600:.2f} hours")
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                res = {
                    "Dataset": dataset_name, "Model": m_name, "Seed": seed,
                    "Status": "FAILED", "Failure Reason": str(e)
                }
                pd.DataFrame([res]).reindex(columns=RAW_COLUMNS).to_csv(raw_csv, mode='a', header=not raw_csv.exists(), index=False)
        if time_up:
            break

    if raw_csv.exists():
        df = pd.read_csv(raw_csv).drop_duplicates(
            subset=["Dataset", "Model", "Seed"], keep="last"
        )
        successful = (df["Status"] == "COMPLETED").sum()
        failed = (df["Status"] == "FAILED").sum()
        summary = {"Dataset": dataset_name, "Successful Runs": successful, "Failed Runs": failed}
        pd.DataFrame([summary]).to_csv(sum_csv, index=False)
        print("Done.")

