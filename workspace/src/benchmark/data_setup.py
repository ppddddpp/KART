import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_dataloaders(dataset_name="FashionMNIST", base_dir="./data", batch_size=256, seed=42):
    if dataset_name == "FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data_aug = torchvision.datasets.FashionMNIST(root=base_dir, train=True, download=True, transform=transform)
        train_data_eval = torchvision.datasets.FashionMNIST(root=base_dir, train=True, download=True, transform=transform)
        test_data = torchvision.datasets.FashionMNIST(root=base_dir, train=False, download=True, transform=transform)
        
    elif dataset_name == "CIFAR100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_data_aug = torchvision.datasets.CIFAR100(root=base_dir, train=True, download=True, transform=train_transform)
        train_data_eval = torchvision.datasets.CIFAR100(root=base_dir, train=True, download=True, transform=test_transform)
        test_data = torchvision.datasets.CIFAR100(root=base_dir, train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    g_split = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_data_aug), generator=g_split).tolist()
    
    train_size = int(0.9 * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_ds = Subset(train_data_aug, train_indices)
    val_ds = Subset(train_data_eval, val_indices)
    test_ds = test_data
    
    print(f"Dataset {dataset_name} | Train: {len(train_ds)} | Val: {len(val_ds)} | Official Test: {len(test_ds)}")
    g_data = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g_data, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader