import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from KART import KARTNet, KARTConfig
from other_net import ResMLPNet
import copy

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_model(name, input_dim, output_dim, d_model, num_layers, device, D_rank=8, R_rank=8):
    if "ResMLP" in name:
        return ResMLPNet(input_dim, output_dim, d_model, num_layers).to(device)
    elif "Fourier" in name:
        return KARTNet(input_dim, output_dim, d_model, num_layers, KARTConfig(
            in_features=d_model, out_features=d_model, Q=4, D=D_rank, R=R_rank, K=8, M=8, basis_type='fourier', act_type='silu')).to(device)
    elif "B-Spline Deg 3" in name:
        return KARTNet(input_dim, output_dim, d_model, num_layers, KARTConfig(
            in_features=d_model, out_features=d_model, Q=4, D=D_rank, R=R_rank, K=8, M=8, basis_type='b_spline', degree=3, act_type='silu')).to(device)
    elif "B-Spline Deg 1" in name:
        return KARTNet(input_dim, output_dim, d_model, num_layers, KARTConfig(
            in_features=d_model, out_features=d_model, Q=4, D=D_rank, R=R_rank, K=8, M=8, basis_type='b_spline', degree=1, act_type='silu')).to(device)
    elif "Linear" in name:
        return KARTNet(input_dim, output_dim, d_model, num_layers, KARTConfig(
            in_features=d_model, out_features=d_model, Q=4, D=D_rank, R=R_rank, K=8, M=8, basis_type='linear', act_type='silu')).to(device)
    raise ValueError(f"Unknown model name: {name}")

def train_model_with_time(name, model, train_loader, val_loader, epochs, device, patience=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-3)
    
    train_losses, val_accuracies = [], []
    start_time = time.time()
    
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        acc = 100 * correct / total
        val_accuracies.append(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            # print(f"   [Early Stopping] {name} has not improved for {patience} epochs. Stopping training at epoch {epoch+1}")
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    total_time = time.time() - start_time
    
    return train_losses, val_accuracies, total_time, best_epoch

def train_vit_model(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    start_time = time.time()
    best_acc, best_epoch = 0.0, 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
            
    return time.time() - start_time, best_epoch, best_acc