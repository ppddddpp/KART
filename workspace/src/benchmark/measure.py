import torch
from torch.profiler import profile, record_function, ProfilerActivity

def extract_1d_learned_function(model, device):
    layer = model.layers[0]
    d_model = layer.config.in_features
    
    x_vals = torch.linspace(0.0, 2.0, 300).to(device)
    
    x_tilde = torch.zeros(300, d_model).to(device)
    x_tilde[:, 0] = x_vals
    
    with torch.no_grad():
        z = layer.shifter(x_tilde)
        B_val = layer.basis(z)
        B_val_feature0 = B_val[:, 0, 0, :]
        c_weights = layer.c[0, :, 0]
        y_vals = torch.sum(B_val_feature0 * c_weights, dim=-1)
        
    # | y[i+1] - 2y[i] + y[i-1] | / (dz^2)
    dz = x_vals[1] - x_vals[0]
    d2y = y_vals[2:] - 2 * y_vals[1:-1] + y_vals[:-2]
    local_curvature = (torch.abs(d2y / (dz ** 2))).mean().item()
        
    return x_vals.cpu().numpy(), y_vals.cpu().numpy(), local_curvature

def get_theoretical_flops_kart(config, d_model, num_layers, batch_size, input_dim=784, output_dim=10):
    B, n, Q, K, D, M, R = batch_size, d_model, config.Q, config.K, config.D, config.M, config.R
    stem_macs = B * input_dim * d_model
    layer_macs = (B*n*Q*K*D) + (B*Q*D*M) + (B*Q*M*R) + (B*R*d_model)
    head_macs = B * d_model * output_dim
    return (stem_macs + (layer_macs * num_layers) + head_macs) * 2

def get_theoretical_flops_mlp(d_model, num_layers, batch_size, input_dim=784, output_dim=10):
    B = batch_size
    stem_macs = B * input_dim * d_model
    layer_macs = (B * d_model * d_model) * 2
    head_macs = B * d_model * output_dim
    return (stem_macs + (layer_macs * num_layers) + head_macs) * 2

def run_hardware_profiling(model, device, input_shape=(256, 784)):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    
    peak_vram = 0.0
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad(): _ = model(dummy_input)
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input)
    
    import time
    num_iterations = 100
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations): _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    throughput = (num_iterations * input_shape[0]) / (time.time() - start_time)
    
    model_cpu, dummy_cpu = model.to('cpu'), dummy_input.to('cpu')
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with record_function("inference"): model_cpu(dummy_cpu)
    model.to(device)
    prof_flops = sum([evt.flops for evt in prof.key_averages()])
    
    return peak_vram, throughput, prof_flops