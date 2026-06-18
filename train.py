"""
train.py
========
Training script for the two-stage hip moment estimation pipeline.

Stage 1: IMU data → Joint angles (hip_angle, knee_angle)
Stage 2: Joint angles + velocities → Joint moments (hip_moment, knee_moment)

Uses the same TCN architecture from Molinaro et al., retrained on Camargo data.

Usage:
    python train.py --stage 1 --data_dir ./data_converted --epochs 100
    python train.py --stage 2 --data_dir ./data_converted --epochs 100
    python train.py --stage both --data_dir ./data_converted --epochs 100
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
from tcn import TCN
from dataloader_camargo import CamargoTcnDataset

# ============================================================================
# CONFIGURATION
# ============================================================================

# Stage 1: IMU → Joint Angles
STAGE1_CONFIG = {
    'name': 'Stage 1: IMU → Joint Angles',
    'input_names': [
        'shank_imu_r_gyro_x', 'shank_imu_r_gyro_y', 'shank_imu_r_gyro_z',
        'shank_imu_r_accel_x', 'shank_imu_r_accel_y', 'shank_imu_r_accel_z',
        'thigh_imu_r_gyro_x', 'thigh_imu_r_gyro_y', 'thigh_imu_r_gyro_z',
        'thigh_imu_r_accel_x', 'thigh_imu_r_accel_y', 'thigh_imu_r_accel_z',
    ],
    'label_names': [
        'hip_angle_r',
        'knee_angle_r',
    ],
    # TCN hyperparameters
    'num_channels': [64, 64, 64, 64, 64],  # 5 layers of 64 channels
    'kernel_size': 3,
    'dropout': 0.1,
    'norm': 'InstanceNorm1d',
    'model_save_name': 'trained_tcn_stage1_imu_to_angles.tar',
}

# Stage 2: Joint Angles → Joint Moments
STAGE2_CONFIG = {
    'name': 'Stage 2: Joint Angles → Joint Moments',
    'input_names': [
        'hip_angle_r', 'hip_angle_r_velocity_filt',
        'knee_angle_r', 'knee_angle_r_velocity_filt',
    ],
    'label_names': [
        'hip_flexion_r_moment',
        'knee_angle_r_moment',
    ],
    'num_channels': [64, 64, 64, 64, 64],
    'kernel_size': 3,
    'dropout': 0.1,
    'norm': 'InstanceNorm1d', 
    'model_save_name': 'trained_tcn_stage2_angles_to_moments.tar',
}


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def compute_normalization(dataset: CamargoTcnDataset, 
                          input_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std of input features across the entire dataset
    for normalization.
    """
    print("Computing normalization statistics...")
    all_data = []
    
    for i in range(len(dataset)):
        input_data, _, _ = dataset[i]
        all_data.append(input_data)
    
    # Concatenate along time dimension
    all_data = torch.cat(all_data, dim=2)  # (1, n_features, total_time)
    
    # Compute per-channel mean and std
    center = all_data.mean(dim=2, keepdim=True)  # (1, n_features, 1)
    scale = all_data.std(dim=2, keepdim=True)     # (1, n_features, 1)
    
    # Avoid division by zero
    scale = torch.clamp(scale, min=1e-8)
    
    print(f"  Input shape: {all_data.shape}")
    print(f"  Center (mean): {center.squeeze()}")
    print(f"  Scale (std):   {scale.squeeze()}")
    
    return center.squeeze(0).squeeze(-1), scale.squeeze(0).squeeze(-1)


def compute_effective_history(num_channels: List[int], kernel_size: int) -> int:
    """Compute the effective receptive field of the TCN."""
    num_levels = len(num_channels)
    eff_hist = 1
    for i in range(num_levels):
        dilation = 2 ** i
        eff_hist += 2 * (kernel_size - 1) * dilation
    return eff_hist


def train_one_epoch(model: TCN, 
                    dataloader: DataLoader, 
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    eff_hist: int) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for input_data, label_data, seq_lengths in dataloader:
        input_data = input_data.to(device)
        label_data = label_data.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(input_data)
        
        # Compute loss only on valid (non-padded) regions, after effective history
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        batch_size = input_data.shape[0]
        
        for b in range(batch_size):
            seq_len = seq_lengths[b]
            start = eff_hist
            end = seq_len
            
            if end <= start:
                continue
            
            pred = output[b, :, start:end]
            target = label_data[b, :, start:end]
            
            # Mask NaN values
            valid = ~torch.isnan(pred) & ~torch.isnan(target)
            if valid.any():
                loss = loss + criterion(pred[valid], target[valid])
        
        loss = loss / max(batch_size, 1)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model: TCN,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             eff_hist: int,
             label_names: List[str]) -> Dict:
    """Evaluate model. Returns dict with loss and per-output RMSE."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    per_output_errors = {name: [] for name in label_names}
    
    with torch.no_grad():
        for input_data, label_data, seq_lengths in dataloader:
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            
            output = model(input_data)
            batch_size = input_data.shape[0]
            
            batch_loss = 0.0
            for b in range(batch_size):
                seq_len = seq_lengths[b]
                start = eff_hist
                end = seq_len
                
                if end <= start:
                    continue
                
                pred = output[b, :, start:end]
                target = label_data[b, :, start:end]
                
                valid = ~torch.isnan(pred) & ~torch.isnan(target)
                if valid.any():
                    batch_loss += criterion(pred[valid], target[valid]).item()
                
                # Per-output RMSE
                for j, name in enumerate(label_names):
                    p = pred[j, :]
                    t = target[j, :]
                    v = ~torch.isnan(p) & ~torch.isnan(t)
                    if v.any():
                        rmse = torch.sqrt(torch.mean((p[v] - t[v])**2)).item()
                        per_output_errors[name].append(rmse)
            
            total_loss += batch_loss / max(batch_size, 1)
            n_batches += 1
    
    results = {
        'loss': total_loss / max(n_batches, 1),
        'rmse': {name: np.mean(errors) if errors else float('nan') 
                 for name, errors in per_output_errors.items()}
    }
    return results


def train_stage(config: Dict, 
                data_dir: str, 
                output_dir: str,
                epochs: int = 100,
                batch_size: int = 8,
                learning_rate: float = 1e-3,
                val_split: float = 0.2,
                device_str: str = 'cpu',
                participant_masses: Dict = None):
    """
    Complete training pipeline for one stage.
    """
    print("\n" + "=" * 70)
    print(f"  TRAINING: {config['name']}")
    print("=" * 70)
    
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load data ---
    input_names = config['input_names']
    label_names = config['label_names']
    
    if participant_masses is None:
        participant_masses = {}
    
    dataset = CamargoTcnDataset(
        data_dir=data_dir,
        input_names=input_names,
        label_names=label_names,
        side='r',
        participant_masses=participant_masses,
        device=torch.device('cpu')  # Load to CPU, move in training loop
    )
    
    n_trials = len(dataset)
    if n_trials == 0:
        print("ERROR: No trials found! Check your data_dir and column names.")
        return
    
    print(f"Found {n_trials} trials.")
    
    # --- Train/val split ---
    n_val = max(1, int(n_trials * val_split))
    n_train = n_trials - n_val
    
    # Shuffle trial indices
    indices = list(range(n_trials))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    print(f"Train: {n_train} trials, Val: {n_val} trials")
    
    # --- Compute normalization from training data ---
    # Load all training data to compute stats
    train_inputs = []
    for idx in train_indices:
        inp, _, _ = dataset[idx]
        train_inputs.append(inp)
    
    all_train = torch.cat(train_inputs, dim=2)
    center = all_train.mean(dim=2).squeeze(0).unsqueeze(-1)
    scale = all_train.std(dim=2).squeeze(0).unsqueeze(-1)
    scale = torch.clamp(scale, min=1e-8)
    
    print(f"Normalization - center: {center.shape}, scale: {scale.shape}")
    
    # --- Create model ---
    input_size = len(input_names)
    output_size = len(label_names)
    num_channels = config['num_channels']
    kernel_size = config['kernel_size']
    dropout = config['dropout']
    eff_hist = compute_effective_history(num_channels, kernel_size)
    
    model = TCN(
        input_size=input_size,
        output_size=output_size,
        num_channels=num_channels,
        ksize=kernel_size,
        dropout=dropout,
        eff_hist=eff_hist,
        center=center,
        scale=scale,
        norm=config.get('norm', 'weight_norm'),
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"  Input size:  {input_size} ({input_names})")
    print(f"  Output size: {output_size} ({label_names})")
    print(f"  Channels:    {num_channels}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Eff. history: {eff_hist} samples ({eff_hist * 5} ms at 200Hz)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    
    # --- Training setup ---
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # --- Training loop ---
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 70)
    
    for epoch in range(epochs):
        # Save a copy of model weights in case of NaN
        good_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Train
        model.train()
        train_loss = 0.0
        train_count = 0
        np.random.shuffle(train_indices)
        
        nan_detected = False
        for idx in train_indices:
            input_data, label_data, seq_lengths = dataset[idx]
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            
            optimizer.zero_grad()
            output = model(input_data)
            
            # Check for NaN in output
            if torch.isnan(output).any():
                nan_detected = True
                break
            
            seq_len = seq_lengths[0] if isinstance(seq_lengths, list) else seq_lengths
            start = eff_hist
            end = seq_len
            
            if end > start:
                pred = output[0, :, start:end]
                target = label_data[0, :, start:end]
                valid = torch.isfinite(pred) & torch.isfinite(target)
                if valid.any():
                    loss = criterion(pred[valid], target[valid])
                    if torch.isfinite(loss) and loss.item() > 0:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                        optimizer.step()
                         # Clamp weight_g to prevent NaN from weight_norm
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                if 'weight_g' in name:
                                    param.clamp_(min=1e-3)
                        train_loss += loss.item()
                        train_count += 1
        
        # If NaN detected, restore good weights and reduce learning rate
        if nan_detected:
            model.load_state_dict(good_state)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            new_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{epochs} | NaN detected! Restoring weights, reducing lr to {new_lr:.6f}")
            continue
        
        train_loss /= max(train_count, 1)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_count = 0
        val_rmse = {name: [] for name in label_names}
        
        with torch.no_grad():
            for idx in val_indices:
                input_data, label_data, seq_lengths = dataset[idx]
                input_data = input_data.to(device)
                label_data = label_data.to(device)
                
                output = model(input_data)
                
                if torch.isnan(output).any():
                    continue
                
                seq_len = seq_lengths[0] if isinstance(seq_lengths, list) else seq_lengths
                start = eff_hist
                end = seq_len
                
                if end > start:
                    pred = output[0, :, start:end]
                    target = label_data[0, :, start:end]
                    valid = torch.isfinite(pred) & torch.isfinite(target)
                    if valid.any():
                        val_loss += criterion(pred[valid], target[valid]).item()
                        val_count += 1
                    
                    for j, name in enumerate(label_names):
                        p = pred[j, :]
                        t = target[j, :]
                        v = torch.isfinite(p) & torch.isfinite(t)
                        if v.any():
                            rmse = torch.sqrt(torch.mean((p[v] - t[v])**2)).item()
                            val_rmse[name].append(rmse)
        
        val_loss /= max(val_count, 1)
        avg_rmse = {k: np.mean(v) if v else float('nan') for k, v in val_rmse.items()}
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss and val_loss > 0:
            best_val_loss = val_loss
            save_path = os.path.join(output_dir, config['model_save_name'])
            torch.save({
                'input_size': input_size,
                'output_size': output_size,
                'num_channels': num_channels,
                'ksize': kernel_size,
                'dropout': dropout,
                'eff_hist': eff_hist,
                'center': center,
                'scale': scale,
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'input_names': input_names,
                'label_names': label_names,
            }, save_path)
            marker = " *** BEST ***"
        else:
            marker = ""
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(avg_rmse)
        
        if epoch % 5 == 0 or marker:
            rmse_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_rmse.items()])
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"RMSE: {rmse_str}{marker}")
    
    # Save training history
    history_path = os.path.join(output_dir, config['model_save_name'].replace('.tar', '_history.json'))
    with open(history_path, 'w') as f:
        # Convert numpy to python types for JSON
        json_history = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_rmse': [{k: float(v) for k, v in d.items()} for d in history['val_rmse']],
        }
        json.dump(json_history, f, indent=2)
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {os.path.join(output_dir, config['model_save_name'])}")
    print(f"History saved to: {history_path}")
    
    return model, history


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN for hip moment estimation")
    parser.add_argument("--stage", type=str, required=True, choices=['1', '2', 'both'],
                        help="Which stage to train: 1 (IMU→angles), 2 (angles→moments), or both")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to converted data (output of camargo_adapter.py)")
    parser.add_argument("--output_dir", type=str, default="./trained_models",
                        help="Where to save trained models")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, cuda, mps")
    
    args = parser.parse_args()
    
    if args.stage in ['1', 'both']:
        train_stage(
            config=STAGE1_CONFIG,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device_str=args.device,
        )
    
    if args.stage in ['2', 'both']:
        train_stage(
            config=STAGE2_CONFIG,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device_str=args.device,
        )
