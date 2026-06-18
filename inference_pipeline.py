"""
inference_pipeline.py
=====================
Runs the full two-stage pipeline:
    IMU data → [Stage 1 TCN] → Joint Angles → [Stage 2 TCN] → Joint Moments

Can also run each stage independently for testing.

Usage:
    # Full pipeline: IMU → moments
    python inference_pipeline.py --data_dir ./data_converted \
        --stage1_model ./trained_models/trained_tcn_stage1_imu_to_angles.tar \
        --stage2_model ./trained_models/trained_tcn_stage2_angles_to_moments.tar

    # Stage 1 only: IMU → angles
    python inference_pipeline.py --data_dir ./data_converted \
        --stage1_model ./trained_models/trained_tcn_stage1_imu_to_angles.tar \
        --stage_only 1

    # Stage 2 only: angles → moments (uses ground truth angles)
    python inference_pipeline.py --data_dir ./data_converted \
        --stage2_model ./trained_models/trained_tcn_stage2_angles_to_moments.tar \
        --stage_only 2
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional
from tcn import TCN
from dataloader_camargo import CamargoTcnDataset


def load_trained_model(model_path: str, device: torch.device) -> TCN:
    """Load a trained TCN model from a .tar file."""
    model_info = torch.load(model_path, map_location=device)
    state_dict = model_info.pop('state_dict')
    
    # Remove non-TCN keys
    extra_keys = ['epoch', 'val_loss', 'input_names', 'label_names']
    saved_meta = {}
    for key in extra_keys:
        if key in model_info:
            saved_meta[key] = model_info.pop(key)
    
    model = TCN(**model_info).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"  Input size: {model_info.get('input_size')}")
    print(f"  Output size: {model_info.get('output_size')}")
    if 'epoch' in saved_meta:
        print(f"  Trained for {saved_meta['epoch']} epochs")
    if 'val_loss' in saved_meta:
        print(f"  Best val loss: {saved_meta['val_loss']:.6f}")
    
    return model, saved_meta


def compute_velocity(angle_signal: np.ndarray, dt: float = 1/200.0) -> np.ndarray:
    """Compute filtered velocity from angle signal."""
    vel = np.gradient(angle_signal, dt)
    # Simple moving average filter
    kernel = np.ones(5) / 5
    vel_filt = np.convolve(vel, kernel, mode='same')
    return vel_filt


def run_full_pipeline(data_dir: str,
                      stage1_model_path: str,
                      stage2_model_path: str,
                      device_str: str = 'cpu',
                      n_trials: int = 5,
                      save_plots: bool = True,
                      output_dir: str = './results'):
    """
    Run the complete IMU → angles → moments pipeline.
    """
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    stage1_model, stage1_meta = load_trained_model(stage1_model_path, device)
    stage2_model, stage2_meta = load_trained_model(stage2_model_path, device)
    
    # Stage 1 inputs: IMU data
    stage1_inputs = [
        'shank_imu_r_gyro_x', 'shank_imu_r_gyro_y', 'shank_imu_r_gyro_z',
        'shank_imu_r_accel_x', 'shank_imu_r_accel_y', 'shank_imu_r_accel_z',
        'thigh_imu_r_gyro_x', 'thigh_imu_r_gyro_y', 'thigh_imu_r_gyro_z',
        'thigh_imu_r_accel_x', 'thigh_imu_r_accel_y', 'thigh_imu_r_accel_z',
    ]
    
    # Ground truth labels for evaluation
    angle_labels = ['hip_angle_r', 'knee_angle_r']
    moment_labels = ['hip_flexion_r_moment', 'knee_angle_r_moment']
    
    # Load dataset with IMU inputs (for Stage 1)
    # We also need ground truth angles and moments for evaluation
    all_inputs = stage1_inputs + ['hip_angle_r', 'hip_angle_r_velocity_filt',
                                   'knee_angle_r', 'knee_angle_r_velocity_filt']
    
    dataset = CamargoTcnDataset(
        data_dir=data_dir,
        input_names=all_inputs,
        label_names=moment_labels,
        side='r',
        device=device,
    )
    
    n_eval = min(n_trials, len(dataset))
    print(f"\nRunning pipeline on {n_eval} trials...")
    
    results = {
        'stage1_hip_rmse': [], 'stage1_knee_rmse': [],
        'stage2_hip_rmse': [], 'stage2_knee_rmse': [],
        'pipeline_hip_rmse': [], 'pipeline_knee_rmse': [],
    }
    
    for i in range(n_eval):
        input_data, label_data, seq_lengths = dataset[i]
        seq_len = seq_lengths[0]
        
        # Split input: first 12 channels are IMU, rest are ground truth angles
        imu_data = input_data[:, :12, :]  # IMU channels
        gt_angles = input_data[:, 12:, :]  # Ground truth: hip_angle, hip_vel, knee_angle, knee_vel
        gt_moments = label_data  # Ground truth moments
        
        eff_hist1 = stage1_model.get_effective_history()
        eff_hist2 = stage2_model.get_effective_history()
        
        with torch.no_grad():
            # --- STAGE 1: IMU → Angles ---
            predicted_angles = stage1_model(imu_data)  # (1, 2, T) → hip_angle, knee_angle
            
            # Compute velocities from predicted angles
            hip_angle_pred = predicted_angles[0, 0, :].cpu().numpy()
            knee_angle_pred = predicted_angles[0, 1, :].cpu().numpy()
            
            hip_vel_pred = compute_velocity(hip_angle_pred)
            knee_vel_pred = compute_velocity(knee_angle_pred)
            
            # Build Stage 2 input from predicted angles
            stage2_input = torch.zeros(1, 4, input_data.shape[2], device=device)
            stage2_input[0, 0, :] = predicted_angles[0, 0, :]  # hip angle
            stage2_input[0, 1, :] = torch.tensor(hip_vel_pred, device=device)
            stage2_input[0, 2, :] = predicted_angles[0, 1, :]  # knee angle
            stage2_input[0, 3, :] = torch.tensor(knee_vel_pred, device=device)
            
            # --- STAGE 2: Angles → Moments (using predicted angles) ---
            predicted_moments_pipeline = stage2_model(stage2_input)
            
            # --- STAGE 2 STANDALONE: Using ground truth angles ---
            predicted_moments_gt = stage2_model(gt_angles)
        
        # Compute metrics on valid region
        start = max(eff_hist1, eff_hist2) + 10  # Extra margin
        end = seq_len
        
        if end <= start:
            continue
        
        # Stage 1 RMSE (angle prediction)
        for j, name in enumerate(['hip', 'knee']):
            pred = predicted_angles[0, j, start:end].cpu().numpy()
            gt = gt_angles[0, j*2, start:end].cpu().numpy()  # j*2 because angles are at indices 0, 2
            valid = ~np.isnan(pred) & ~np.isnan(gt)
            if valid.any():
                rmse = np.sqrt(np.mean((pred[valid] - gt[valid])**2))
                results[f'stage1_{name}_rmse'].append(rmse)
        
        # Stage 2 standalone RMSE (moment from GT angles)
        for j, name in enumerate(['hip', 'knee']):
            pred = predicted_moments_gt[0, j, start:end].cpu().numpy()
            gt = gt_moments[0, j, start:end].cpu().numpy()
            valid = ~np.isnan(pred) & ~np.isnan(gt)
            if valid.any():
                rmse = np.sqrt(np.mean((pred[valid] - gt[valid])**2))
                results[f'stage2_{name}_rmse'].append(rmse)
        
        # Full pipeline RMSE (moment from predicted angles)
        for j, name in enumerate(['hip', 'knee']):
            pred = predicted_moments_pipeline[0, j, start:end].cpu().numpy()
            gt = gt_moments[0, j, start:end].cpu().numpy()
            valid = ~np.isnan(pred) & ~np.isnan(gt)
            if valid.any():
                rmse = np.sqrt(np.mean((pred[valid] - gt[valid])**2))
                results[f'pipeline_{name}_rmse'].append(rmse)
        
        # --- Plot results for this trial ---
        if save_plots and i < 5:  # Plot first 5 trials
            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            time = np.arange(start, end) / 200.0  # Convert to seconds
            
            # Row 1: Stage 1 - Angle predictions
            for j, (name, unit) in enumerate([('Hip Angle', '°'), ('Knee Angle', '°')]):
                ax = axes[0, j]
                gt_vals = gt_angles[0, j*2, start:end].cpu().numpy()
                pred_vals = predicted_angles[0, j, start:end].cpu().numpy()
                ax.plot(time, gt_vals, 'b-', label='Ground Truth', linewidth=1)
                ax.plot(time, pred_vals, 'r--', label='Stage 1 Prediction', linewidth=1)
                ax.set_ylabel(f'{name} ({unit})')
                ax.set_title(f'Stage 1: {name}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Row 2: Stage 2 standalone - Moment from GT angles
            for j, (name, unit) in enumerate([('Hip Moment', 'Nm/kg'), ('Knee Moment', 'Nm/kg')]):
                ax = axes[1, j]
                gt_vals = gt_moments[0, j, start:end].cpu().numpy()
                pred_vals = predicted_moments_gt[0, j, start:end].cpu().numpy()
                ax.plot(time, gt_vals, 'b-', label='Ground Truth', linewidth=1)
                ax.plot(time, pred_vals, 'r--', label='Stage 2 (GT angles)', linewidth=1)
                ax.set_ylabel(f'{name} ({unit})')
                ax.set_title(f'Stage 2 Standalone: {name}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Row 3: Full pipeline - Moment from predicted angles
            for j, (name, unit) in enumerate([('Hip Moment', 'Nm/kg'), ('Knee Moment', 'Nm/kg')]):
                ax = axes[2, j]
                gt_vals = gt_moments[0, j, start:end].cpu().numpy()
                pred_vals = predicted_moments_pipeline[0, j, start:end].cpu().numpy()
                ax.plot(time, gt_vals, 'b-', label='Ground Truth', linewidth=1)
                ax.plot(time, pred_vals, 'r--', label='Full Pipeline', linewidth=1)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(f'{name} ({unit})')
                ax.set_title(f'Full Pipeline: {name}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            fig.suptitle(f'Trial {i} - Two-Stage Pipeline Results', fontsize=14)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'trial_{i:03d}_results.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Saved plot: {plot_path}")
    
    # --- Print summary ---
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    
    for key, values in results.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {key:30s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nUnits: angles in degrees (°), moments in Nm/kg")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run two-stage inference pipeline")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--stage1_model", type=str, default=None)
    parser.add_argument("--stage2_model", type=str, default=None)
    parser.add_argument("--stage_only", type=int, default=None, choices=[1, 2])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./results")
    
    args = parser.parse_args()
    
    if args.stage_only is None:
        # Full pipeline
        assert args.stage1_model and args.stage2_model, \
            "Both --stage1_model and --stage2_model required for full pipeline"
        run_full_pipeline(
            data_dir=args.data_dir,
            stage1_model_path=args.stage1_model,
            stage2_model_path=args.stage2_model,
            device_str=args.device,
            n_trials=args.n_trials,
            output_dir=args.output_dir,
        )
    else:
        print(f"Stage-only mode ({args.stage_only}) - use train.py for standalone evaluation")
