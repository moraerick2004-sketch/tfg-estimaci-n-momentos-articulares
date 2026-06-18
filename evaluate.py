"""
evaluate.py

Evaluacion completa del sistema de dos etapas. Calcula todas las metricas que
necesito para los resultados del TFG:

  1. Etapa 1 por separado (IMU -> angulos)
  2. Etapa 2 por separado (angulos reales -> momentos)
  3. Sistema completo (IMU -> angulos predichos -> momentos)
  4. Metricas por sujeto
  5. Metricas por modo de marcha (cinta vs suelo)
  6. Media +/- desviacion entre ensayos
  7. nRMSE (%) para reportar en la memoria
  8. Comparacion aproximada con el baseline de Molinaro et al.

Separo la Etapa 2 "pura" (con angulos reales) del sistema completo (con angulos
predichos por la Etapa 1) para ver cuanto error aporta cada etapa por su cuenta.

Salidas en output_dir:
    evaluation_summary.json   todas las metricas
    per_trial_results.csv     detalle por ensayo
    per_subject_results.csv   agregado por sujeto
    per_mode_results.csv      agregado por modo
    paper_table.txt           tablas formateadas para la memoria

Uso:
    python3 evaluate.py \
        --data_dir /ruta/data_molinaro_gait \
        --stage1_model ./trained_models/trained_tcn_stage1_imu_to_angles.tar \
        --stage2_model ./trained_models/trained_tcn_stage2_angles_to_moments.tar \
        --output_dir ./results
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from tcn import TCN
from dataloader_camargo import CamargoTcnDataset


# Funciones de metricas. Todas descartan NaN antes de calcular, porque en los
# momentos hay tramos sin contacto sobre plataforma que llegan como NaN.

def compute_velocity(angle_signal, dt=1/200.0):
    """Velocidad angular filtrada (diferencias centrales + media movil de 5)."""
    vel = np.gradient(angle_signal, dt)
    kernel = np.ones(5) / 5
    vel_filt = np.convolve(vel, kernel, mode='same')
    return vel_filt


def load_trained_model(model_path, device):
    """Carga un TCN entrenado desde un .tar."""
    model_info = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = model_info.pop('state_dict')

    extra_keys = ['epoch', 'val_loss', 'input_names', 'label_names']
    saved_meta = {}
    for key in extra_keys:
        if key in model_info:
            saved_meta[key] = model_info.pop(key)

    # Los modelos buenos se entrenaron con InstanceNorm1d. Si el checkpoint es
    # antiguo (weight_norm) y no acepta ese argumento, se reintenta sin forzarlo.
    try:
        model = TCN(**model_info, norm='InstanceNorm1d').to(device)
        model.load_state_dict(state_dict)
    except (TypeError, RuntimeError):
        try:
            model = TCN(**model_info).to(device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"ERROR loading {model_path}: {e}")
            raise

    model.eval()
    return model, saved_meta


def compute_rmse(pred, target):
    """RMSE entre dos series, ignorando NaN."""
    valid = np.isfinite(pred) & np.isfinite(target)
    if not valid.any():
        return np.nan
    return float(np.sqrt(np.mean((pred[valid] - target[valid]) ** 2)))


def compute_mae(pred, target):
    """Error absoluto medio, ignorando NaN."""
    valid = np.isfinite(pred) & np.isfinite(target)
    if not valid.any():
        return np.nan
    return float(np.mean(np.abs(pred[valid] - target[valid])))


def compute_r2(pred, target):
    """Coeficiente de determinacion R^2, ignorando NaN."""
    valid = np.isfinite(pred) & np.isfinite(target)
    if not valid.any():
        return np.nan
    ss_res = np.sum((target[valid] - pred[valid]) ** 2)
    ss_tot = np.sum((target[valid] - np.mean(target[valid])) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def compute_peak_max(target):
    """Pico absoluto de la senal real. Es el denominador del nRMSE."""
    valid = np.isfinite(target)
    if not valid.any():
        return np.nan
    return float(np.max(np.abs(target[valid])))


def evaluate_pipeline(data_dir, stage1_path, stage2_path, output_dir, device_str='cpu'):
    """Lanza la evaluacion completa y escribe los CSV/JSON de resultados."""
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

    print("Cargando modelos...")
    stage1_model, _ = load_trained_model(stage1_path, device)
    stage2_model, _ = load_trained_model(stage2_path, device)

    eff_hist1 = stage1_model.get_effective_history()
    eff_hist2 = stage2_model.get_effective_history()

    # Entradas/salidas de cada etapa.
    stage1_inputs = [
        'shank_imu_r_gyro_x', 'shank_imu_r_gyro_y', 'shank_imu_r_gyro_z',
        'shank_imu_r_accel_x', 'shank_imu_r_accel_y', 'shank_imu_r_accel_z',
        'thigh_imu_r_gyro_x', 'thigh_imu_r_gyro_y', 'thigh_imu_r_gyro_z',
        'thigh_imu_r_accel_x', 'thigh_imu_r_accel_y', 'thigh_imu_r_accel_z',
    ]
    angle_labels = ['hip_angle_r', 'knee_angle_r']
    moment_labels = ['hip_flexion_r_moment', 'knee_angle_r_moment']

    # El dataset carga de una vez las IMU (Etapa 1) y los angulos+velocidades
    # reales (Etapa 2), para poder evaluar las dos etapas con el mismo recorrido.
    all_inputs = stage1_inputs + [
        'hip_angle_r', 'hip_angle_r_velocity_filt',
        'knee_angle_r', 'knee_angle_r_velocity_filt',
    ]

    dataset = CamargoTcnDataset(
        data_dir=data_dir,
        input_names=all_inputs,
        label_names=moment_labels,
        side='r',
        device=device,
    )

    n_trials = len(dataset)
    print(f"Evaluating on {n_trials} trials...")

    # Particion por sujeto: aparto el ultimo 20% de los sujetos (por orden de
    # nombre) como conjunto no visto, para medir generalizacion a personas que
    # el modelo no ha visto entrenar. La particion es por SUJETO, no por ensayo,
    # para que no se filtren ensayos del mismo sujeto entre entrenamiento y test.
    all_subjects = sorted(set(
        os.path.basename(os.path.dirname(p))
        for p in dataset.trial_paths
    ))
    n_heldout = max(1, int(len(all_subjects) * 0.2))
    heldout_subjects = set(all_subjects[-n_heldout:])
    print(f"Sujetos no vistos ({n_heldout}/{len(all_subjects)}): {sorted(heldout_subjects)}")

    per_trial_records = []

    for i in range(n_trials):
        trial_path = dataset.trial_paths[i]
        subject = os.path.basename(os.path.dirname(trial_path))
        trial_name = os.path.basename(trial_path)

        # El modo de marcha va al principio del nombre del ensayo (treadmill_...).
        mode = trial_name.split('_')[0]

        is_heldout = subject in heldout_subjects

        input_data, label_data, seq_lengths = dataset[i]
        seq_len = seq_lengths[0]

        # Los 12 primeros canales son IMU; los 4 siguientes son los angulos y
        # velocidades reales (cadera y rodilla), intercalados.
        imu_data = input_data[:, :12, :]
        gt_angles_full = input_data[:, 12:, :]
        gt_moments = label_data

        with torch.no_grad():
            # Etapa 1: IMU -> angulos predichos
            predicted_angles = stage1_model(imu_data)
            hip_angle_pred = predicted_angles[0, 0, :].cpu().numpy()
            knee_angle_pred = predicted_angles[0, 1, :].cpu().numpy()

            # Para encadenar etapas, la velocidad del sistema completo se deriva
            # de los angulos PREDICHOS, no de los reales (asi seria en uso real).
            hip_vel_pred = compute_velocity(hip_angle_pred)
            knee_vel_pred = compute_velocity(knee_angle_pred)

            # Entrada de la Etapa 2 con orden intercalado [hip_ang, hip_vel,
            # knee_ang, knee_vel], el mismo con el que se entreno.
            stage2_input_pred = torch.zeros(1, 4, input_data.shape[2], device=device)
            stage2_input_pred[0, 0, :] = predicted_angles[0, 0, :]
            stage2_input_pred[0, 1, :] = torch.tensor(hip_vel_pred, device=device, dtype=torch.float32)
            stage2_input_pred[0, 2, :] = predicted_angles[0, 1, :]
            stage2_input_pred[0, 3, :] = torch.tensor(knee_vel_pred, device=device, dtype=torch.float32)

            # Etapa 2 "pura": momentos a partir de los angulos REALES.
            predicted_moments_gt = stage2_model(gt_angles_full)

            # Sistema completo: momentos a partir de los angulos predichos.
            predicted_moments_full = stage2_model(stage2_input_pred)

        # Se descartan las primeras muestras (warm-up causal de las dos TCN)
        # mas un margen de 10, porque ahi la salida aun no es fiable.
        start = max(eff_hist1, eff_hist2) + 10
        end = seq_len
        if end <= start:
            continue

        # Recorto al tramo valido y paso a numpy para calcular metricas.
        gt_hip_angle = gt_angles_full[0, 0, start:end].cpu().numpy()
        gt_knee_angle = gt_angles_full[0, 2, start:end].cpu().numpy()
        pred_hip_angle = predicted_angles[0, 0, start:end].cpu().numpy()
        pred_knee_angle = predicted_angles[0, 1, start:end].cpu().numpy()

        gt_hip_mom = gt_moments[0, 0, start:end].cpu().numpy()
        gt_knee_mom = gt_moments[0, 1, start:end].cpu().numpy()
        pred_hip_mom_s2 = predicted_moments_gt[0, 0, start:end].cpu().numpy()
        pred_knee_mom_s2 = predicted_moments_gt[0, 1, start:end].cpu().numpy()
        pred_hip_mom_full = predicted_moments_full[0, 0, start:end].cpu().numpy()
        pred_knee_mom_full = predicted_moments_full[0, 1, start:end].cpu().numpy()

        record = {
            'subject': subject,
            'trial': trial_name,
            'mode': mode,
            'heldout': is_heldout,
            'n_samples': end - start,

            # Etapa 1: IMU -> angulos
            'stage1_hip_rmse_deg': compute_rmse(pred_hip_angle, gt_hip_angle),
            'stage1_hip_mae_deg': compute_mae(pred_hip_angle, gt_hip_angle),
            'stage1_hip_r2': compute_r2(pred_hip_angle, gt_hip_angle),
            'stage1_hip_peak_max_deg': compute_peak_max(gt_hip_angle),

            'stage1_knee_rmse_deg': compute_rmse(pred_knee_angle, gt_knee_angle),
            'stage1_knee_mae_deg': compute_mae(pred_knee_angle, gt_knee_angle),
            'stage1_knee_r2': compute_r2(pred_knee_angle, gt_knee_angle),
            'stage1_knee_peak_max_deg': compute_peak_max(gt_knee_angle),

            # Etapa 2 pura: angulos reales -> momentos
            'stage2_hip_rmse_Nmkg': compute_rmse(pred_hip_mom_s2, gt_hip_mom),
            'stage2_hip_mae_Nmkg': compute_mae(pred_hip_mom_s2, gt_hip_mom),
            'stage2_hip_r2': compute_r2(pred_hip_mom_s2, gt_hip_mom),
            'stage2_hip_peak_max_Nmkg': compute_peak_max(gt_hip_mom),

            'stage2_knee_rmse_Nmkg': compute_rmse(pred_knee_mom_s2, gt_knee_mom),
            'stage2_knee_mae_Nmkg': compute_mae(pred_knee_mom_s2, gt_knee_mom),
            'stage2_knee_r2': compute_r2(pred_knee_mom_s2, gt_knee_mom),
            'stage2_knee_peak_max_Nmkg': compute_peak_max(gt_knee_mom),

            # Sistema completo: IMU -> angulos predichos -> momentos
            'pipeline_hip_rmse_Nmkg': compute_rmse(pred_hip_mom_full, gt_hip_mom),
            'pipeline_hip_mae_Nmkg': compute_mae(pred_hip_mom_full, gt_hip_mom),
            'pipeline_hip_r2': compute_r2(pred_hip_mom_full, gt_hip_mom),

            'pipeline_knee_rmse_Nmkg': compute_rmse(pred_knee_mom_full, gt_knee_mom),
            'pipeline_knee_mae_Nmkg': compute_mae(pred_knee_mom_full, gt_knee_mom),
            'pipeline_knee_r2': compute_r2(pred_knee_mom_full, gt_knee_mom),
        }

        per_trial_records.append(record)

        if i % 50 == 0:
            print(f"  Procesados {i}/{n_trials}")

    df = pd.DataFrame(per_trial_records)
    df.to_csv(os.path.join(output_dir, 'per_trial_results.csv'), index=False)
    print(f"\nResultados por ensayo guardados: {len(df)} filas")

    # Agregados (media +/- desviacion y nRMSE) sobre distintos subconjuntos.
    print("\nCalculando metricas agregadas...")

    summary = {}

    def aggregate(subset_df, label):
        """Media +/- desviacion de las columnas de metricas de un subconjunto."""
        if len(subset_df) == 0:
            return {}
        metrics = {}
        cols_to_aggregate = [c for c in subset_df.columns
                             if c.endswith('_rmse_deg') or c.endswith('_mae_deg')
                             or c.endswith('_rmse_Nmkg') or c.endswith('_mae_Nmkg')
                             or c.endswith('_r2')]
        for col in cols_to_aggregate:
            vals = subset_df[col].dropna().values
            metrics[f'{col}_mean'] = float(np.mean(vals)) if len(vals) > 0 else None
            metrics[f'{col}_std'] = float(np.std(vals)) if len(vals) > 0 else None
        metrics['n_trials'] = len(subset_df)
        return metrics

    def compute_nrmse_summary(subset_df):
        """nRMSE = RMSE / pico_absoluto_medio * 100. Misma definicion que uso
        en toda la memoria, para que las tablas sean comparables entre si."""
        if len(subset_df) == 0:
            return {}
        nrmse = {}
        for joint in ['hip', 'knee']:
            # Etapa 1 (angulos)
            rmse = subset_df[f'stage1_{joint}_rmse_deg'].mean()
            peak = subset_df[f'stage1_{joint}_peak_max_deg'].mean()
            nrmse[f'stage1_{joint}_nrmse_pct'] = float(rmse / peak * 100) if peak > 0 else None
            # Etapa 2 y sistema completo (momentos)
            rmse_s2 = subset_df[f'stage2_{joint}_rmse_Nmkg'].mean()
            rmse_full = subset_df[f'pipeline_{joint}_rmse_Nmkg'].mean()
            peak = subset_df[f'stage2_{joint}_peak_max_Nmkg'].mean()
            nrmse[f'stage2_{joint}_nrmse_pct'] = float(rmse_s2 / peak * 100) if peak > 0 else None
            nrmse[f'pipeline_{joint}_nrmse_pct'] = float(rmse_full / peak * 100) if peak > 0 else None
        return nrmse

    summary['all_trials'] = {**aggregate(df, 'all'), **compute_nrmse_summary(df)}
    summary['heldout_subjects'] = {**aggregate(df[df.heldout], 'heldout'),
                                    **compute_nrmse_summary(df[df.heldout])}
    summary['training_subjects'] = {**aggregate(df[~df.heldout], 'training'),
                                     **compute_nrmse_summary(df[~df.heldout])}

    # Por modo de marcha
    summary['per_mode'] = {}
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        summary['per_mode'][mode] = {**aggregate(mode_df, mode),
                                      **compute_nrmse_summary(mode_df)}

    # Por sujeto (solo los no vistos)
    summary['per_subject_heldout'] = {}
    for subj in sorted(df[df.heldout]['subject'].unique()):
        subj_df = df[df['subject'] == subj]
        summary['per_subject_heldout'][subj] = aggregate(subj_df, subj)

    # Guardar JSON con todo
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Tabla en texto plano, lista para volcar a la memoria.
    paper_table = []
    paper_table.append("=" * 80)
    paper_table.append("RESULTADOS DE EVALUACION PARA EL TFG")
    paper_table.append("=" * 80)
    paper_table.append("")
    paper_table.append(f"Dataset: Camargo et al. (2021), solo marcha (cinta + suelo)")
    paper_table.append(f"Ensayos totales: {len(df)}")
    paper_table.append(f"Sujetos totales: {len(df.subject.unique())}")
    paper_table.append(f"Sujetos de entrenamiento: {len(df[~df.heldout].subject.unique())}")
    paper_table.append(f"Sujetos no vistos: {sorted(df[df.heldout].subject.unique())}")
    paper_table.append("")

    paper_table.append("-" * 80)
    paper_table.append("TABLA 1: Resultados en sujetos de entrenamiento (validacion)")
    paper_table.append("-" * 80)
    s = summary['all_trials']
    paper_table.append(f"")
    paper_table.append(f"  ETAPA 1: IMU -> angulos articulares")
    paper_table.append(f"    Hip angle:  RMSE = {s['stage1_hip_rmse_deg_mean']:.2f} +/- {s['stage1_hip_rmse_deg_std']:.2f} deg "
                       f"(nRMSE = {s['stage1_hip_nrmse_pct']:.1f}%, R^2 = {s['stage1_hip_r2_mean']:.3f})")
    paper_table.append(f"    Knee angle: RMSE = {s['stage1_knee_rmse_deg_mean']:.2f} +/- {s['stage1_knee_rmse_deg_std']:.2f} deg "
                       f"(nRMSE = {s['stage1_knee_nrmse_pct']:.1f}%, R^2 = {s['stage1_knee_r2_mean']:.3f})")
    paper_table.append(f"")
    paper_table.append(f"  ETAPA 2: angulos -> momentos (con angulos reales)")
    paper_table.append(f"    Hip moment:  RMSE = {s['stage2_hip_rmse_Nmkg_mean']:.3f} +/- {s['stage2_hip_rmse_Nmkg_std']:.3f} Nm/kg "
                       f"(nRMSE = {s['stage2_hip_nrmse_pct']:.1f}%, R^2 = {s['stage2_hip_r2_mean']:.3f})")
    paper_table.append(f"    Knee moment: RMSE = {s['stage2_knee_rmse_Nmkg_mean']:.3f} +/- {s['stage2_knee_rmse_Nmkg_std']:.3f} Nm/kg "
                       f"(nRMSE = {s['stage2_knee_nrmse_pct']:.1f}%, R^2 = {s['stage2_knee_r2_mean']:.3f})")
    paper_table.append(f"")
    paper_table.append(f"  SISTEMA COMPLETO: IMU -> angulos predichos -> momentos")
    paper_table.append(f"    Hip moment:  RMSE = {s['pipeline_hip_rmse_Nmkg_mean']:.3f} +/- {s['pipeline_hip_rmse_Nmkg_std']:.3f} Nm/kg "
                       f"(nRMSE = {s['pipeline_hip_nrmse_pct']:.1f}%, R^2 = {s['pipeline_hip_r2_mean']:.3f})")
    paper_table.append(f"    Knee moment: RMSE = {s['pipeline_knee_rmse_Nmkg_mean']:.3f} +/- {s['pipeline_knee_rmse_Nmkg_std']:.3f} Nm/kg "
                       f"(nRMSE = {s['pipeline_knee_nrmse_pct']:.1f}%, R^2 = {s['pipeline_knee_r2_mean']:.3f})")

    if len(df[df.heldout]) > 0:
        paper_table.append("")
        paper_table.append("-" * 80)
        paper_table.append("TABLA 2: Sujetos no vistos (test de generalizacion)")
        paper_table.append("-" * 80)
        s = summary['heldout_subjects']
        paper_table.append(f"")
        paper_table.append(f"  ETAPA 1:")
        paper_table.append(f"    Hip angle:  RMSE = {s['stage1_hip_rmse_deg_mean']:.2f} +/- {s['stage1_hip_rmse_deg_std']:.2f} deg "
                           f"(nRMSE = {s['stage1_hip_nrmse_pct']:.1f}%)")
        paper_table.append(f"    Knee angle: RMSE = {s['stage1_knee_rmse_deg_mean']:.2f} +/- {s['stage1_knee_rmse_deg_std']:.2f} deg "
                           f"(nRMSE = {s['stage1_knee_nrmse_pct']:.1f}%)")
        paper_table.append(f"")
        paper_table.append(f"  ETAPA 2:")
        paper_table.append(f"    Hip moment:  RMSE = {s['stage2_hip_rmse_Nmkg_mean']:.3f} +/- {s['stage2_hip_rmse_Nmkg_std']:.3f} Nm/kg "
                           f"(nRMSE = {s['stage2_hip_nrmse_pct']:.1f}%)")
        paper_table.append(f"    Knee moment: RMSE = {s['stage2_knee_rmse_Nmkg_mean']:.3f} +/- {s['stage2_knee_rmse_Nmkg_std']:.3f} Nm/kg "
                           f"(nRMSE = {s['stage2_knee_nrmse_pct']:.1f}%)")
        paper_table.append(f"")
        paper_table.append(f"  SISTEMA COMPLETO:")
        paper_table.append(f"    Hip moment:  RMSE = {s['pipeline_hip_rmse_Nmkg_mean']:.3f} +/- {s['pipeline_hip_rmse_Nmkg_std']:.3f} Nm/kg "
                           f"(nRMSE = {s['pipeline_hip_nrmse_pct']:.1f}%)")
        paper_table.append(f"    Knee moment: RMSE = {s['pipeline_knee_rmse_Nmkg_mean']:.3f} +/- {s['pipeline_knee_rmse_Nmkg_std']:.3f} Nm/kg "
                           f"(nRMSE = {s['pipeline_knee_nrmse_pct']:.1f}%)")

    paper_table.append("")
    paper_table.append("-" * 80)
    paper_table.append("TABLA 3: Desglose por modo de marcha (todos los ensayos)")
    paper_table.append("-" * 80)
    for mode, mode_metrics in summary['per_mode'].items():
        paper_table.append(f"")
        paper_table.append(f"  {mode.upper()} (n={mode_metrics['n_trials']} trials):")
        paper_table.append(f"    Stage 1  - Hip:  {mode_metrics['stage1_hip_rmse_deg_mean']:.2f} deg "
                           f"({mode_metrics['stage1_hip_nrmse_pct']:.1f}%)")
        paper_table.append(f"    Stage 1  - Knee: {mode_metrics['stage1_knee_rmse_deg_mean']:.2f} deg "
                           f"({mode_metrics['stage1_knee_nrmse_pct']:.1f}%)")
        paper_table.append(f"    Stage 2  - Hip:  {mode_metrics['stage2_hip_rmse_Nmkg_mean']:.3f} Nm/kg "
                           f"({mode_metrics['stage2_hip_nrmse_pct']:.1f}%)")
        paper_table.append(f"    Stage 2  - Knee: {mode_metrics['stage2_knee_rmse_Nmkg_mean']:.3f} Nm/kg "
                           f"({mode_metrics['stage2_knee_nrmse_pct']:.1f}%)")
        paper_table.append(f"    Pipeline - Hip:  {mode_metrics['pipeline_hip_rmse_Nmkg_mean']:.3f} Nm/kg "
                           f"({mode_metrics['pipeline_hip_nrmse_pct']:.1f}%)")
        paper_table.append(f"    Pipeline - Knee: {mode_metrics['pipeline_knee_rmse_Nmkg_mean']:.3f} Nm/kg "
                           f"({mode_metrics['pipeline_knee_nrmse_pct']:.1f}%)")

    paper_table.append("")
    paper_table.append("-" * 80)
    paper_table.append("TABLA 4: Comparacion con el baseline de Molinaro et al. (2024)")
    paper_table.append("-" * 80)
    paper_table.append("")
    paper_table.append("  Nota: Molinaro uso un dataset privado (BT01-BT24), distinto de")
    paper_table.append("  Camargo y con otra poblacion. La comparacion numerica directa es")
    paper_table.append("  solo orientativa, no una equivalencia.")
    paper_table.append("")
    paper_table.append(f"  Metrica             | Molinaro (privado) | Este trabajo (Camargo)")
    paper_table.append(f"  -----------------   | -----------------  | ----------------------")
    s = summary['all_trials']
    paper_table.append(f"  RMSE momento cadera | ~0.13 Nm/kg        | {s['pipeline_hip_rmse_Nmkg_mean']:.3f} Nm/kg")
    paper_table.append(f"  RMSE momento rodilla| ~0.14 Nm/kg        | {s['pipeline_knee_rmse_Nmkg_mean']:.3f} Nm/kg")
    paper_table.append(f"  RMSE angulo cadera  | no reportado       | {s['stage1_hip_rmse_deg_mean']:.2f} deg")
    paper_table.append(f"  RMSE angulo rodilla | no reportado       | {s['stage1_knee_rmse_deg_mean']:.2f} deg")

    paper_table_str = "\n".join(paper_table)
    print("\n" + paper_table_str)

    with open(os.path.join(output_dir, 'paper_table.txt'), 'w') as f:
        f.write(paper_table_str)

    # CSV por sujeto
    subj_records = []
    for subj in sorted(df['subject'].unique()):
        subj_df = df[df['subject'] == subj]
        rec = {
            'subject': subj,
            'heldout': bool(subj_df.heldout.iloc[0]),
            'n_trials': len(subj_df),
            'stage1_hip_rmse_deg': subj_df['stage1_hip_rmse_deg'].mean(),
            'stage1_knee_rmse_deg': subj_df['stage1_knee_rmse_deg'].mean(),
            'stage2_hip_rmse_Nmkg': subj_df['stage2_hip_rmse_Nmkg'].mean(),
            'stage2_knee_rmse_Nmkg': subj_df['stage2_knee_rmse_Nmkg'].mean(),
            'pipeline_hip_rmse_Nmkg': subj_df['pipeline_hip_rmse_Nmkg'].mean(),
            'pipeline_knee_rmse_Nmkg': subj_df['pipeline_knee_rmse_Nmkg'].mean(),
        }
        subj_records.append(rec)

    pd.DataFrame(subj_records).to_csv(
        os.path.join(output_dir, 'per_subject_results.csv'), index=False)

    # CSV por modo
    mode_records = []
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        rec = {
            'mode': mode,
            'n_trials': len(mode_df),
            'stage1_hip_rmse_deg': mode_df['stage1_hip_rmse_deg'].mean(),
            'stage1_knee_rmse_deg': mode_df['stage1_knee_rmse_deg'].mean(),
            'stage2_hip_rmse_Nmkg': mode_df['stage2_hip_rmse_Nmkg'].mean(),
            'stage2_knee_rmse_Nmkg': mode_df['stage2_knee_rmse_Nmkg'].mean(),
            'pipeline_hip_rmse_Nmkg': mode_df['pipeline_hip_rmse_Nmkg'].mean(),
            'pipeline_knee_rmse_Nmkg': mode_df['pipeline_knee_rmse_Nmkg'].mean(),
        }
        mode_records.append(rec)

    pd.DataFrame(mode_records).to_csv(
        os.path.join(output_dir, 'per_mode_results.csv'), index=False)

    print("\n" + "=" * 80)
    print("SALIDAS GUARDADAS EN:", output_dir)
    print("  - per_trial_results.csv     (metricas por ensayo)")
    print("  - per_subject_results.csv   (agregado por sujeto)")
    print("  - per_mode_results.csv      (agregado por modo)")
    print("  - evaluation_summary.json   (todas las metricas)")
    print("  - paper_table.txt           (tablas para la memoria)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--stage1_model", type=str, required=True)
    parser.add_argument("--stage2_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_pipeline(
        data_dir=args.data_dir,
        stage1_path=args.stage1_model,
        stage2_path=args.stage2_model,
        output_dir=args.output_dir,
        device_str=args.device,
    )
