"""
generate_figures.py

Genera las figuras de resultados del TFG a partir de los modelos entrenados y
del CSV de evaluacion (per_trial_results.csv de evaluate.py).

Figuras:
    fig1_pipeline.png                  esquema del sistema de dos etapas
    fig2_stage1_example.png            ejemplo cualitativo de la Etapa 1
    fig3_moment_rmse_distribution.png  distribucion del RMSE de momentos
    fig4_stage2_vs_pipeline_rmse.png   Etapa 2 vs sistema completo
    fig5_per_mode_bars.png             comparacion por modo de marcha
    fig6_subject_box.png               dispersion por sujeto

Para las figuras de momentos uso distribuciones sobre TODOS los ensayos en vez
de un ensayo suelto: un unico ensayo elegido a ojo puede enganar, y un boxplot
agregado es mas honesto y defendible.

Uso:
    python3 generate_figures.py \\
        --data_dir /ruta/data_molinaro_gait \\
        --stage1_model ./trained_models/trained_tcn_stage1_imu_to_angles.tar \\
        --stage2_model ./trained_models/trained_tcn_stage2_angles_to_moments.tar \\
        --results_csv ./results/per_trial_results.csv \\
        --output_dir ./figures
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Rectangle
from tcn import TCN
from dataloader_camargo import CamargoTcnDataset


def load_trained_model(model_path, device):
    """Carga un TCN entrenado. Igual que en evaluate.py: prueba InstanceNorm1d
    y cae a la norma por defecto si el checkpoint es de los antiguos."""
    model_info = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = model_info.pop('state_dict')
    for key in ['epoch', 'val_loss', 'input_names', 'label_names']:
        if key in model_info:
            model_info.pop(key)
    try:
        model = TCN(**model_info, norm='InstanceNorm1d').to(device)
        model.load_state_dict(state_dict)
    except (TypeError, RuntimeError):
        model = TCN(**model_info).to(device)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_velocity(signal, dt=1/200.0):
    vel = np.gradient(signal, dt)
    kernel = np.ones(5) / 5
    return np.convolve(vel, kernel, mode='same')


# Figura 1: diagrama del sistema de dos etapas (dibujado a mano con cajas).
def fig1_pipeline(output_dir):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Caja 1: entrada IMU
    box1 = Rectangle((0.2, 1.3), 1.8, 1.4, linewidth=2,
                     edgecolor='steelblue', facecolor='lightblue', alpha=0.7)
    ax.add_patch(box1)
    ax.text(1.1, 2.3, 'IMU\n(12 canales)', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(1.1, 1.7, 'Muslo + Pantorrilla\nAcel + Giro (x,y,z)',
            ha='center', va='center', fontsize=8, style='italic')

    # Flecha 1
    ax.annotate('', xy=(3.0, 2), xytext=(2.05, 2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Caja 2: TCN Etapa 1
    box2 = Rectangle((3.0, 1.3), 1.8, 1.4, linewidth=2,
                     edgecolor='darkgreen', facecolor='lightgreen', alpha=0.7)
    ax.add_patch(box2)
    ax.text(3.9, 2.3, 'TCN\nEtapa 1', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(3.9, 1.7, '5 bloques\n64 canales', ha='center', va='center',
            fontsize=8, style='italic')

    # Flecha 2
    ax.annotate('', xy=(5.8, 2), xytext=(4.85, 2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Caja 3: angulos articulares
    box3 = Rectangle((5.8, 1.3), 1.8, 1.4, linewidth=2,
                     edgecolor='orange', facecolor='moccasin', alpha=0.7)
    ax.add_patch(box3)
    ax.text(6.7, 2.3, 'Ángulos\nArticulares', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(6.7, 1.7, 'Cadera + Rodilla\n+ velocidades', ha='center', va='center',
            fontsize=8, style='italic')

    # Flecha 3
    ax.annotate('', xy=(8.6, 2), xytext=(7.65, 2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Caja 4: TCN Etapa 2 + salida
    box4 = Rectangle((8.6, 1.3), 1.3, 1.4, linewidth=2,
                     edgecolor='darkgreen', facecolor='lightgreen', alpha=0.7)
    ax.add_patch(box4)
    ax.text(9.25, 2.3, 'TCN\nEtapa 2', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(9.25, 1.7, '+ Momento\n(Nm/kg)', ha='center', va='center',
            fontsize=8, fontweight='bold', color='darkred')

    # Etiquetas de cada etapa
    ax.text(2.5, 3.4, 'ENTRADA', ha='center', fontsize=9, color='gray',
            fontweight='bold')
    ax.text(5.35, 3.4, 'ETAPA 1: IMU → Ángulos', ha='center', fontsize=9,
            color='gray', fontweight='bold')
    ax.text(8.5, 3.4, 'ETAPA 2: Ángulos → Momento', ha='center', fontsize=9,
            color='gray', fontweight='bold')

    # Pie de figura
    ax.text(5, 0.4,
            'Pipeline de dos etapas para estimación de momento articular durante la marcha',
            ha='center', fontsize=10, style='italic', color='#444444')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_pipeline.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig1_pipeline.pdf'),
                bbox_inches='tight')
    plt.close()
    print("Guardada fig1_pipeline")


# Figura 2: ejemplo cualitativo de la Etapa 1.
def fig_examples(data_dir, stage1_path, stage2_path, output_dir, device='cpu'):
    """
    Genera solo la figura cualitativa de la Etapa 1.

    Decision de diseno: los angulos de la Etapa 1 son suaves y periodicos, asi
    que una traza temporal de un ensayo se entiende bien de un vistazo. Los
    momentos, en cambio, los represento con figuras agregadas sobre todos los
    ensayos (fig3 y fig4), porque un ensayo suelto puede dar una impresion
    enganosa de lo bien o mal que va.
    """
    device = torch.device(device)
    stage1 = load_trained_model(stage1_path, device)
    eff_hist = stage1.get_effective_history()

    inputs = [
        'shank_imu_r_gyro_x', 'shank_imu_r_gyro_y', 'shank_imu_r_gyro_z',
        'shank_imu_r_accel_x', 'shank_imu_r_accel_y', 'shank_imu_r_accel_z',
        'thigh_imu_r_gyro_x', 'thigh_imu_r_gyro_y', 'thigh_imu_r_gyro_z',
        'thigh_imu_r_accel_x', 'thigh_imu_r_accel_y', 'thigh_imu_r_accel_z',
        'hip_angle_r', 'hip_angle_r_velocity_filt',
        'knee_angle_r', 'knee_angle_r_velocity_filt',
    ]
    labels = ['hip_flexion_r_moment', 'knee_angle_r_moment']
    dataset = CamargoTcnDataset(
        data_dir=data_dir,
        input_names=inputs,
        label_names=labels,
        device=device,
    )

    # Busco un ensayo largo de cinta para que el ejemplo de la Etapa 1 salga limpio.
    target_idx = None
    for i in range(len(dataset)):
        path = dataset.trial_paths[i]
        if 'treadmill' in os.path.basename(path).lower():
            inp, lab, sl = dataset[i]
            if sl[0] > 800:
                target_idx = i
                break

    if target_idx is None:
        target_idx = 0

    input_data, label_data, seq_lengths = dataset[target_idx]
    seq_len = seq_lengths[0]

    imu = input_data[:, :12, :]
    gt_ang = input_data[:, 12:, :]

    with torch.no_grad():
        pred_ang = stage1(imu)

    start = eff_hist + 10
    end = min(seq_len, start + 600)  # ~3 s de datos a 200 Hz
    t = np.arange(start, end) / 200.0

    # Figura 2: Etapa 1 (IMU -> angulos)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for j, title in enumerate(['Ángulo de cadera', 'Ángulo de rodilla']):
        ax = axes[j]
        gt = gt_ang[0, j * 2, start:end].cpu().numpy()
        pr = pred_ang[0, j, start:end].cpu().numpy()
        ax.plot(t, gt, 'b-', label='Referencia (OpenSim IK)', linewidth=1.5)
        ax.plot(t, pr, 'r--', label='Predicción TCN', linewidth=1.5)
        ax.set_ylabel(f'{title} (°)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Etapa 1: {title}', fontsize=11, loc='left')
    axes[-1].set_xlabel('Tiempo (s)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_stage1_example.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig2_stage1_example.pdf'),
                bbox_inches='tight')
    plt.close()
    print("Guardada fig2_stage1_example")


def _first_existing_column(df, candidates, required=True):
    """Devuelve la primera columna que exista de una lista de nombres posibles.
    Da margen a que el CSV use nombres ligeramente distintos sin romper."""
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            "Ninguna de estas columnas esta en el CSV de resultados: "
            + ", ".join(candidates)
            + "\nColumnas disponibles: "
            + ", ".join(df.columns)
        )
    return None


# Figura 3: distribucion del RMSE de momentos sobre todos los ensayos.
def fig3_moment_rmse_distribution(results_csv, output_dir):
    """
    En vez de la traza de momento de un solo ensayo, muestro la distribucion del
    RMSE sobre todos los ensayos. Es mas defendible: representa el conjunto
    entero, no un ejemplo elegido a ojo.
    """
    df = pd.read_csv(results_csv)

    col_s2_hip = _first_existing_column(df, [
        'stage2_hip_rmse_Nmkg',
        'stage2_hip_rmse',
        'stage2_hip_moment_rmse_Nmkg',
        'stage2_hip_moment_rmse',
    ])
    col_pipe_hip = _first_existing_column(df, [
        'pipeline_hip_rmse_Nmkg',
        'pipeline_hip_rmse',
        'full_pipeline_hip_rmse_Nmkg',
        'pipeline_hip_moment_rmse_Nmkg',
    ])
    col_s2_knee = _first_existing_column(df, [
        'stage2_knee_rmse_Nmkg',
        'stage2_knee_rmse',
        'stage2_knee_moment_rmse_Nmkg',
        'stage2_knee_moment_rmse',
    ])
    col_pipe_knee = _first_existing_column(df, [
        'pipeline_knee_rmse_Nmkg',
        'pipeline_knee_rmse',
        'full_pipeline_knee_rmse_Nmkg',
        'pipeline_knee_moment_rmse_Nmkg',
    ])

    data = [
        df[col_s2_hip].dropna().values,
        df[col_pipe_hip].dropna().values,
        df[col_s2_knee].dropna().values,
        df[col_pipe_knee].dropna().values,
    ]
    labels = [
        'Etapa 2\nCadera',
        'Sistema\nCadera',
        'Etapa 2\nRodilla',
        'Sistema\nRodilla',
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)

    # Colores neutros, solo para separar visualmente los grupos.
    colors = ['#d9eaf7', '#bcd7ee', '#f7e2c6', '#edcfa4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    ax.set_ylabel('RMSE del momento articular (Nm/kg)', fontsize=11)
    ax.set_title('Distribución del error por ensayo en la estimación de momentos',
                 fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Medias como puntos negros
    means = [np.nanmean(x) for x in data]
    ax.scatter(np.arange(1, len(data) + 1), means, color='black',
               marker='o', s=30, zorder=3, label='Media')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_moment_rmse_distribution.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig3_moment_rmse_distribution.pdf'),
                bbox_inches='tight')
    # Guardo tambien con el nombre antiguo (fig3_stage2_example) para que
    # Overleaf actualice aunque el LaTeX aun apunte a ese fichero.
    plt.savefig(os.path.join(output_dir, 'fig3_stage2_example.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig3_stage2_example.pdf'),
                bbox_inches='tight')
    plt.close()

    print("Guardada fig3_moment_rmse_distribution")
    print("Sobrescrita tambien fig3_stage2_example con la version agregada")


# Figura 4: Etapa 2 (angulos reales) frente al sistema completo (angulos predichos).
def fig4_stage2_vs_pipeline_rmse(results_csv, output_dir):
    """
    Compara la Etapa 2 con angulos reales contra el sistema completo con angulos
    predichos por la Etapa 1. Los puntos cerca de la diagonal indican que el
    error de la Etapa 1 apenas se propaga a la estimacion de momentos.
    """
    df = pd.read_csv(results_csv)

    col_s2_hip = _first_existing_column(df, [
        'stage2_hip_rmse_Nmkg',
        'stage2_hip_rmse',
        'stage2_hip_moment_rmse_Nmkg',
        'stage2_hip_moment_rmse',
    ])
    col_pipe_hip = _first_existing_column(df, [
        'pipeline_hip_rmse_Nmkg',
        'pipeline_hip_rmse',
        'full_pipeline_hip_rmse_Nmkg',
        'pipeline_hip_moment_rmse_Nmkg',
    ])
    col_s2_knee = _first_existing_column(df, [
        'stage2_knee_rmse_Nmkg',
        'stage2_knee_rmse',
        'stage2_knee_moment_rmse_Nmkg',
        'stage2_knee_moment_rmse',
    ])
    col_pipe_knee = _first_existing_column(df, [
        'pipeline_knee_rmse_Nmkg',
        'pipeline_knee_rmse',
        'full_pipeline_knee_rmse_Nmkg',
        'pipeline_knee_moment_rmse_Nmkg',
    ])

    pairs = [
        ('Cadera', col_s2_hip, col_pipe_hip),
        ('Rodilla', col_s2_knee, col_pipe_knee),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)

    for ax, (title, col_s2, col_pipe) in zip(axes, pairs):
        x = df[col_s2].astype(float)
        y = df[col_pipe].astype(float)
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]

        ax.scatter(x, y, alpha=0.55, s=18, edgecolor='none')
        max_val = max(float(x.max()), float(y.max())) * 1.05
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1,
                label='Sin aumento de error')
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel('Etapa 2 con ángulos reales (RMSE, Nm/kg)', fontsize=10)
        ax.set_ylabel('Sistema completo (RMSE, Nm/kg)', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

        delta = (y - x).mean()
        ax.text(0.98, 0.05, f'Δ medio = {delta:.3f} Nm/kg',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8))

    plt.suptitle('Propagación del error entre Etapa 2 y sistema completo',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_stage2_vs_pipeline_rmse.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig4_stage2_vs_pipeline_rmse.pdf'),
                bbox_inches='tight')
    # Nombre antiguo (fig4_pipeline_example) para que Overleaf actualice
    # aunque el LaTeX aun lo referencie.
    plt.savefig(os.path.join(output_dir, 'fig4_pipeline_example.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig4_pipeline_example.pdf'),
                bbox_inches='tight')
    plt.close()

    print("Guardada fig4_stage2_vs_pipeline_rmse")
    print("Sobrescrita tambien fig4_pipeline_example")


# Figura 5: boxplots del RMSE por modo de marcha.
def fig5_per_mode(results_csv, output_dir):
    """
    Boxplots por modo de marcha. Oculto los outliers extremos en esta figura
    porque los analizo aparte; asi la parte central de la distribucion se compara
    mejor entre suelo y cinta. (showfliers=False solo afecta al dibujo, no a las
    metricas, que se calculan con todos los datos.)
    """
    df = pd.read_csv(results_csv)

    mode_order = [m for m in ['levelground', 'treadmill'] if m in df['mode'].unique()]
    if not mode_order:
        mode_order = sorted(df['mode'].dropna().unique())

    mode_labels = {
        'levelground': 'Suelo nivelado',
        'treadmill': 'Cinta',
    }

    metrics = [
        ('Etapa 1 - Cadera (°)', 'stage1_hip_rmse_deg', 'RMSE (°)'),
        ('Etapa 1 - Rodilla (°)', 'stage1_knee_rmse_deg', 'RMSE (°)'),
        ('Sistema - Cadera (Nm/kg)', 'pipeline_hip_rmse_Nmkg', 'RMSE (Nm/kg)'),
        ('Sistema - Rodilla (Nm/kg)', 'pipeline_knee_rmse_Nmkg', 'RMSE (Nm/kg)'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))

    for ax, (title, col, ylabel) in zip(axes, metrics):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {results_csv}")

        data = [
            df[df['mode'] == mode][col].dropna().astype(float).values
            for mode in mode_order
        ]

        bp = ax.boxplot(
            data,
            labels=[mode_labels.get(m, m.capitalize()) for m in mode_order],
            patch_artist=True,
            showfliers=False,  # oculta outliers solo en el dibujo, no en los calculos
            medianprops=dict(linewidth=1.5),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
        )

        colors = ['#d9eaf7', '#c8edd6']
        for patch, color in zip(bp['boxes'], colors[:len(mode_order)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)

        # Media como punto negro.
        means = [np.nanmean(vals) if len(vals) else np.nan for vals in data]
        ax.scatter(
            np.arange(1, len(mode_order) + 1),
            means,
            color='black',
            marker='o',
            s=24,
            zorder=3,
            label='Media' if ax is axes[0] else None,
        )

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

    axes[0].legend(loc='upper right', fontsize=8)
    plt.suptitle('Distribución del error por modo de locomoción', fontsize=12, y=1.03)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'fig5_per_mode_bars.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig5_per_mode_bars.pdf'),
                bbox_inches='tight')
    # Guardo tambien con un nombre mas claro por si actualizo el LaTeX.
    plt.savefig(os.path.join(output_dir, 'fig5_per_mode_boxplots.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig5_per_mode_boxplots.pdf'),
                bbox_inches='tight')
    plt.close()
    print("Guardada fig5_per_mode_boxplots")
    print("Sobrescrita tambien fig5_per_mode_bars con la version de boxplots")


# Figura 6: dispersion del error por sujeto (resalta los no vistos en rojo).
def fig6_subject_box(results_csv, output_dir):
    df = pd.read_csv(results_csv)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Angulo de cadera
    ax = axes[0]
    subjects = sorted(df['subject'].unique())
    data = [df[df['subject'] == s]['stage1_hip_rmse_deg'].dropna().values
            for s in subjects]
    bp = ax.boxplot(data, labels=subjects, patch_artist=True, showfliers=True)
    for patch, subj in zip(bp['boxes'], subjects):
        is_heldout = bool(df[df['subject'] == subj]['heldout'].iloc[0])
        patch.set_facecolor('#e74c3c' if is_heldout else '#3498db')
        patch.set_alpha(0.6)
    ax.set_ylim(0, 15)
    ax.set_ylabel('RMSE ángulo cadera (°)', fontsize=11)
    ax.set_title('Distribución del error de Etapa 1 por sujeto', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Momento de cadera del sistema completo
    ax = axes[1]
    data = [df[df['subject'] == s]['pipeline_hip_rmse_Nmkg'].dropna().values
            for s in subjects]
    bp = ax.boxplot(data, labels=subjects, patch_artist=True, showfliers=True)
    for patch, subj in zip(bp['boxes'], subjects):
        is_heldout = bool(df[df['subject'] == subj]['heldout'].iloc[0])
        patch.set_facecolor('#e74c3c' if is_heldout else '#3498db')
        patch.set_alpha(0.6)
    ax.set_ylabel('RMSE momento cadera pipeline (Nm/kg)', fontsize=11)
    ax.set_xlabel('Sujeto', fontsize=11)
    ax.set_title('Distribución del error del pipeline completo por sujeto', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.6, label='Sujetos de entrenamiento'),
        Patch(facecolor='#e74c3c', alpha=0.6, label='Sujetos held-out'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_subject_box.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig6_subject_box.pdf'),
                bbox_inches='tight')
    plt.close()
    print("Guardada fig6_subject_box")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--stage1_model", type=str, required=True)
    parser.add_argument("--stage2_model", type=str, required=True)
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fig1_pipeline(args.output_dir)
    fig_examples(args.data_dir, args.stage1_model, args.stage2_model,
                 args.output_dir, args.device)
    fig3_moment_rmse_distribution(args.results_csv, args.output_dir)
    fig4_stage2_vs_pipeline_rmse(args.results_csv, args.output_dir)
    fig5_per_mode(args.results_csv, args.output_dir)
    fig6_subject_box(args.results_csv, args.output_dir)

    print("\nTodas las figuras guardadas en:", args.output_dir)
