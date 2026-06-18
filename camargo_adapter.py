"""
camargo_adapter.py

Convierte los CSV de Camargo et al. (2021) al formato que consume el
dataloader, mapeando nombres de columnas IMU/IK/ID y calculando las
velocidades angulares.

El punto clave es que PRESERVA los NaN de los momentos en vez de rellenarlos
con cero. Esos NaN no son ruido: aparecen donde no hay contacto sobre
plataforma de fuerza, asi que el momento no esta medido. Rellenarlos con cero
(como se hacia antes) mete discontinuidades artificiales que el modelo aprende
como si fueran reales. La perdida los enmascara con torch.isfinite(), pero eso
solo funciona si llegan como NaN hasta aqui; por eso el adaptador no los toca.

El modo antiguo (rellenar con cero) sigue disponible con --legacy_fill_zeros,
unicamente para reproducir los resultados previos a la correccion y poder
compararlos.

Uso:
    python3 camargo_adapter.py \\
        --data_dir /ruta/data_converted \\
        --output_dir /ruta/data_molinaro_nan
"""

import os
import argparse
import numpy as np
import pandas as pd


# Mapeo de nombres de columna: Camargo (origen) -> formato del dataloader.

IMU_MAPPING = {
    'thigh_Accel_X': 'thigh_imu_r_accel_x',
    'thigh_Accel_Y': 'thigh_imu_r_accel_y',
    'thigh_Accel_Z': 'thigh_imu_r_accel_z',
    'thigh_Gyro_X':  'thigh_imu_r_gyro_x',
    'thigh_Gyro_Y':  'thigh_imu_r_gyro_y',
    'thigh_Gyro_Z':  'thigh_imu_r_gyro_z',
    'shank_Accel_X': 'shank_imu_r_accel_x',
    'shank_Accel_Y': 'shank_imu_r_accel_y',
    'shank_Accel_Z': 'shank_imu_r_accel_z',
    'shank_Gyro_X':  'shank_imu_r_gyro_x',
    'shank_Gyro_Y':  'shank_imu_r_gyro_y',
    'shank_Gyro_Z':  'shank_imu_r_gyro_z',
    'foot_Accel_X':  'foot_imu_r_accel_x',
    'foot_Accel_Y':  'foot_imu_r_accel_y',
    'foot_Accel_Z':  'foot_imu_r_accel_z',
    'foot_Gyro_X':   'foot_imu_r_gyro_x',
    'foot_Gyro_Y':   'foot_imu_r_gyro_y',
    'foot_Gyro_Z':   'foot_imu_r_gyro_z',
}

IK_MAPPING = {
    'hip_flexion_r': 'hip_angle_r',
    'knee_angle_r':  'knee_angle_r',
}

ID_MAPPING = {
    'hip_flexion_r_moment': 'hip_flexion_r_moment',
    'knee_angle_r_moment':  'knee_angle_r_moment',
}

SAMPLING_RATE = 200.0


def compute_filtered_velocity(signal, dt, window=5):
    # Diferencias centrales + media movil. Si la entrada tiene NaN, np.gradient
    # los propaga, que es lo que queremos: no inventamos velocidad donde no hay
    # angulo.
    velocity = np.gradient(signal, dt)
    kernel = np.ones(window) / window
    velocity_filt = np.convolve(velocity, kernel, mode='same')
    return velocity_filt


def convert_trial(trial_dir, output_dir, legacy_fill_zeros=False):
    """
    Convierte un ensayo. Devuelve dict con estadisticas de NaN o None si falla.
    """
    imu_path = os.path.join(trial_dir, 'imu.csv')
    ik_path = os.path.join(trial_dir, 'ik.csv')
    id_path = os.path.join(trial_dir, 'id.csv')

    if not (os.path.exists(imu_path) and os.path.exists(ik_path)
            and os.path.exists(id_path)):
        return None

    try:
        imu_df = pd.read_csv(imu_path)
        ik_df = pd.read_csv(ik_path)
        id_df = pd.read_csv(id_path)
    except Exception as e:
        print(f"  ERROR leyendo {trial_dir}: {e}")
        return None

    # --- Exo.csv (entradas) ---
    exo = pd.DataFrame()
    for c_col, m_col in IMU_MAPPING.items():
        if c_col in imu_df.columns:
            exo[m_col] = imu_df[c_col].values

    for c_col, m_col in IK_MAPPING.items():
        if c_col in ik_df.columns:
            n = min(len(exo), len(ik_df))
            exo = exo.iloc[:n].copy()
            exo[m_col] = ik_df[c_col].values[:n]

    dt = 1.0 / SAMPLING_RATE
    if 'hip_angle_r' in exo.columns:
        exo['hip_angle_r_velocity_filt'] = compute_filtered_velocity(
            exo['hip_angle_r'].values, dt)
    if 'knee_angle_r' in exo.columns:
        exo['knee_angle_r_velocity_filt'] = compute_filtered_velocity(
            exo['knee_angle_r'].values, dt)

    # --- Joint_Moments_Filt.csv (etiquetas) ---
    moments = pd.DataFrame()
    for c_col, m_col in ID_MAPPING.items():
        if c_col in id_df.columns:
            moments[m_col] = id_df[c_col].values

    if legacy_fill_zeros:
        # Modo legacy: rellena los NaN con cero. Reintroduce las
        # discontinuidades que la version actual evita; solo para comparar
        # con resultados antiguos.
        moments = moments.fillna(0.0)
    # En modo normal los NaN se dejan intactos para que la perdida los enmascare.

    # Recortar a longitud comun
    n = min(len(exo), len(moments))
    if n < 50:
        return None
    exo = exo.iloc[:n]
    moments = moments.iloc[:n]

    # Verificacion de columnas requeridas
    required = (['shank_imu_r_gyro_x', 'thigh_imu_r_gyro_x',
                 'hip_angle_r', 'knee_angle_r'])
    if not all(c in exo.columns for c in required):
        return None
    required_m = ['hip_flexion_r_moment', 'knee_angle_r_moment']
    if not all(c in moments.columns for c in required_m):
        return None

    # Estadisticas de NaN ANTES de guardar
    stats = {
        'n_samples': n,
        'n_nan_hip': int(moments['hip_flexion_r_moment'].isna().sum()),
        'n_nan_knee': int(moments['knee_angle_r_moment'].isna().sum()),
    }
    stats['pct_nan_hip'] = 100.0 * stats['n_nan_hip'] / n
    stats['pct_nan_knee'] = 100.0 * stats['n_nan_knee'] / n

    # na_rep='' fuerza que los NaN se escriban como celda vacia, para que al
    # releer el CSV vuelvan a interpretarse como NaN y no como texto.
    os.makedirs(output_dir, exist_ok=True)
    exo.to_csv(os.path.join(output_dir, 'Exo.csv'), index=False)
    moments.to_csv(os.path.join(output_dir, 'Joint_Moments_Filt.csv'),
                   index=False, na_rep='')

    # Round-trip de seguridad: releer y confirmar que los NaN sobreviven al CSV.
    moments_back = pd.read_csv(os.path.join(output_dir, 'Joint_Moments_Filt.csv'))
    n_nan_after = int(moments_back['hip_flexion_r_moment'].isna().sum())
    if not legacy_fill_zeros and n_nan_after != stats['n_nan_hip']:
        print(f"  AVISO: NaN no preservados correctamente en {output_dir}")
        print(f"  (esperado {stats['n_nan_hip']}, recuperado {n_nan_after})")

    return stats


def get_mode(trial_name):
    name = trial_name.lower()
    for mode in ['treadmill', 'levelground', 'stair', 'ramp']:
        if name.startswith(mode):
            return mode
    return 'unknown'


def convert_all(data_dir, output_dir, modes=None, legacy_fill_zeros=False):
    if modes is None:
        modes = ['treadmill', 'levelground', 'stair', 'ramp']

    all_stats = []
    converted = 0
    skipped = 0

    subjects = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))
                       and d.startswith('AB')])

    if not subjects:
        print(f"ERROR: No se encontraron sujetos AB* en {data_dir}")
        return

    print(f"Sujetos encontrados: {len(subjects)}")
    print(f"Modos filtrados: {modes}")
    print(f"Modo legacy (rellenar con cero): {legacy_fill_zeros}")
    print(f"Salida: {output_dir}")
    print("-" * 60)

    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        trials = sorted([t for t in os.listdir(subj_dir)
                         if os.path.isdir(os.path.join(subj_dir, t))])
        subj_converted = 0

        for trial_name in trials:
            mode = get_mode(trial_name)
            if mode not in modes:
                skipped += 1
                continue

            trial_dir = os.path.join(subj_dir, trial_name)
            out_dir = os.path.join(output_dir, subj, trial_name)
            stats = convert_trial(trial_dir, out_dir, legacy_fill_zeros)
            if stats is not None:
                stats['subject'] = subj
                stats['trial'] = trial_name
                stats['mode'] = mode
                all_stats.append(stats)
                converted += 1
                subj_converted += 1
            else:
                skipped += 1

        print(f"  {subj}: {subj_converted} ensayos convertidos")

    print("-" * 60)
    print(f"TOTAL: {converted} ensayos convertidos, {skipped} ignorados")

    # Resumen de NaN por modo: sirve para confirmar de un vistazo que la
    # preservacion ha funcionado (un 0% global delataria un relleno indebido).
    if all_stats:
        df = pd.DataFrame(all_stats)
        print("\n" + "=" * 70)
        print("ESTADISTICAS DE NaN EN MOMENTOS GUARDADOS")
        print("=" * 70)

        if legacy_fill_zeros:
            print("MODO LEGACY ACTIVO: los NaN se han sustituido por ceros.")
            print("Las estadisticas siguientes reflejan el numero de muestras")
            print("que ANTES eran NaN (y ahora son cero).")
        else:
            print("MODO CORRECTO: los NaN se han preservado.")

        print("")
        print(f"{'Modo':<13} {'N':>5} "
              f"{'%NaN hip (medio)':>18} {'%NaN knee (medio)':>19}")
        print("-" * 60)
        for mode in sorted(df['mode'].unique()):
            sub = df[df['mode'] == mode]
            print(f"{mode:<13} {len(sub):>5} "
                  f"{sub['pct_nan_hip'].mean():>17.1f}% "
                  f"{sub['pct_nan_knee'].mean():>18.1f}%")

        print("")
        global_hip = df['pct_nan_hip'].mean()
        global_knee = df['pct_nan_knee'].mean()
        print(f"GLOBAL: {global_hip:.1f}% NaN en hip, {global_knee:.1f}% en knee")

        # Guardar CSV con estadisticas
        stats_path = os.path.join(output_dir, '_nan_stats.csv')
        df.to_csv(stats_path, index=False)
        print(f"\nEstadisticas detalladas guardadas en: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--modes", type=str, nargs='+',
                        default=['treadmill', 'levelground', 'stair', 'ramp'])
    parser.add_argument("--legacy_fill_zeros", action='store_true',
                        help="SOLO PARA REPRODUCIR RESULTADOS ANTIGUOS: "
                             "rellena NaN con cero (metodologicamente incorrecto, "
                             "ver feedback de David).")
    args = parser.parse_args()
    convert_all(args.data_dir, args.output_dir, args.modes,
                args.legacy_fill_zeros)
