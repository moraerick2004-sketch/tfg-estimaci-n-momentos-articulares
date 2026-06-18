#!/usr/bin/env python3
"""
Genera dos figuras para el estado del arte (Seccion 2.1.1):

  - cinematica_tipica.pdf : angulos de cadera y rodilla (plano sagital)
  - cinetica_tipica.pdf   : momentos de cadera y rodilla (Nm/kg)

Ambas como media +/- desviacion tipica a lo largo del ciclo de marcha
(0-100%), sobre los ensayos de marcha en CINTA del conjunto de Camargo et al.
ya convertidos al formato del proyecto (Exo.csv + Joint_Moments_Filt.csv).

Requisitos: numpy, pandas, scipy, matplotlib
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ======================= CONFIGURACION =======================
# Ruta relativa por defecto; editable aqui o con la variable de entorno DATA_ROOT.
DATA_ROOT  = os.environ.get("DATA_ROOT", "./data_molinaro_nan")  # <sujeto>/<ensayo>/Exo.csv
OUT_DIR    = "img"
FS         = 200.0        # Hz
N_POINTS   = 101          # muestras por ciclo normalizado (0-100%)
SOLO_CINTA = True         # solo treadmill: ciclos estables y sin NaN en el momento
METODO_HS  = "impacto"    # "impacto" (IMU pantorrilla, recomendado) o "cadera" (heuristica antigua)
# =============================================================

os.makedirs(OUT_DIR, exist_ok=True)


def detectar_contactos(exo, fs=FS, metodo=METODO_HS):
    """Detecta los contactos iniciales (heel strikes).

    "impacto": pico del transitorio de alta frecuencia del modulo del
    acelerometro de la pantorrilla. Es un evento nitido y consistente,
    asi que al promediar zancadas el resultado sale mucho mas limpio que
    con el pico (ancho) de flexion de cadera.

    Si tu conjunto convertido conserva los eventos REALES de marcha de
    Camargo, sustituye esta funcion por su lectura: seria lo ideal.
    """
    dist_min = int(0.7 * fs)  # zancada minima ~0.7 s
    cols_acc = ["shank_imu_r_accel_x", "shank_imu_r_accel_y", "shank_imu_r_accel_z"]
    if metodo == "impacto" and all(c in exo.columns for c in cols_acc):
        a = np.sqrt(sum(exo[c].to_numpy(dtype=float) ** 2 for c in cols_acc))
        w = max(int(0.10 * fs), 3)
        suave = np.convolve(a, np.ones(w) / w, mode="same")
        impacto = a - suave                       # resalta el golpe del contacto
        umbral = np.percentile(impacto, 90)
        picos, _ = find_peaks(impacto, distance=dist_min, height=umbral)
        if len(picos) >= 5:
            return picos
    # fallback: maximo de flexion de cadera
    picos, _ = find_peaks(exo["hip_angle_r"].to_numpy(dtype=float), distance=dist_min)
    return picos

def refinar_a_cruce_momento(hip_m, contactos, fs=FS, ventana=0.15):
    """Reajusta cada contacto al cruce por cero ascendente del momento
    de cadera mas cercano (marca el inicio del apoyo con mas precision
    que el impacto del IMU). ventana en segundos."""
    w = int(ventana * fs)
    refinados = []
    for c in contactos:
        a, b = max(0, c - w), min(len(hip_m) - 1, c + w)
        seg = hip_m[a:b]
        cruces = np.where((seg[:-1] <= 0) & (seg[1:] > 0))[0]
        refinados.append(a + cruces[0] if len(cruces) else c)
    return np.array(sorted(set(refinados)))

    
def normalizar_ciclo(signal, ini, fin, n=N_POINTS):
    seg = np.asarray(signal[ini:fin], dtype=float)
    if len(seg) < 10 or np.any(~np.isfinite(seg)):
        return None
    x_old = np.linspace(0, 100, len(seg))
    x_new = np.linspace(0, 100, n)
    return np.interp(x_new, x_old, seg)


ciclos = {"hip_angle": [], "knee_angle": [], "hip_moment": [], "knee_moment": []}

ficheros = glob.glob(os.path.join(DATA_ROOT, "*", "*", "Exo.csv"))
print(f"Encontrados {len(ficheros)} ensayos en {DATA_ROOT}")

for exo_path in ficheros:
    trial_dir = os.path.dirname(exo_path)
    if SOLO_CINTA and "treadmill" not in trial_dir.lower():
        continue
    mom_path = os.path.join(trial_dir, "Joint_Moments_Filt.csv")
    if not os.path.exists(mom_path):
        continue
    try:
        exo = pd.read_csv(exo_path)
        mom = pd.read_csv(mom_path)
    except Exception:
        continue
    if "hip_angle_r" not in exo.columns or "knee_angle_r" not in exo.columns:
        continue

    n = min(len(exo), len(mom))
    hip_a  = exo["hip_angle_r"].to_numpy()[:n]
    knee_a = exo["knee_angle_r"].to_numpy()[:n]
    hip_m  = mom["hip_flexion_r_moment"].to_numpy()[:n]
    knee_m = mom["knee_angle_r_moment"].to_numpy()[:n]

    contactos = detectar_contactos(exo)
    contactos = refinar_a_cruce_momento(hip_m, contactos)
    contactos = contactos[contactos < n]
    for i in range(len(contactos) - 1):
        ini, fin = contactos[i], contactos[i + 1]
        for clave, sig in [("hip_angle", hip_a), ("knee_angle", knee_a),
                           ("hip_moment", hip_m), ("knee_moment", knee_m)]:
            c = normalizar_ciclo(sig, ini, fin)
            if c is not None:
                ciclos[clave].append(c)

x = np.linspace(0, 100, N_POINTS)
stats = {}
for clave, lista in ciclos.items():
    if not lista:
        raise RuntimeError(f"No se han recogido ciclos para '{clave}'. "
                           f"Revisa DATA_ROOT y los nombres de columna.")
    arr = np.array(lista)
    stats[clave] = (arr.mean(axis=0), arr.std(axis=0), len(arr))

# ---------- DIAGNOSTICO DE ALINEAMIENTO ----------
m_knee = stats["knee_angle"][0]
m_hip  = stats["hip_angle"][0]
print("\n--- Diagnostico (metodo de contacto: %s) ---" % METODO_HS)
print("Ciclos usados:", {k: v[2] for k, v in stats.items()})
print(f"Rodilla en contacto inicial (0%): {m_knee[0]:6.1f} grados   (esperado ~ -5 a -12)")
print(f"ROM rodilla (max-min):            {m_knee.max() - m_knee.min():6.1f} grados   (esperado ~ 55-65)")
print(f"ROM cadera  (max-min):            {m_hip.max() - m_hip.min():6.1f} grados   (esperado ~ 40-45)")
pico_mom = max(np.abs(stats["hip_moment"][0]).max(), np.abs(stats["knee_moment"][0]).max())
print(f"Pico de momento:                  {pico_mom:6.2f}        (si fuese ~decenas, esta en Nm: divide por la masa)")
print("-------------------------------------------\n")


def figura(claves_titulos, ylabel, color, fichero):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (clave, titulo) in zip(axes, claves_titulos):
        m, s, _ = stats[clave]
        ax.plot(x, m, color=color, lw=2)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)
        ax.set_title(titulo)
        ax.set_xlabel("Ciclo de marcha (%)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(OUT_DIR, fichero)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", ruta)


figura([("hip_angle", "Cadera"), ("knee_angle", "Rodilla")],
       "Angulo (grados)", "#1f4e79", "cinematica_tipica.pdf")
figura([("hip_moment", "Cadera"), ("knee_moment", "Rodilla")],
       "Momento (Nm/kg)", "#7d2e2e", "cinetica_tipica.pdf")
