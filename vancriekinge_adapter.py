"""
vancriekinge_adapter.py

Adapta los C3D de Van Criekinge et al. (2023) al formato de entrada de la
Etapa 2 para evaluar la generalizacion zero-shot a adultos mayores. Solo se
usa la pierna derecha.

Camargo (datos de entrenamiento) esta en convencion OpenSim y Van Criekinge en
Plug-in Gait, asi que hay que alinear signo y unidades. Verifique cada signo
superponiendo un ciclo contra la media de Camargo (no me fie de la literatura,
cada laboratorio define los ejes distinto):

  hip_angle_r          =  RHipAngles            misma convencion, flexion +
  knee_angle_r         = -RKneeAngles           PiG flexion +, OpenSim flexion -
  hip_flexion_r_moment = -RHipMoment / 1000     signo opuesto a OpenSim
  knee_angle_r_moment  =  RKneeMoment / 1000    mismo signo, NO se invierte

Los momentos del C3D ya vienen normalizados por masa, por eso solo /1000
(paso de Nmm/kg a Nm/kg) y nunca se divide otra vez por la masa.

El momento solo es valido cuando el pie derecho pisa una plataforma de fuerza
limpia. Fuera de esas ventanas se escribe NaN aunque el valor numerico sea
finito: un valor finito sin contacto debajo no es un momento medido. La perdida
y la evaluacion enmascaran esos NaN con torch.isfinite(). Los angulos, en
cambio, vienen de marcadores y son validos en todo el ensayo.

El remuestreo 100 -> 200 Hz se hace ANTES de derivar, porque el campo receptivo
de la TCN esta definido a 200 Hz. La velocidad angular se calcula sobre el
angulo ya convertido para que el signo de la velocidad herede el del angulo.

Requisitos: pip install ezc3d numpy pandas scipy
Necesita en la misma carpeta: older_adults_cohort.py, assign_right_contacts.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from ezc3d import c3d

from older_adults_cohort import cohort, SUBJECT_MASS_KG
from assign_right_contacts import right_contact_windows

# ----------------------------- configuracion -------------------------------
# Rutas por defecto, relativas a la carpeta del proyecto. Se pueden cambiar al
# ejecutar con --base_dir y --out_dir (ver __main__).
BASE_DIR = "./138_HealthyPiG_10.05"      # C3D originales de Van Criekinge
OUT_DIR  = "./data_vancriekinge"         # CSV adaptados de salida
MIN_AGE  = 61          # '>60' del tutor  (usa 60 para incluir los de 60 justos)
RATE_OUT = 200         # Hz objetivo (campo receptivo de la TCN definido a 200 Hz)

# Nombres de las columnas de velocidad (las que lee tu CamargoTcnDataset).
HIP_VEL_COL  = "hip_angle_r_velocity_filt"
KNEE_VEL_COL = "knee_angle_r_velocity_filt"
# ---------------------------------------------------------------------------


def moving_average(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="same")


def fill_nan(y):
    # Interpola huecos solo para poder remuestrear sin que el spline explote.
    # NO afecta a los momentos finales: la mascara de validez se aplica despues.
    y = np.asarray(y, dtype=float).copy()
    bad = ~np.isfinite(y)
    if bad.any() and not bad.all():
        i = np.arange(y.size)
        y[bad] = np.interp(i[bad], i[~bad], y[~bad])
    return y


def resample_200(y, rate_in):
    n = y.size
    t_old = np.arange(n) / rate_in
    t_new = np.arange(0, t_old[-1] + 1e-9, 1.0 / RATE_OUT)
    yf = fill_nan(y)
    try:
        from scipy.interpolate import CubicSpline
        return t_new, CubicSpline(t_old, yf)(t_new)
    except Exception:
        return t_new, np.interp(t_new, t_old, yf)


def convert_trial(path, mass):
    c = c3d(path)
    rate = c["parameters"]["POINT"]["RATE"]["value"][0]
    L = c["parameters"]["POINT"]["LABELS"]["value"]
    idx = {n: i for i, n in enumerate(L)}
    need = ["RHipAngles", "RKneeAngles", "RHipMoment", "RKneeMoment"]
    if any(k not in idx for k in need):
        return None
    P = c["data"]["points"]

    # Componente X = plano sagital. Signos y unidades segun la cabecera.
    hip_ang  =  P[0, idx["RHipAngles"], :]
    knee_ang = -P[0, idx["RKneeAngles"], :]
    hip_mom  = -P[0, idx["RHipMoment"], :] / 1000.0
    knee_mom = P[0, idx["RKneeMoment"], :] / 1000.0

    # --- ventanas validas del pie derecho (apoyo limpio sobre plataforma) ---
    wins = right_contact_windows(path, mass, verbose=False)
    if not wins:
        return None

    # --- remuestreo a 200 Hz ---
    t_new, hip_ang2  = resample_200(hip_ang, rate)
    _,     knee_ang2 = resample_200(knee_ang, rate)
    _,     hip_mom2  = resample_200(hip_mom, rate)
    _,     knee_mom2 = resample_200(knee_mom, rate)

    # --- velocidad angular sobre el angulo convertido (deg/s) ---
    hip_vel  = moving_average(np.gradient(hip_ang2,  1.0 / RATE_OUT), 5)
    knee_vel = moving_average(np.gradient(knee_ang2, 1.0 / RATE_OUT), 5)

    # --- mascara: momento valido SOLO dentro de las ventanas; resto NaN ---
    valid = np.zeros_like(t_new, dtype=bool)
    for t0, t1 in wins:
        valid |= (t_new >= t0) & (t_new <= t1)
    hip_mom2  = np.where(valid, hip_mom2, np.nan)
    knee_mom2 = np.where(valid, knee_mom2, np.nan)

    return pd.DataFrame({
        "hip_angle_r": hip_ang2,
        "knee_angle_r": knee_ang2,
        HIP_VEL_COL: hip_vel,
        KNEE_VEL_COL: knee_vel,
        "hip_flexion_r_moment": hip_mom2,
        "knee_angle_r_moment": knee_mom2,
    })


def resolve_subject_dir(base, sid):
    """Localiza la carpeta del sujeto probando 'SUBJ1' y 'SUBJ01' (cero inicial).
    Devuelve (Path, nombre_en_disco) o (None, None) si no existe ninguna."""
    num = int(sid[len("SUBJ"):])
    for name in (sid, f"SUBJ{num:02d}"):
        d = base / name
        if d.exists():
            return d, name
    return None, None


def main(base_dir=BASE_DIR, out_dir=OUT_DIR, min_age=MIN_AGE):
    out = Path(out_dir)
    base = Path(base_dir)
    ids = cohort(min_age)
    print(f"cohorte (edad >= {min_age}): {len(ids)} sujetos")
    n_ok = n_skip = n_valid_frames = 0
    for sid in ids:
        mass = SUBJECT_MASS_KG[sid]            # masa con el ID original de cohort()
        sdir, folder = resolve_subject_dir(base, sid)
        if sdir is None:
            print(f"  AVISO: falta carpeta {sid} (ni '{sid}' ni con cero inicial)")
            continue
        for cpath in sorted(sdir.glob("*.c3d")):   # no asume prefijo = nombre carpeta
            try:
                df = convert_trial(str(cpath), mass)
            except Exception as e:
                print(f"  error {cpath.name}: {e}")
                df = None
            if df is None:
                n_skip += 1
                continue
            (out / sid).mkdir(parents=True, exist_ok=True)
            df.to_csv(out / sid / (cpath.stem.replace(" ", "_") + ".csv"),
                      index=False)
            n_ok += 1
            n_valid_frames += int(np.isfinite(df["hip_flexion_r_moment"]).sum())
    print(f"\ntrials escritos: {n_ok} | descartados (sin ventana): {n_skip}")
    print(f"muestras con momento valido (no NaN): {n_valid_frames}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_dir", default=BASE_DIR,
                    help="carpeta con los C3D originales de Van Criekinge")
    ap.add_argument("--out_dir", default=OUT_DIR,
                    help="carpeta donde escribir los CSV adaptados")
    ap.add_argument("--min_age", type=int, default=MIN_AGE,
                    help="edad minima de la cohorte (61 = '>60')")
    args = ap.parse_args()
    main(args.base_dir, args.out_dir, args.min_age)
