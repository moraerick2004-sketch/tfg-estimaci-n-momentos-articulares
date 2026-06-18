#!/usr/bin/env python3
"""
exportar_csv_vancriekinge.py

Reutiliza la logica de evaluate_stage2_vancriekinge.py para volcar a CSV lo que
necesitan las figuras de cap4:

  - vc_per_subject.csv : una fila por sujeto (rmse, r, r2, amplitudes) cadera+rodilla
  - vc_overlay.csv     : curvas medias 0-100% (real y pred) del sujeto representativo

Ejecutar desde la carpeta del proyecto:
    python exportar_csv_vancriekinge.py [ruta_stage2.tar]
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch

# Reutiliza tal cual el cargador y el predictor de tu evaluador (misma logica):
from evaluate_stage2_vancriekinge import (
    load_stage2, predict_trial, DATA_DIR, STAGE2_MODEL, OUT_COLS
)

N_POINTS = 101  # muestras por ciclo normalizado (0-100%)


def amplitud(x):
    """Amplitud robusta: rango entre percentiles 5 y 95 (Nm/kg).
    Evita que los transitorios de borde de plataforma, frecuentes en la
    senal de rodilla, inflen el rango pico a pico."""
    if x.size < 2:
        return float("nan")
    return float(np.percentile(x, 95) - np.percentile(x, 5))

def metrics_simple(pred, true):
    e = pred - true
    rmse = float(np.sqrt(np.mean(e ** 2)))
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(pred, true)[0, 1]) if true.size > 1 else float("nan")
    return rmse, r2, r


def ventanas_validas(mask):
    """Devuelve lista de (ini, fin) de tramos contiguos True en mask."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    cortes = np.where(np.diff(idx) > 1)[0]
    grupos = np.split(idx, cortes + 1)
    return [(g[0], g[-1] + 1) for g in grupos if len(g) >= 10]


def normalizar(seg, n=N_POINTS):
    seg = np.asarray(seg, dtype=float)
    if len(seg) < 10 or not np.isfinite(seg).all():
        return None
    return np.interp(np.linspace(0, 100, n), np.linspace(0, 100, len(seg)), seg)


def main(stage2_path):
    device = torch.device("cpu")
    model = load_stage2(stage2_path, device)
    warm = model.get_effective_history() + 10
    print(f"warm-up excluido: primeras {warm} muestras por trial\n")

    # Acumula por sujeto: predicho y real concatenados, y los ciclos normalizados
    per_subj = {}        # subj -> {col: ([pred...],[true...])}
    ciclos = {}          # subj -> {col: {"real": [101...], "pred": [101...]}}

    for csv in sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.csv"))):
        subj = os.path.basename(os.path.dirname(csv))
        df = pd.read_csv(csv)
        yp = predict_trial(model, df, device)
        if yp is None:
            continue
        valid_warm = np.zeros(len(df), dtype=bool)
        valid_warm[warm:] = True

        for k, col in enumerate(OUT_COLS):
            t = df[col].to_numpy()
            m = np.isfinite(t) & valid_warm
            if not m.any():
                continue
            ps = per_subj.setdefault(subj, {c: ([], []) for c in OUT_COLS})
            ps[col][0].append(yp[m, k]); ps[col][1].append(t[m])

            # ciclos normalizados (cada ventana de apoyo valida -> 101 puntos)
            cy = ciclos.setdefault(subj, {c: {"real": [], "pred": []} for c in OUT_COLS})
            for ini, fin in ventanas_validas(m):
                cr = normalizar(t[ini:fin]); cp = normalizar(yp[ini:fin, k])
                if cr is not None and cp is not None:
                    cy[col]["real"].append(cr); cy[col]["pred"].append(cp)

    # ---------- vc_per_subject.csv ----------
    filas = []
    for subj in sorted(per_subj):
        fila = {"subject": subj}
        for col, pref in [(OUT_COLS[0], "hip"), (OUT_COLS[1], "knee")]:
            p = np.concatenate(per_subj[subj][col][0])
            t = np.concatenate(per_subj[subj][col][1])
            rmse, r2, r = metrics_simple(p, t)
            fila[f"{pref}_rmse"] = rmse
            fila[f"{pref}_r"] = r
            fila[f"{pref}_r2"] = r2
            fila[f"{pref}_amp_real"] = amplitud(t)
            fila[f"{pref}_amp_pred"] = amplitud(p)
        filas.append(fila)
    ps_df = pd.DataFrame(filas)
    ps_df.to_csv("vc_per_subject.csv", index=False)
    print(f"Guardado vc_per_subject.csv ({len(ps_df)} sujetos)")

    # ---------- sujeto representativo (RMSE global ~ mediana) ----------
    ps_df = ps_df.copy()
    ps_df["rmse_global"] = ps_df[["hip_rmse", "knee_rmse"]].mean(axis=1)
    mediana = ps_df["rmse_global"].median()
    suj = ps_df.iloc[(ps_df["rmse_global"] - mediana).abs().argmin()]["subject"]
    print(f"Sujeto representativo (RMSE ~ mediana): {suj}")

    # ---------- vc_overlay.csv (medias 0-100% del sujeto representativo) ----------
    cy = ciclos.get(suj)
    out = {"pct": np.linspace(0, 100, N_POINTS)}
    ok = True
    for col, pref in [(OUT_COLS[0], "hip"), (OUT_COLS[1], "knee")]:
        if not cy or not cy[col]["real"] or not cy[col]["pred"]:
            ok = False; break
        R = np.array(cy[col]["real"]); P = np.array(cy[col]["pred"])
        out[f"{pref}_real"] = R.mean(axis=0); out[f"{pref}_pred"] = P.mean(axis=0)
        out[f"{pref}_real_sd"] = R.std(axis=0); out[f"{pref}_pred_sd"] = P.std(axis=0)
    if ok:
        pd.DataFrame(out).to_csv("vc_overlay.csv", index=False)
        print(f"Guardado vc_overlay.csv (sujeto {suj})")
    else:
        print(f"AVISO: el sujeto {suj} no tiene suficientes ciclos en alguna "
              f"articulacion; prueba con otro de RMSE cercano a la mediana.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else STAGE2_MODEL
    main(path)
