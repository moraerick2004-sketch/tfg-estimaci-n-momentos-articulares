"""
evaluate_stage2_vancriekinge.py

Evalua SOLO la Etapa 2 (angulos + velocidades -> momentos) sobre la cohorte
mayor (>60) de Van Criekinge ya adaptada en data_vancriekinge/.

Claves (confirmadas en tcn.py / inference_pipeline.py):
  - El modelo normaliza la entrada INTERNAMENTE: forward() hace (x-center)/scale
    con center/scale del checkpoint. => se alimenta con unidades crudas (deg, deg/s).
  - Orden de canales de entrada INTERCALADO:
        [hip_angle, hip_vel, knee_angle, knee_vel]
  - Salida (1, 2, T) en Nm/kg crudos. No hay que desnormalizar.
  - Se excluye el warm-up causal (eff_hist + 10), como en inference_pipeline.

Ejecutar desde la carpeta del proyecto (necesita tcn.py, dataloader_camargo.py,
inference_pipeline.py importables):
    python evaluate_stage2_vancriekinge.py [ruta_stage2.tar]
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import inspect
from tcn import TCN

# Rutas por defecto, relativas a la carpeta del proyecto. Se pueden cambiar con
# la variable de entorno VC_DATA_DIR o pasando la ruta al ejecutar (ver __main__).
DATA_DIR = os.environ.get("VC_DATA_DIR", "./data_vancriekinge")
STAGE2_MODEL = "./trained_models/trained_tcn_stage2_angles_to_moments.tar"

# Orden EXACTO de entrada de la Etapa 2 (intercalado, como en inference_pipeline):
IN_COLS  = ["hip_angle_r", "hip_angle_r_velocity_filt",
            "knee_angle_r", "knee_angle_r_velocity_filt"]
# Salida: indice 0 = cadera, 1 = rodilla
OUT_COLS = ["hip_flexion_r_moment", "knee_angle_r_moment"]


_TCN_PARAMS = set(inspect.signature(TCN.__init__).parameters) - {"self"}


def load_stage2(path, device):
    """Carga el checkpoint de Etapa 2 reconstruyendo el TCN con la norma correcta.
    El modelo se entreno con InstanceNorm1d, pero el checkpoint no guarda 'norm',
    asi que se infiere del state_dict (weight_norm tiene claves *.weight_g)."""
    ckpt = torch.load(path, map_location=device)
    state = ckpt["state_dict"]
    kwargs = {k: v for k, v in ckpt.items() if k in _TCN_PARAMS}
    has_wn = any(k.endswith("weight_g") for k in state)
    kwargs["norm"] = "weight_norm" if has_wn else "InstanceNorm1d"
    model = TCN(**kwargs).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"modelo cargado (norm={kwargs['norm']}, "
          f"input_size={kwargs.get('input_size')}, output_size={kwargs.get('output_size')})")
    return model


@torch.no_grad()
def predict_trial(model, df, device):
    X = df[IN_COLS].to_numpy(dtype=np.float32)        # (T, 4) crudo
    if not np.isfinite(X).all():
        return None                                    # entradas deben ser finitas
    xt = torch.from_numpy(np.ascontiguousarray(X.T)[None]).to(device)  # (1,4,T)
    return model(xt).cpu().numpy()[0].T                # (T, 2) Nm/kg


def metrics(pred, true):
    e = pred - true
    rmse = float(np.sqrt(np.mean(e ** 2)))
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(pred, true)[0, 1]) if true.size > 1 else float("nan")
    amp = float(np.ptp(pred) / np.ptp(true)) if np.ptp(true) > 0 else float("nan")
    # nRMSE con la MISMA definicion que en Camargo: RMSE / |x|max * 100
    peak = float(np.max(np.abs(true)))
    nrmse = 100.0 * rmse / peak if peak > 0 else float("nan")
    return rmse, nrmse, r2, r, amp, int(true.size)


def main(stage2_path):
    device = torch.device("cpu")
    model = load_stage2(stage2_path, device)
    warm = model.get_effective_history() + 10
    print(f"warm-up excluido: primeras {warm} muestras por trial\n")

    pooled = {c: {"p": [], "t": []} for c in OUT_COLS}
    per_subj = {}
    overlay = None
    n_skip_inputs = 0

    for csv in sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.csv"))):
        subj = os.path.basename(os.path.dirname(csv))
        df = pd.read_csv(csv)
        yp = predict_trial(model, df, device)
        if yp is None:
            n_skip_inputs += 1
            continue
        valid_warm = np.zeros(len(df), dtype=bool)
        valid_warm[warm:] = True
        for k, col in enumerate(OUT_COLS):
            t = df[col].to_numpy()
            m = np.isfinite(t) & valid_warm
            if not m.any():
                continue
            pooled[col]["p"].append(yp[m, k]); pooled[col]["t"].append(t[m])
            ps = per_subj.setdefault(subj, {c: ([], []) for c in OUT_COLS})
            ps[col][0].append(yp[m, k]); ps[col][1].append(t[m])
        if overlay is None and (np.isfinite(df[OUT_COLS[0]]) & valid_warm).sum() > 50:
            overlay = (subj, df, yp, valid_warm)

    if n_skip_inputs:
        print(f"AVISO: {n_skip_inputs} trials saltados por entradas no finitas\n")

    print("== GLOBAL (cohorte >60, solo apoyo derecho valido) ==")
    for col in OUT_COLS:
        p = np.concatenate(pooled[col]["p"]); t = np.concatenate(pooled[col]["t"])
        rmse, nrmse, r2, r, amp, n = metrics(p, t)
        print(f"  {col:24s} RMSE={rmse:.4f} Nm/kg   nRMSE={nrmse:.1f}%   R2={r2:6.3f}   "
              f"r={r:.3f}   amp_pred/real={amp:.2f}   n={n}")

    print(f"\n== POR SUJETO ({len(per_subj)} sujetos) ==")
    dist = {c: {"r2": [], "r": []} for c in OUT_COLS}
    for subj in sorted(per_subj):
        out = []
        for col in OUT_COLS:
            p = np.concatenate(per_subj[subj][col][0])
            t = np.concatenate(per_subj[subj][col][1])
            _, _, r2, r, _, n = metrics(p, t)
            dist[col]["r2"].append(r2)
            dist[col]["r"].append(r)
            out.append(f"{col.split('_')[0]} R2={r2:6.2f} r={r:5.2f}")
        print(f"  {subj:8s} n={n:4d}  " + "   ".join(out))

    print("\n== RESUMEN POR SUJETO ==")
    for col in OUT_COLS:
        r2 = np.array(dist[col]["r2"]); r = np.array(dist[col]["r"])
        print(f"  {col:24s} R2 mediana={np.median(r2):6.2f} (positivos {int((r2 > 0).sum())}/{len(r2)})"
              f"   r mediana={np.median(r):.2f}")

    if overlay:
        subj, df, yp, vw = overlay
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, col in enumerate(OUT_COLS):
            t = df[col].to_numpy(); m = np.isfinite(t) & vw
            idx = np.where(m)[0]
            ax[k].plot(idx, t[m], lw=2, label="real")
            ax[k].plot(idx, yp[m, k], "--", lw=2, label="predicho")
            ax[k].set_title(f"{subj} | {col}"); ax[k].set_xlabel("muestra")
            ax[k].legend()
        plt.tight_layout(); plt.savefig("overlay_pred_vs_real.png", dpi=130)
        print("\nGuardado overlay_pred_vs_real.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else STAGE2_MODEL
    main(path)
