#!/usr/bin/env python3
"""
Genera las tres figuras de la adicion de Van Criekinge (cap4 resultados):

  - vc_overlay.pdf     : momento predicho vs real de un sujeto representativo
  - vc_per_subject.pdf : r intra-sujeto frente a R2 por sujeto (forma vs amplitud)
  - vc_amplitude.pdf   : amplitud predicha vs real por sujeto (subestimacion)

Lee dos CSV (ver esquema en el mensaje). Ejecuta una vez para obtener las figuras
2 y 3 y la sugerencia de sujeto representativo; exporta luego vc_overlay.csv de
ese sujeto y vuelve a ejecutar para la figura 1.

Requisitos: numpy, pandas, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================= CONFIGURACION =======================
PER_SUBJECT_CSV = "vc_per_subject.csv"   # una fila por sujeto
OVERLAY_CSV     = "vc_overlay.csv"       # una fila por % de ciclo (sujeto representativo)
OUT_DIR         = "img"
# =============================================================

os.makedirs(OUT_DIR, exist_ok=True)
AZUL, ROJO = "#1f4e79", "#7d2e2e"

# ---------- Figuras 2 y 3 + sugerencia de sujeto ----------
ps = pd.read_csv(PER_SUBJECT_CSV).copy()

# Sugerencia de sujeto representativo: RMSE global mas cercano a la mediana y al
# percentil 25 (NO el mejor, para no hacer cherry-picking).
ps["rmse_global"] = ps[["hip_rmse", "knee_rmse"]].mean(axis=1)
mediana = ps["rmse_global"].median()
p25 = ps["rmse_global"].quantile(0.25)
suj_mediana = ps.iloc[(ps["rmse_global"] - mediana).abs().argmin()]["subject"]
suj_p25 = ps.iloc[(ps["rmse_global"] - p25).abs().argmin()]["subject"]
print("\n--- Sujeto representativo para el overlay (figura 1) ---")
print(f"  Mas cercano a la MEDIANA de RMSE: {suj_mediana}")
print(f"  Mas cercano al PERCENTIL 25:      {suj_p25}")
print("  Exporta vc_overlay.csv para uno de estos (no el de mejor R2).")
print("-------------------------------------------------------\n")

# Figura 2: r intra-sujeto frente a R2 por sujeto
fig, ax = plt.subplots(figsize=(5.2, 4))
ax.scatter(ps["hip_r"], ps["hip_r2"], color=AZUL, label="Cadera", s=30, alpha=0.8)
ax.scatter(ps["knee_r"], ps["knee_r2"], color=ROJO, label="Rodilla", s=30, alpha=0.8, marker="^")
ax.axhline(0, color="gray", lw=1, ls="--")
ax.set_xlabel("r intra-sujeto")
ax.set_ylabel("R$^2$ por sujeto")
ax.set_xlim(0, 1)
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "vc_per_subject.pdf"), bbox_inches="tight")
plt.close(fig)
print("Guardado:", os.path.join(OUT_DIR, "vc_per_subject.pdf"))

# Figura 3: amplitud predicha frente a amplitud real por sujeto
fig, ax = plt.subplots(figsize=(5.2, 4))
ax.scatter(ps["hip_amp_real"], ps["hip_amp_pred"], color=AZUL, label="Cadera", s=30, alpha=0.8)
ax.scatter(ps["knee_amp_real"], ps["knee_amp_pred"], color=ROJO, label="Rodilla", s=30, alpha=0.8, marker="^")
amp_max = float(np.nanmax([ps["hip_amp_real"].max(), ps["hip_amp_pred"].max(),
                           ps["knee_amp_real"].max(), ps["knee_amp_pred"].max()])) * 1.05
ax.plot([0, amp_max], [0, amp_max], color="gray", lw=1, ls="--", label="Predicción perfecta")
ax.set_xlabel("Amplitud real (Nm/kg)")
ax.set_ylabel("Amplitud predicha (Nm/kg)")
ax.set_xlim(0, amp_max)
ax.set_ylim(0, amp_max)
ax.set_aspect("equal")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "vc_amplitude.pdf"), bbox_inches="tight")
plt.close(fig)
print("Guardado:", os.path.join(OUT_DIR, "vc_amplitude.pdf"))

# ---------- Figura 1: overlay (solo si existe el CSV) ----------
if os.path.exists(OVERLAY_CSV):
    ov = pd.read_csv(OVERLAY_CSV)
    x = ov["pct"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, art, titulo in [(axes[0], "hip", "Cadera"), (axes[1], "knee", "Rodilla")]:
        real, pred = ov[f"{art}_real"].to_numpy(), ov[f"{art}_pred"].to_numpy()
        ax.plot(x, real, color=AZUL, lw=2, label="Real (Van Criekinge)")
        ax.plot(x, pred, color=ROJO, lw=2, ls="--", label="Predicción (Etapa 2)")
        if f"{art}_real_sd" in ov.columns:
            sd = ov[f"{art}_real_sd"].to_numpy()
            ax.fill_between(x, real - sd, real + sd, color=AZUL, alpha=0.15)
        if f"{art}_pred_sd" in ov.columns:
            sd = ov[f"{art}_pred_sd"].to_numpy()
            ax.fill_between(x, pred - sd, pred + sd, color=ROJO, alpha=0.15)
        ax.set_title(titulo)
        ax.set_xlabel("Ciclo de marcha (%)")
        ax.set_ylabel("Momento (Nm/kg)")
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "vc_overlay.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", os.path.join(OUT_DIR, "vc_overlay.pdf"))
else:
    print(f"(Falta {OVERLAY_CSV}: exportalo para el sujeto representativo y vuelve a ejecutar.)")
