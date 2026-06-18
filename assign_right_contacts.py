"""
assign_right_contacts.py

Para un C3D: encuentra contactos de plataforma, asigna cada uno a pie izq/der,
y devuelve las ventanas VALIDAS del pie DERECHO (con recorte de bordes).

Asignacion de pie por magnitud del momento de la pierna:
  en apoyo el momento de esa pierna es grande; en swing es pequenio.
  => el contacto cuyo |RHipMoment| domina sobre |LHipMoment| es del pie DERECHO.
(Solo usa datos POINT -> evita ambiguedades del sistema de referencia de la COP.)

pip install ezc3d numpy
"""
import numpy as np
from ezc3d import c3d


def right_contact_windows(path, mass_kg, fz_thr=0.05, edge_trim=0.08,
                          peak_min=0.7, verbose=True):
    c = c3d(path, extract_forceplat_data=True)
    prate = c["parameters"]["POINT"]["RATE"]["value"][0]
    arate = c["parameters"]["ANALOG"]["RATE"]["value"][0]
    L = c["parameters"]["POINT"]["LABELS"]["value"]
    idx = {n: i for i, n in enumerate(L)}
    pts = c["data"]["points"]
    rhip = np.abs(pts[0, idx["RHipMoment"], :]) / 1000.0
    lhip = np.abs(pts[0, idx["LHipMoment"], :]) / 1000.0
    step = int(round(arate / prate))
    BW = mass_kg * 9.81

    windows = []
    for i, p in enumerate(c["data"].get("platform", [])):
        fz = np.abs(p["force"][2, :])
        on = fz > fz_thr * BW
        if not on.any():
            continue
        a0, a1 = np.where(on)[0][[0, -1]]
        peak = fz.max() / BW
        f0, f1 = a0 // step, a1 // step
        mR, mL = rhip[f0:f1 + 1].mean(), lhip[f0:f1 + 1].mean()
        side = "R" if mR > mL else "L"
        t0, t1 = f0 / prate, f1 / prate
        partial = peak < peak_min
        if verbose:
            tag = " PARCIAL" if partial else ""
            print(f"  plato {i+1}: {t0:.2f}-{t1:.2f}s  pico={peak:.2f}xBW  "
                  f"|Rmom|={mR:.2f} |Lmom|={mL:.2f} -> {side}{tag}")
        if side == "R" and not partial:
            windows.append((round(t0 + edge_trim, 3), round(t1 - edge_trim, 3)))
    return windows


if __name__ == "__main__":
    # Ejemplo de uso: edita estas rutas a dos C3D tuyos para ver las ventanas
    # validas del pie derecho. SUBJ34 -> 76 kg, SUBJ45 -> 79 kg (older_adults_cohort.py).
    BASE = "./138_HealthyPiG_10.05"
    for path, mass in [
        (f"{BASE}/SUBJ34/SUBJ34 (2).c3d", 76),
        (f"{BASE}/SUBJ45/SUBJ45 (2).c3d", 79),
    ]:
        print(path.split("/")[-1])
        w = right_contact_windows(path, mass)
        print("  -> ventanas validas pie DERECHO (s):", w, "\n")
