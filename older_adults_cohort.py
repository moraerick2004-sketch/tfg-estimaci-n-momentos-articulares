"""
older_adults_cohort.py

Cohorte de adultos mayores sanos de:
Van Criekinge et al. (2023), "A full-body motion capture gait dataset of 138
able-bodied adults across the life span and 50 stroke survivors", Scientific Data.
Demografia: Supplementary Table 1 (able-bodied adults).
(Transcrito a mano del PDF suplementario -> conviene verificar contra el xls original.)

AVISO IMPORTANTE SOBRE MOMENTOS Y MASA
--------------------------------------
En este dataset los momentos de Plug-in Gait guardados en el C3D YA estan
normalizados por masa corporal (unidades ~Nmm/kg; dividir entre 1000 -> Nm/kg).
=> SUBJECT_MASS_KG de abajo es solo para VERIFICACION / informe demografico.
   NO vuelvas a dividir los momentos por la masa (eso era correcto en Camargo,
   aqui seria un error de doble normalizacion).
"""

# id -> (edad_anios, sexo, masa_kg, altura_m)
_RAW = {
    "SUBJ1":  (86, "M",  64, 1.580), "SUBJ2":  (85, "F",  78, 1.500),
    "SUBJ3":  (85, "F",  69, 1.510), "SUBJ4":  (84, "M",  70, 1.625),
    "SUBJ5":  (84, "F",  50, 1.450), "SUBJ6":  (83, "M",  80, 1.695),
    "SUBJ7":  (82, "F",  62, 1.555), "SUBJ8":  (82, "F",  72, 1.600),
    "SUBJ9":  (82, "M",  87, 1.710), "SUBJ10": (81, "M",  73, 1.620),
    "SUBJ11": (81, "M",  76, 1.700), "SUBJ12": (80, "M", 101, 1.890),
    "SUBJ13": (80, "F",  70, 1.540), "SUBJ14": (80, "M",  87, 1.755),
    "SUBJ15": (83, "F",  74, 1.530), "SUBJ16": (79, "F",  74, 1.610),
    "SUBJ17": (77, "M",  86, 1.750), "SUBJ18": (77, "M",  90, 1.770),
    "SUBJ19": (77, "M",  88, 1.830), "SUBJ20": (76, "M",  79, 1.750),
    "SUBJ21": (75, "F",  78, 1.420), "SUBJ22": (79, "M",  83, 1.645),
    "SUBJ23": (75, "M",  89, 1.665), "SUBJ24": (74, "F",  72, 1.585),
    "SUBJ25": (74, "F",  83, 1.545), "SUBJ26": (73, "F",  64, 1.605),
    "SUBJ27": (73, "F",  61, 1.595), "SUBJ28": (72, "F",  69, 1.605),
    "SUBJ29": (72, "F",  75, 1.560), "SUBJ30": (72, "F",  61, 1.585),
    "SUBJ31": (71, "M",  65, 1.610), "SUBJ32": (71, "M",  83, 1.725),
    "SUBJ33": (70, "M",  82, 1.795), "SUBJ34": (69, "M",  76, 1.655),
    "SUBJ35": (67, "M",  92, 1.830), "SUBJ36": (67, "M",  74, 1.690),
    "SUBJ37": (66, "F",  55, 1.615), "SUBJ38": (66, "F",  57, 1.635),
    "SUBJ39": (65, "F",  69, 1.645), "SUBJ40": (65, "M",  95, 1.750),
    "SUBJ41": (64, "F",  75, 1.580), "SUBJ42": (64, "F",  82, 1.550),
    "SUBJ43": (62, "M",  95, 1.770), "SUBJ44": (62, "F",  80, 1.565),
    "SUBJ45": (61, "F",  79, 1.615),
    # --- exactamente 60 anios (incluir solo si usas umbral >=60) ---
    "SUBJ46": (60, "F",  66, 1.580), "SUBJ47": (60, "M",  69, 1.710),
    "SUBJ48": (60, "M",  59, 1.740), "SUBJ49": (60, "F",  85, 1.645),
}

SUBJECTS = {
    sid: {"age": a, "sex": s, "mass_kg": m, "height_m": h}
    for sid, (a, s, m, h) in _RAW.items()
}

SUBJECT_MASS_KG = {sid: v["mass_kg"] for sid, v in SUBJECTS.items()}
SUBJECT_HEIGHT_M = {sid: v["height_m"] for sid, v in SUBJECTS.items()}


def cohort(min_age=61):
    """IDs con edad >= min_age. '>60' del tutor -> min_age=61 (45 sujetos)."""
    return [sid for sid, v in SUBJECTS.items() if v["age"] >= min_age]


if __name__ == "__main__":
    for thr in (61, 65, 70):
        ids = cohort(thr)
        n = len(ids)
        nf = sum(SUBJECTS[i]["sex"] == "F" for i in ids)
        ages = [SUBJECTS[i]["age"] for i in ids]
        print(f"edad >= {thr}: {n} sujetos ({n - nf}H/{nf}M), "
              f"rango {min(ages)}-{max(ages)} anios")
