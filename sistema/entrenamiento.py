import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from algoritmos import crear
from sistema.vision import sha256

DESCONOCIDO = "__desconocido__"


def plantillas(vectors, labels, people):
    centers = np.stack([vectors[labels == person].mean(axis=0) for person in people])
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Plantilla facial degenerada.")
    return centers / norms


def decidir(predicted, vectors, centers, people, threshold, margin):
    scores = vectors @ centers.T
    order = np.argsort(scores, axis=1)
    best = order[:, -1]
    top = scores[np.arange(len(scores)), best]
    gap = top - scores[np.arange(len(scores)), order[:, -2]]
    nearest = np.asarray(people)[best]
    accepted = (predicted == nearest) & (top >= threshold) & (gap >= margin)
    return np.where(accepted, predicted, DESCONOCIDO), top, gap


def metricas(y, predicted, decisions, people):
    known = y != DESCONOCIDO
    unknown = ~known
    return {
        "muestras": len(y),
        "error_clasificacion_conocidos": float(np.mean(predicted[known] != y[known])),
        "error_acceso": float(np.mean(decisions != y)),
        "rechazo_o_identidad_incorrecta_conocidos": float(np.mean(decisions[known] != y[known])),
        "aceptacion_desconocidos": float(np.mean(decisions[unknown] != DESCONOCIDO)) if unknown.any() else None,
        "n_desconocidos": int(unknown.sum()),
        "matriz_confusion_acceso": confusion_matrix(y, decisions, labels=[*people, DESCONOCIDO]).tolist(),
    }


def entrenar(nombre, data, cfg, humanos=None):
    print(f"\nEntrenando identidad: {nombre}", flush=True)
    people = cfg["autorizados"]
    train = data["train"]
    model = crear(nombre, cfg["seed"])
    key = "crops" if nombre == "cnn" else "vectors"
    model.fit(train[key], train["labels"])
    centers = plantillas(train["vectors"], train["labels"], people)
    predictions = {split: model.predict(part[key]) for split, part in data.items()}
    # Calibración únicamente con val. Test nunca decide los umbrales.
    val = data["val"]
    threshold, margin = cfg["coseno_minimo"], cfg["margen_minimo"]
    if np.any(val["labels"] == DESCONOCIDO):
        candidates = []
        for t in np.unique(np.r_[np.linspace(threshold, .99, 30), 1.000001]):
            for m in np.unique(np.r_[margin, .16, .22, .30]):
                m = max(m, margin)
                decisions, _, _ = decidir(predictions["val"], val["vectors"], centers, people, t, m)
                stats = metricas(val["labels"], predictions["val"], decisions, people)
                if stats["aceptacion_desconocidos"] == 0:
                    candidates.append((stats["rechazo_o_identidad_incorrecta_conocidos"], t, m))
        _, threshold, margin = min(candidates)
    stats = {}
    coverage = {}
    for split, part in data.items():
        decisions, _, _ = decidir(predictions[split], part["vectors"], centers, people, threshold, margin)
        stats[split] = metricas(part["labels"], predictions[split], decisions, people)
        excluded = part.get("n_excluidas", 0)
        stats[split]["excluidas_calidad"] = excluded
        stats[split]["error_acceso_incluyendo_calidad"] = float(
            (np.sum(decisions != part["labels"]) + excluded) / (len(decisions) + excluded))
        if humanos is not None:
            human_predictions = humanos["modelo"].predict(part["human_" + key])
            cascade = np.where(human_predictions == "humano", decisions, DESCONOCIDO)
            stats[split]["error_cascada_incluyendo_calidad"] = float(
                (np.sum(cascade != part["labels"]) + excluded) / (len(cascade) + excluded))
        coverage[split] = all(np.any((part["labels"] == p) & (decisions == p)) for p in people)
    gap = stats["test"]["error_clasificacion_conocidos"] - stats["train"]["error_clasificacion_conocidos"]
    access_gap = stats["test"]["error_acceso"] - stats["train"]["error_acceso"]
    report = {
        "tarea": "identidad", "algoritmo": nombre, "seed": cfg["seed"], "metricas": stats,
        "brecha_test_menos_train": gap, "brecha_acceso": access_gap,
        "coseno_minimo": float(threshold), "margen_minimo": float(margin),
        "orden_matriz": [*people, DESCONOCIDO],
        "particiones": {s: d["paths"] for s, d in data.items()},
        "nota": "Errores empíricos de identidad; no miden PAD ni garantizan seguridad.",
    }
    # Control mínimo para habilitar pruebas GPIO; no equivale a certificación.
    ready = all(stats[s]["n_desconocidos"] > 0
                and stats[s]["aceptacion_desconocidos"] == 0 and coverage[s]
                for s in ("val", "test"))
    artifact = {
        "version": 1, "algoritmo": nombre, "modelo": model, "personas": people,
        "centros": centers, "coseno_minimo": float(threshold), "margen_minimo": float(margin),
        "config_coseno_minimo": cfg["coseno_minimo"], "config_margen_minimo": cfg["margen_minimo"],
        "detector_sha256": sha256(cfg["detector"]),
        "reconocedor_sha256": sha256(cfg["reconocedor"]), "evaluacion_basica_aprobada": ready,
        "reporte": report,
    }
    output = Path(cfg["artefactos"])
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output / f"{nombre}.joblib")
    (output / f"{nombre}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{nombre.upper()}")
    for split in ("train", "val", "test"):
        row = stats[split]
        far = row["aceptacion_desconocidos"]
        far_text = "no medida" if far is None else f"{far:.2%}"
        print(f"  Error {split}: {row['error_clasificacion_conocidos']:.2%}; "
              f"error acceso: {row['error_acceso']:.2%}; FAR desconocidos: {far_text}")
        print(f"    Excluidas por calidad: {row['excluidas_calidad']}; "
              f"error acceso incluyendo calidad: {row['error_acceso_incluyendo_calidad']:.2%}")
        if "error_cascada_incluyendo_calidad" in row:
            print(f"    Error humano + identidad (incluye calidad): {row['error_cascada_incluyendo_calidad']:.2%}")
    print(f"  Brecha test - entrenamiento: {gap * 100:.2f} puntos porcentuales")
    print(f"  Brecha de acceso: {access_gap * 100:.2f} puntos porcentuales")
    print(f"  Evaluación básica para GPIO: {'aprobada' if ready else 'pendiente'}")
    return report


def cargar_artefacto(nombre, cfg):
    # joblib/pickle solo debe cargarse de archivos propios y confiables.
    artifact = joblib.load(Path(cfg["artefactos"]) / f"{nombre}.joblib")
    if artifact["version"] != 1 or artifact["algoritmo"] != nombre or artifact["personas"] != cfg["autorizados"]:
        raise ValueError("Artefacto incompatible con la configuración actual. Reentrena.")
    if (artifact["config_coseno_minimo"] != cfg["coseno_minimo"]
            or artifact["config_margen_minimo"] != cfg["margen_minimo"]):
        raise ValueError("Cambiaste los umbrales base. Reentrena y calibra.")
    for key in ("detector", "reconocedor"):
        if artifact[f"{key}_sha256"] != sha256(cfg[key]):
            raise ValueError(f"Cambió el modelo de {key}. Reentrena.")
    return artifact


def identificar(artifact, vector, crop):
    X = crop[None] if artifact["algoritmo"] == "cnn" else vector[None]
    predicted = artifact["modelo"].predict(X)
    decisions, score, gap = decidir(predicted, vector[None], artifact["centros"],
        artifact["personas"], artifact["coseno_minimo"], artifact["margen_minimo"])
    return str(decisions[0]), float(score[0]), float(gap[0])
