"""Clasificación binaria rostro humano/objeto, separada de identidad y PAD."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from clasificacion.algoritmos import crear
from reconocimiento.preparacion import rasgos_humanos


def entrenar_humanos(nombre, data, cfg):
    print(f"\nEntrenando humano/objeto: {nombre}", flush=True)
    model = crear(nombre, cfg["seed"])
    key = "crops" if nombre == "cnn" else "vectors"
    model.fit(data["train"][key], data["train"]["labels"])
    stats = {}
    for split in ("train", "val", "test"):
        part = data[split]
        y = part["labels"]
        predicted = model.predict(part[key])
        stats[split] = {"muestras": len(y), "error_clasificacion": float(np.mean(y != predicted)),
            "humanos_rechazados": float(np.mean(predicted[y == "humano"] != "humano")),
            "objetos_aceptados": float(np.mean(predicted[y == "objeto"] == "humano")),
            "matriz_confusion": confusion_matrix(y, predicted, labels=["humano", "objeto"]).tolist()}
    report = {"tarea": "humanos", "algoritmo": nombre, "seed": cfg["seed"], "metricas": stats,
        "orden_matriz": ["humano", "objeto"],
        "brecha_test_menos_train": stats["test"]["error_clasificacion"] - stats["train"]["error_clasificacion"],
        "particiones": {s: p["paths"] for s, p in data.items()},
        "representacion": "RGB 64x64" if nombre == "cnn" else "HOG 64x64",
        "nota": "Rostros frente a objetos; no mide cuerpo humano, identidad, presencia física ni PAD. "
                "Sin IDs de personas en la fuente, no se garantiza independencia por identidad."}
    artifact = {"version": 1, "algoritmo": nombre, "modelo": model, "tarea": "humanos", "reporte": report}
    output = Path(cfg["artefactos"])
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output / f"humanos_{nombre}.joblib")
    (output / f"humanos_{nombre}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nHUMANO / OBJETO — {nombre.upper()}")
    for s in ("train", "val", "test"):
        print(f"  Error {s}: {stats[s]['error_clasificacion']:.2%} (n={stats[s]['muestras']})", flush=True)
    return report, artifact


def es_humano(artifact, frame):
    vector, crop = rasgos_humanos(frame)
    X = crop[None] if artifact["algoritmo"] == "cnn" else vector[None]
    return artifact["modelo"].predict(X)[0] == "humano"


def cargar_humanos(nombre, cfg):
    artifact = joblib.load(Path(cfg["artefactos"]) / f"humanos_{nombre}.joblib")
    if artifact["version"] != 1 or artifact["tarea"] != "humanos" or artifact["algoritmo"] != nombre:
        raise ValueError("Modelo humano/objeto incompatible. Reentrena.")
    return artifact
