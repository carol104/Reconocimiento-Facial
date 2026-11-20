"""Clasificación binaria rostro humano/objeto, separada de identidad y PAD."""
import json
from time import perf_counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from algoritmos import crear
from sistema.preparacion import rasgos_humanos

from sistema.preprocesamiento import contrato, entrada_entrenamiento, latencia_modelo


def entrenar_humanos(nombre, data, cfg):
    print(f"\nEntrenando humano/objeto: {nombre}", flush=True)
    model = crear(nombre, cfg["seed"])
    key = "crops" if nombre == "cnn" else "vectors"
    start_fit = perf_counter()
    X_fit, y_fit = entrada_entrenamiento(data["train"], cfg, nombre)
    if nombre == "cnn":
        for option in ("epochs", "dropout", "l2", "patience"):
            if option in cfg.get("cnn", {}):
                setattr(model, option, cfg["cnn"][option])
        model.fit(X_fit, y_fit, validation_data=(data["val"][key], data["val"]["labels"]))
    else:
        model.fit(X_fit, y_fit)
    segundos_ajuste = perf_counter() - start_fit
    latencia_ms = latencia_modelo(model, data["val"][key])
    stats = {}
    for split in ("train", "val", "test"):
        part = data[split]
        y = part["labels"]
        predicted = model.predict(part[key])
        stats[split] = {"muestras": len(y), "error_clasificacion": float(np.mean(y != predicted)),
            "accuracy": float(np.mean(y == predicted)),
            "humanos_rechazados": float(np.mean(predicted[y == "humano"] != "humano")),
            "objetos_aceptados": float(np.mean(predicted[y == "objeto"] == "humano")),
            "matriz_confusion": confusion_matrix(y, predicted, labels=["humano", "objeto"]).tolist()}
    report = {"tarea": "humanos", "algoritmo": nombre, "seed": cfg["seed"], "metricas": stats,
        "orden_matriz": ["humano", "objeto"],
        "preprocesamiento": contrato(cfg), "muestras_ajuste": len(y_fit),
        "segundos_ajuste": segundos_ajuste, "inferencia_host_ms": latencia_ms,
        "historial_cnn": getattr(model, "history_", None),
        "backend": "tensorflow.keras" if nombre == "cnn" else "scikit-learn",
        "brecha_test_menos_train": stats["test"]["error_clasificacion"] - stats["train"]["error_clasificacion"],
        "particiones": {s: p["paths"] for s, p in data.items()},
        "representacion": "gris 64x64x1" if nombre == "cnn" else "HOG del mismo gris 64x64",
        "nota": "Rostros frente a objetos; no mide cuerpo humano, identidad, presencia física ni PAD. "
                "Sin IDs de personas en la fuente, no se garantiza independencia por identidad."}
    artifact = {"version": 2, "preprocesamiento": contrato(cfg), "algoritmo": nombre, "modelo": model, "tarea": "humanos", "reporte": report}
    output = Path(cfg["artefactos"])
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output / f"humanos_{nombre}.joblib")
    (output / f"humanos_{nombre}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nHUMANO / OBJETO — {nombre.upper()}")
    for s in ("train", "val", "test"):
        print(f"  Error {s}: {stats[s]['error_clasificacion']:.2%} (n={stats[s]['muestras']})", flush=True)
    return report, artifact


def es_humano(artifact, frame):
    vector, crop = rasgos_humanos(frame, {"preprocesamiento": artifact["preprocesamiento"]})
    X = crop[None] if artifact["algoritmo"] == "cnn" else vector[None]
    return artifact["modelo"].predict(X)[0] == "humano"


def cargar_humanos(nombre, cfg):
    report_path = Path(cfg["artefactos"]) / f"humanos_{nombre}.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("preprocesamiento") != contrato(cfg):
            raise ValueError("Modelo humano/objeto incompatible: reentrena con el nuevo preprocesamiento.")
    artifact = joblib.load(Path(cfg["artefactos"]) / f"humanos_{nombre}.joblib")
    if artifact["version"] != 2 or artifact.get("preprocesamiento") != contrato(cfg) or artifact["tarea"] != "humanos" or artifact["algoritmo"] != nombre:
        raise ValueError("Modelo humano/objeto incompatible. Reentrena.")
    return artifact
