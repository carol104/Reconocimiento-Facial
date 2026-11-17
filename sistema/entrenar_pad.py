"""Entrenamiento opcional de PAD binario con ataques reales capturados por cámara."""
import json
from pathlib import Path
import cv2
import joblib
import numpy as np
from algoritmos.cnn import CNN
from sistema.datos import EXTENSIONES
from sistema.vision import sha256, tensor_pad


def entrenar_pad(cfg, root, vision, epochs=30):
    root = Path(root)
    data, hashes, sessions = {}, set(), {}
    for split in ("train", "val", "test"):
        X, y, paths = [], [], []
        for kind in ("ataque", "real"):
            directory = root / split / kind
            for path in sorted(directory.rglob("*")):
                if path.suffix.lower() not in EXTENSIONES or not path.is_file():
                    continue
                parts = path.relative_to(directory).parts
                if len(parts) != 2:
                    raise ValueError(f"Usa pad/{split}/{kind}/sesion/imagen: {path}")
                session = parts[0]
                if session in sessions and sessions[session] != split:
                    raise ValueError(f"Sesión PAD compartida: {session}")
                sessions[session] = split
                digest = sha256(path)
                if digest in hashes:
                    raise ValueError(f"Imagen PAD duplicada: {path}")
                hashes.add(digest)
                frame = cv2.imread(str(path))
                if frame is None:
                    raise ValueError(f"Imagen ilegible: {path}")
                try:
                    _, _, face = vision.extraer(frame)
                except ValueError as exc:
                    raise ValueError(f"{path}: {exc}") from exc
                X.append(tensor_pad(frame, face))
                y.append(kind)
                paths.append(str(path.relative_to(root)))
            if y.count(kind) < 3:
                raise ValueError(f"PAD: requiere >= 3 imágenes por clase en {split}; usa muchas más en práctica.")
        data[split] = (np.asarray(X), np.asarray(y), paths)
    model = CNN(cfg["seed"], epochs)
    model.fit(*data["train"][:2])
    index = list(model.classes_).index("real")
    scores = {s: model.predict_proba(part[0])[:, index] for s, part in data.items()}
    val_labels = data["val"][1]
    choices = []
    for threshold in np.unique(np.r_[cfg["pad"]["umbral"], np.linspace(cfg["pad"]["umbral"], 1, 100)]):
        accepted = scores["val"] >= threshold
        if not accepted[val_labels == "ataque"].any():
            choices.append((float(np.mean(~accepted[val_labels == "real"])), float(threshold)))
    if not choices:
        raise ValueError("No se pudo calibrar PAD sin aceptar ataques en val; mejora los datos/modelo.")
    _, threshold = min(choices)
    report = {"umbral": threshold, "metricas": {}, "particiones": {s: p[2] for s, p in data.items()}}
    for split, (_, y, _) in data.items():
        accepted = scores[split] >= threshold
        predicted = model.predict(data[split][0])
        report["metricas"][split] = {
            "error_clasificacion": float(np.mean(predicted != y)),
            "APCER_ataques_aceptados": float(np.mean(accepted[y == "ataque"])),
            "BPCER_reales_rechazados": float(np.mean(~accepted[y == "real"])),
        }
    report["brecha_test_menos_train"] = (report["metricas"]["test"]["error_clasificacion"]
                                          - report["metricas"]["train"]["error_clasificacion"])
    output = Path(cfg["artefactos"])
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"tipo": "pad_cnn_v1", "modelo": model, "umbral": threshold}, output / "pad.joblib")
    (output / "pad.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metricas"], indent=2))
    print(f"Brecha PAD: {report['brecha_test_menos_train'] * 100:.2f} puntos porcentuales")
    print(f"Configura pad.modelo = {output / 'pad.joblib'} y pad.umbral = {threshold}")
    print("Valida también el reto completo con ataques en vivo antes de habilitar GPIO.")
    return report
