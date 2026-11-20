"""Manifiesto reproducible sin mover ni copiar los datasets originales."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sistema.datos import EXTENSIONES
from sistema.vision import sha256

from sistema.preprocesamiento import representar, contrato

PARTICIONES = ("train", "val", "test")


def leer_imagen(path):
    # imdecode admite rutas con tildes en Windows.
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Imagen ilegible: {path}")
    return frame


def preparar_fuentes(cfg):
    source = cfg["fuentes"]
    webcam, humanos = Path(source["webcam"]), Path(source["humanos"])
    modo = source.get("reparto_identidad", "sesiones")
    if modo not in ("sesiones", "mezclado"):
        raise ValueError("reparto_identidad debe ser sesiones o mezclado.")
    sesiones = source["sesiones"]
    if set(sesiones) != set(PARTICIONES) or any(not sesiones[s] for s in PARTICIONES):
        raise ValueError("Configura sesiones de train, val y test.")
    all_sessions = [s for split in PARTICIONES for s in sesiones[split]]
    if len(all_sessions) != len(set(all_sessions)):
        raise ValueError("Sesión compartida entre particiones en fuentes.sesiones.")
    if not webcam.is_dir() or not humanos.is_dir():
        raise ValueError("Faltan las carpetas fuente webcam o Human faces.")
    if {p.name for p in webcam.iterdir() if p.is_dir()} != set(cfg["autorizados"]):
        raise ValueError("Las cuatro carpetas de webcam deben coincidir con autorizados.")
    manifest = {"version": 1, "seed": cfg["seed"], "autorizados": cfg["autorizados"],
                "fuentes": source, "identidad": [], "humanos": [], "duplicados_omitidos": [],
                "conflictos_etiquetas": []}
    seen = {}
    conflict_pixels = set()

    def registro(path, label, split=None, session=None, task="identidad"):
        frame = leer_imagen(path)
        digest = hashlib.sha256(str(frame.shape).encode() + frame.tobytes()).hexdigest()
        previous = seen.get(digest)
        if previous:
            if task == previous["tarea"] == "humanos" and (
                    previous["label"] != label or digest in conflict_pixels):
                conflict_pixels.add(digest)
                manifest["conflictos_etiquetas"].append({"path": str(path), "label": label,
                    "original": previous["path"], "label_original": previous["label"]})
                return None
            if previous["tarea"] != task or previous["label"] != label or (
                    task == "identidad" and modo == "sesiones" and previous["sesion"] != session):
                raise ValueError(f"Duplicado entre clases, sesiones o tareas: {path} y {previous['path']}")
            manifest["duplicados_omitidos"].append({"path": str(path), "original": previous["path"]})
            return None
        row = {"path": str(path.resolve()), "label": label, "split": split,
               "sesion": session, "sha256": sha256(path), "pixeles_sha256": digest, "tarea": task}
        seen[digest] = row
        return row

    for person in cfg["autorizados"]:
        person_rows = []
        actual = {p.name for p in (webcam / person).iterdir() if p.is_dir()}
        if actual != set(all_sessions):
            raise ValueError(f"Sesiones de {person}: {sorted(actual)}; deben figurar todas en fuentes.sesiones.")
        for split in PARTICIONES:
            for session in sesiones[split]:
                paths = sorted(p for p in (webcam / person / session).rglob("*")
                               if p.is_file() and p.suffix.lower() in EXTENSIONES)
                if not paths:
                    raise ValueError(f"Sesión vacía: {person}/{session}")
                for path in paths:
                    row = registro(path, person, split, session)
                    if row:
                        person_rows.append(row)
        if modo == "mezclado":
            if len(person_rows) < 5:
                raise ValueError(f"Muy pocas imágenes únicas para {person} (mínimo 5).")
            train, rest = train_test_split(person_rows, test_size=.4, random_state=cfg["seed"])
            val, test = train_test_split(rest, test_size=.5, random_state=cfg["seed"])
            for split, partition in zip(PARTICIONES, (train, val, test)):
                for row in partition:
                    row["split"] = split
                    manifest["identidad"].append(row)
        else:
            manifest["identidad"].extend(person_rows)
    # Las rutas absolutas del CSV original pertenecen a otro equipo. Usamos las carpetas reales.
    human_rows = {}
    for label, folders in {"humano": ("male_faces", "female_faces"), "objeto": ("Objects",)}.items():
        rows = []
        for folder in folders:
            directory = humanos / folder
            paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES)
            if not paths:
                raise ValueError(f"Clase vacía: {directory}")
            for path in paths:
                row = registro(path, label, task="humanos")
                if row:
                    rows.append(row)
        human_rows[label] = rows
    for label, rows in human_rows.items():
        rows = [r for r in rows if r["pixeles_sha256"] not in conflict_pixels]
        if len(rows) < 5:
            raise ValueError(f"Muy pocas imágenes únicas para la clase {label}.")
        train, rest = train_test_split(rows, test_size=.4, random_state=cfg["seed"])
        val, test = train_test_split(rest, test_size=.5, random_state=cfg["seed"])
        for split, partition in zip(PARTICIONES, (train, val, test)):
            for row in partition:
                row["split"] = split
                manifest["humanos"].append(row)
    manifest["limite_humanos"] = (
        "Separación por imagen única, no por identidad: la fuente no incluye IDs de personas. "
        "No usar estos rostros como prueba de desconocidos independientes ni como PAD.")
    output = Path(cfg["artefactos"])
    output.mkdir(parents=True, exist_ok=True)
    target = output / "particiones.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for task in ("humanos", "identidad"):
        print(f"{task}: " + "; ".join(f"{s}={sum(r['split'] == s for r in manifest[task])}" for s in PARTICIONES))
    print(f"Duplicados exactos omitidos: {len(manifest['duplicados_omitidos'])}. Manifiesto: {target}")
    print(f"Conflictos de etiquetas en cuarentena (ambas clases): {len(conflict_pixels)} imágenes únicas.")
    return manifest


def cargar_manifiesto(cfg):
    path = Path(cfg["artefactos"]) / "particiones.json"
    if not path.exists():
        return preparar_fuentes(cfg)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest["fuentes"] != cfg["fuentes"] or manifest["seed"] != cfg["seed"]
            or manifest["autorizados"] != cfg["autorizados"]):
        raise ValueError("Cambió la configuración del dataset. Ejecuta preparar de nuevo.")
    for task in ("identidad", "humanos"):
        for row in manifest[task]:
            if sha256(row["path"]) != row["sha256"]:
                raise ValueError(f"Cambió una imagen; ejecuta preparar: {row['path']}")
    return manifest


def rasgos_humanos(frame, cfg=None):
    return representar(frame, cfg)


def cargar_fuentes(cfg, vision):
    manifest = cargar_manifiesto(cfg)
    signature = {"manifiesto": sha256(Path(cfg["artefactos"]) / "particiones.json"),
                 "detector": sha256(cfg.get("detector")), "reconocedor": sha256(cfg.get("reconocedor")),
                 "rostro_minimo_px": cfg["rostro_minimo_px"], "preprocesamiento": contrato(cfg)}
    # Caché local de arrays, nunca pickle de una fuente externa.
    import joblib
    cache = Path(cfg["artefactos"]) / "datos_cache.joblib"
    if cache.exists():
        saved = joblib.load(cache)
        if saved["firma"] == signature:
            print("Características cargadas de caché verificada.")
            return saved["datos"]
    result, exclusions = {}, []
    for task in ("humanos", "identidad"):
        result[task] = {}
        for split in PARTICIONES:
            rows = [r for r in manifest[task] if r["split"] == split]
            print(f"Extrayendo {task}/{split}: {len(rows)} imágenes", flush=True)
            vectors, crops, labels, paths = [], [], [], []
            human_vectors, human_crops = [], []
            for i, row in enumerate(rows):
                frame = leer_imagen(row["path"])
                if task == "identidad":
                    try:
                        vector, crop, face = vision.extraer(frame)
                    except ValueError as exc:
                        exclusions.append({**row, "motivo": str(exc)})
                        continue
                    hv, hc = rasgos_humanos(vision.recortar_rostro(frame, face), cfg)
                    human_vectors.append(hv)
                    human_crops.append(hc)
                else:
                    vector, crop = rasgos_humanos(frame, cfg)
                vectors.append(vector)
                crops.append(crop)
                labels.append(row["label"])
                paths.append(row["path"])
                if (i + 1) % 250 == 0:
                    print(f"Extracción {task}/{split}: {i + 1}/{len(rows)}", flush=True)
            result[task][split] = {"vectors": np.asarray(vectors, dtype=np.float32),
                "crops": np.asarray(crops, dtype=np.float32), "labels": np.asarray(labels), "paths": paths}
            if task == "identidad":
                result[task][split].update(human_vectors=np.asarray(human_vectors, dtype=np.float32),
                    human_crops=np.asarray(human_crops, dtype=np.float32),
                    n_excluidas=sum(r["split"] == split for r in exclusions))
    audit = {"exclusiones": exclusions, "resumen": {
        s: {p: {"incluidas": int(np.sum(result["identidad"][s]["labels"] == p)),
                 "excluidas": sum(r["split"] == s and r["label"] == p for r in exclusions)}
            for p in cfg["autorizados"]} for s in PARTICIONES}}
    (Path(cfg["artefactos"]) / "calidad_datos.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Imágenes de identidad excluidas por calidad: {len(exclusions)}. Consulta calidad_datos.json.")
    for s in PARTICIONES:
        counts = {p: audit["resumen"][s][p]["incluidas"] for p in cfg["autorizados"]}
        if any(n < (3 if s == "train" else 1) for n in counts.values()):
            raise ValueError(f"{s}: faltan rostros válidos de las cuatro personas: {counts}")
    joblib.dump({"firma": signature, "datos": result}, cache, compress=3)
    return result
