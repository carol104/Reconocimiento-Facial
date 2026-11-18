import json
from pathlib import Path


def cargar(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    people = cfg["autorizados"]
    if (len(people) != 4 or len(set(people)) != 4
            or any(not isinstance(p, str) or not p.strip() or p in ("desconocidos", "__desconocido__", ".", "..")
                   or "/" in p or "\\" in p for p in people)):
        raise ValueError("Configura exactamente cuatro identificadores únicos de personas.")
    for key in ("dataset", "artefactos"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    for key in ("detector", "reconocedor"):
        if cfg.get(key):
            cfg[key] = str((path.parent / cfg[key]).resolve())
        else:
            cfg[key] = None
    if "fuentes" in cfg:
        for key in ("webcam", "humanos"):
            cfg["fuentes"][key] = str((path.parent / cfg["fuentes"][key]).resolve())
    if cfg["pad"].get("modelo"):
        cfg["pad"]["modelo"] = str((path.parent / cfg["pad"]["modelo"]).resolve())
    if not 0 < cfg["coseno_minimo"] < 1 or not 0 < cfg["margen_minimo"] < 1:
        raise ValueError("Umbrales de identidad fuera de rango.")
    if not 0 < cfg["pad"]["umbral"] <= 1:
        raise ValueError("Umbral PAD fuera de rango.")
    if cfg["pad"]["indice_real"] not in (0, 1) or any(
            not isinstance(cfg["pad"][k], int) or not 1 <= cfg["pad"][k] <= 2048
            for k in ("alto", "ancho")):
        raise ValueError("PAD necesita índice real 0/1 y dimensiones de entrada válidas.")
    if cfg["frames_confirmacion"] < 3 or not 0 < cfg["rele"]["segundos"] <= 10:
        raise ValueError("Confirmación >= 3 fotogramas; apertura entre 0 y 10 segundos.")
    if not 1 <= cfg["rele"]["enfriamiento"] <= 3600 or not 3 <= cfg["reto_segundos"] <= 120:
        raise ValueError("Enfriamiento entre 1 y 3600 s y reto entre 3 y 120 s.")
    if not isinstance(cfg["despliegue_validado"], bool) or not isinstance(cfg["rele"]["activo_alto"], bool):
        raise ValueError("despliegue_validado y activo_alto deben ser booleanos JSON.")
    return cfg
