from pathlib import Path
import cv2
import numpy as np
from sistema.vision import sha256

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}


def cargar_dataset(cfg, vision):
    """Particiones explícitas por sesión; se rechazan duplicados entre particiones."""
    root = Path(cfg["dataset"])
    result, hashes, pixels, sessions = {}, {}, {}, {}
    for split in ("train", "val", "test"):
        vectors, crops, labels, paths = [], [], [], []
        directory = root / split
        if not directory.is_dir():
            raise ValueError(f"Falta partición {directory}; consulta README.md.")
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in EXTENSIONES or not path.is_file():
                continue
            parts = path.relative_to(directory).parts
            unknown = parts[0] == "desconocidos"
            # conocidos: persona/sesion/imagen; desconocidos: desconocidos/persona/sesion/imagen
            expected = 4 if unknown else 3
            if len(parts) != expected:
                raise ValueError(f"Estructura inválida: {path}")
            person = parts[1] if unknown else parts[0]
            session = parts[2] if unknown else parts[1]
            if unknown:
                if split == "train" or person in cfg["autorizados"]:
                    raise ValueError(f"Desconocido mal ubicado: {path}")
                label = "__desconocido__"
            else:
                if person not in cfg["autorizados"]:
                    raise ValueError(f"Identidad fuera de la lista: {person}")
                label = person
            session_key = (person, session)
            if session_key in sessions and sessions[session_key] != split:
                raise ValueError(f"Sesión compartida entre particiones: {session_key}")
            sessions[session_key] = split
            digest = sha256(path)
            if digest in hashes:
                raise ValueError(f"Imagen duplicada: {path} y {hashes[digest]}")
            hashes[digest] = str(path)
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Imagen ilegible: {path}")
            import hashlib
            pixel_hash = hashlib.sha256(frame.tobytes()).hexdigest()
            if pixel_hash in pixels:
                raise ValueError(f"Píxeles duplicados: {path} y {pixels[pixel_hash]}")
            pixels[pixel_hash] = str(path)
            try:
                vector, crop, _ = vision.extraer(frame)
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc
            vectors.append(vector)
            crops.append(crop)
            labels.append(label)
            paths.append(str(path.relative_to(root)))
        counts = {p: labels.count(p) for p in cfg["autorizados"]}
        if any(n < (3 if split == "train" else 1) for n in counts.values()):
            raise ValueError(f"{split}: faltan muestras de las cuatro personas: {counts}")
        result[split] = {"vectors": np.asarray(vectors, dtype=np.float32),
                         "crops": np.asarray(crops, dtype=np.float32),
                         "labels": np.asarray(labels), "paths": paths}
    return result
