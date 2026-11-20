"""Descarga del detector YuNet; los clasificadores se entrenan desde cero."""
import json
from pathlib import Path
import urllib.request
from sistema.vision import sha256

BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELOS = {
    "detector": "face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


def descargar(cfg):
    manifest = {}
    for key, source in MODELOS.items():
        path = Path(cfg.get(key) or Path(cfg["artefactos"]).parent / "modelos" / Path(source).name)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = BASE + source
        if not path.exists():
            temp = path.with_suffix(".download")
            try:
                print(f"Descargando {url}")
                with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                if temp.stat().st_size < 100_000:
                    raise ValueError("Descarga incompleta o puntero Git LFS en lugar de ONNX.")
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        manifest[key] = {"url": url, "sha256": sha256(path)}
        cfg[key] = str(path.resolve())
    output = Path(cfg["artefactos"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "modelos.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Modelos disponibles. Revisa sus licencias en OpenCV Zoo.")
    return cfg["detector"]
