"""Contrato común de imágenes para entrenamiento e inferencia, sin pesos aprendidos."""
import cv2
import numpy as np
from time import perf_counter


def contrato(cfg=None):
    options = (cfg or {}).get("preprocesamiento", {})
    edges = options.get("bordes", "ninguno")
    if edges not in ("ninguno", "sobel", "prewitt", "roberts", "laplaciano"):
        raise ValueError("Filtro de bordes no válido.")
    return {"version": 4, "tamano": 64, "canales": 1,
            "suavizado": "gaussiano_3x3", "bordes": edges, "escala": "0_1"}


def preprocesar(frame, cfg=None):
    if frame is None or frame.size == 0:
        raise ValueError("Imagen vacía.")
    image = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    method = contrato(cfg)["bordes"]
    if method != "ninguno":
        if method == "laplaciano":
            edge = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        elif method == "sobel":
            edge = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        else:
            kernel = (np.array([[1, 0, -1]] * 3, np.float32) if method == "prewitt"
                      else np.array([[1, 0], [0, -1]], np.float32))
            other = kernel.T if method == "prewitt" else np.array([[0, 1], [-1, 0]], np.float32)
            edge = cv2.magnitude(cv2.filter2D(gray, cv2.CV_32F, kernel), cv2.filter2D(gray, cv2.CV_32F, other))
        edge = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gray = cv2.addWeighted(gray, .8, edge, .2, 0)
    return np.ascontiguousarray(gray[..., None], dtype=np.float32) / 255.0


def descriptor(tensor):
    gray = np.rint(np.clip(tensor[..., 0], 0, 1) * 255).astype(np.uint8)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    vector = hog.compute(gray).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def representar(frame, cfg=None):
    tensor = preprocesar(frame, cfg)
    return descriptor(tensor), tensor


def entrada_entrenamiento(part, cfg, nombre):
    """Mismos aumentos/semilla para todos; nunca modifica validación o prueba."""
    key = "crops" if nombre == "cnn" else "vectors"
    count = cfg.get("aumentos_por_imagen", 1)
    if not isinstance(count, int) or not 0 <= count <= 10:
        raise ValueError("aumentos_por_imagen debe estar entre 0 y 10.")
    if not count:
        return part[key], part["labels"]
    rng = np.random.default_rng(cfg["seed"])
    augmented = []
    for _ in range(count):
        for crop in part["crops"]:
            if crop.shape != (64, 64, 1):
                raise ValueError("Caché antigua: vuelve a extraer las características.")
            image = crop[..., 0]
            matrix = cv2.getRotationMatrix2D((31.5, 31.5), rng.uniform(-8, 8), rng.uniform(.95, 1.05))
            matrix[:, 2] += rng.uniform(-2, 2, 2)
            image = cv2.warpAffine(image, matrix, (64, 64), borderMode=cv2.BORDER_REFLECT_101)
            if rng.random() < .5:
                image = cv2.flip(image, 1)
            image = np.clip(image * rng.uniform(.9, 1.1) + rng.uniform(-.03, .03), 0, 1)
            tensor = np.ascontiguousarray(image[..., None], dtype=np.float32)
            augmented.append(tensor if nombre == "cnn" else descriptor(tensor))
    return (np.concatenate([part[key], np.asarray(augmented)], axis=0), np.tile(part["labels"], count + 1))


def latencia_modelo(model, X):
    """Mediana de inferencias individuales en el host, sin cámara ni preprocesamiento."""
    model.predict(X[:1])  # Calentamiento fuera de la medición.
    samples = []
    for i in range(min(10, len(X))):
        start = perf_counter()
        model.predict(X[i:i + 1])
        samples.append((perf_counter() - start) * 1000)
    return float(np.median(samples))
