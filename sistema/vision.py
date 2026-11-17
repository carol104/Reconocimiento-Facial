from pathlib import Path
import hashlib
import cv2
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Vision:
    def __init__(self, cfg):
        for key in ("detector", "reconocedor"):
            if not Path(cfg[key]).is_file():
                raise FileNotFoundError(f"Falta {cfg[key]}. Ejecuta: python app.py descargar-modelos")
        self.detector = cv2.FaceDetectorYN.create(cfg["detector"], "", (320, 320), 0.9)
        self.recognizer = cv2.FaceRecognizerSF.create(cfg["reconocedor"], "")
        self.min_size = cfg["rostro_minimo_px"]

    def extraer(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) != 1:
            raise ValueError("Se requiere exactamente un rostro visible.")
        face = faces[0]
        x, y, fw, fh = face[:4]
        if min(fw, fh) < self.min_size or x < 0 or y < 0 or x + fw > w or y + fh > h:
            raise ValueError("Rostro pequeño o fuera del encuadre.")
        aligned = self.recognizer.alignCrop(frame, face)
        vector = self.recognizer.feature(aligned).flatten()
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm < 1e-8:
            raise ValueError("Descriptor facial inválido.")
        # CNN usa imágenes, los otros cinco modelos usan el mismo descriptor.
        crop = cv2.resize(aligned, (64, 64))
        tensor = np.ascontiguousarray(crop[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255
        return vector / norm, tensor, face


def recorte_pad(frame, face):
    x, y, w, h = face[:4]
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(x - w * .25)), max(0, int(y - h * .25))
    x2, y2 = min(width, int(x + w * 1.25)), min(height, int(y + h * 1.25))
    return frame[y1:y2, x1:x2]


def tensor_pad(frame, face):
    crop = cv2.resize(recorte_pad(frame, face), (64, 64))
    return np.ascontiguousarray(crop[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255


class PAD:
    """PAD binario: CNN propia entrenada o modelo ONNX externo compatible."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.net = None
        self.local = None
        if cfg.get("modelo"):
            if Path(cfg["modelo"]).suffix == ".joblib":
                import joblib
                artifact = joblib.load(cfg["modelo"])
                if artifact["tipo"] != "pad_cnn_v1" or cfg["umbral"] != artifact["umbral"]:
                    raise ValueError("Modelo PAD incompatible o umbral diferente del calibrado.")
                self.local = artifact["modelo"]
            else:
                self.net = cv2.dnn.readNetFromONNX(cfg["modelo"])
        if cfg["salida"] not in ("logits", "probabilidades"):
            raise ValueError("Salida PAD debe ser logits o probabilidades.")

    def evaluar(self, frame, face):
        if self.local is not None:
            index = list(self.local.classes_).index("real")
            score = float(self.local.predict_proba(tensor_pad(frame, face)[None])[0, index])
            return np.isfinite(score) and score >= self.cfg["umbral"], score
        if self.net is None:
            return False, 0.0
        # El PAD recibe el rostro sin alinear y contexto 1.5x.
        blob = cv2.dnn.blobFromImage(recorte_pad(frame, face),
            scalefactor=self.cfg["escala"], size=(self.cfg["ancho"], self.cfg["alto"]),
            mean=tuple(self.cfg["media"]), swapRB=self.cfg["rgb"], crop=False)
        self.net.setInput(blob)
        out = np.asarray(self.net.forward()).reshape(-1)
        if out.size != 2 or not np.isfinite(out).all():
            raise ValueError("PAD requiere salida binaria finita [falso, real] configurable.")
        if self.cfg["salida"] == "logits":
            out = np.exp(out - out.max())
            out /= out.sum()
        elif np.any(out < 0) or np.any(out > 1) or not np.isclose(out.sum(), 1, atol=.01):
            raise ValueError("Probabilidades PAD inválidas.")
        score = float(out[self.cfg["indice_real"]])
        return score >= self.cfg["umbral"], score
