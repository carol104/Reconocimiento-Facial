import contextlib
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import joblib
import numpy as np

from sistema.preprocesamiento import preprocesar, representar, descriptor, entrada_entrenamiento
from sistema.vision import Vision


class BackendTests(unittest.TestCase):
    def test_entrenamiento_y_camara_comparten_representacion(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
        box = np.array([10, 20, 120, 120], dtype=np.float32)
        for method in ("ninguno", "sobel", "prewitt", "roberts", "laplaciano"):
            cfg = {"preprocesamiento": {"bordes": method}, "rostro_minimo_px": 100}
            vision = Vision(cfg)
            vision.detector = Mock()
            vision.detector.detect.return_value = (1, box[None])
            vector, crop, face = vision.extraer(frame)
            online_vector, online_crop = vision.representar_rostro(frame, face)
            np.testing.assert_array_equal(crop, online_crop)
            np.testing.assert_array_equal(vector, online_vector)
            expected_vector, expected_crop = representar(frame[20:140, 10:130], cfg)
            np.testing.assert_array_equal(crop, expected_crop)
            np.testing.assert_array_equal(vector, expected_vector)
            self.assertEqual(crop.shape, (64, 64, 1))
            self.assertTrue(np.all((crop >= 0) & (crop <= 1)))

    def test_aumentos_comunes_no_modifican_originales(self):
        rng = np.random.default_rng(3)
        crops = rng.random((4, 64, 64, 1), dtype=np.float32)
        vectors = np.array([descriptor(c) for c in crops])
        part = {"crops": crops.copy(), "vectors": vectors, "labels": np.array(["a", "b", "c", "d"])}
        cfg = {"seed": 42, "aumentos_por_imagen": 2}
        images, labels = entrada_entrenamiento(part, cfg, "cnn")
        hogs, classical_labels = entrada_entrenamiento(part, cfg, "svm")
        np.testing.assert_array_equal(labels, classical_labels)
        np.testing.assert_array_equal(hogs, np.array([descriptor(c) for c in images]))
        np.testing.assert_array_equal(part["crops"], crops)
        np.testing.assert_array_equal(images, entrada_entrenamiento(part, cfg, "cnn")[0])
        self.assertEqual(len(images), 12)

    def test_keras_arquitectura_validacion_y_persistencia(self):
        from clasificacion.algoritmos.cnn import CNN
        from tensorflow import keras
        rng = np.random.default_rng(12)
        X = rng.random((8, 64, 64, 1), dtype=np.float32)
        y = np.repeat(["a", "b"], 4)
        model = CNN(epochs=2)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X, y, validation_data=(X[:2], y[:2]))
        self.assertEqual(len(model.history_["val_loss"]), 2)
        conv = [l for l in model.model.layers if isinstance(l, keras.layers.Conv2D)]
        self.assertEqual(len(conv), 2)
        self.assertTrue(all(l.padding == "valid" and l.kernel_size == (3, 3) and l.strides == (1, 1) for l in conv))
        probabilities = model.predict_proba(X)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(probabilities, model.predict_proba(X))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cnn.joblib"
            joblib.dump(model, path)
            np.testing.assert_allclose(probabilities, joblib.load(path).predict_proba(X), atol=1e-6)
        with self.assertRaisesRegex(ValueError, "PyTorch"):
            CNN().__setstate__({"weights": {}})

    def test_esp32_conserva_protocolo_y_puerto_explicito(self):
        from clasificacion.arduino import ArduinoControlador
        connection = Mock()
        connection.is_open = True
        with patch("clasificacion.arduino.serial.Serial", return_value=connection) as serial, patch("clasificacion.arduino.time.sleep"), patch("clasificacion.arduino.serial.tools.list_ports.comports", return_value=[]):
            controller = ArduinoControlador({"esp32": {"puerto": "COM7", "baudios": 115200, "activo": True}})
            controller.enviar_estado(True)
            controller.enviar_estado(True)
            controller.enviar_estado(False)
            controller.cerrar()
        serial.assert_called_once_with("COM7", 115200, timeout=1)
        self.assertEqual([c.args[0] for c in connection.write.call_args_list], [b"1\n", b"0\n", b"0\n"])

    def test_cli_admite_usb_y_stream_esp32(self):
        import app
        parser = app.parser()
        self.assertEqual(parser.parse_args(["ejecutar", "--camara", "1"]).camara, 1)
        url = "http://192.168.1.2:81/stream"
        self.assertEqual(parser.parse_args(["ejecutar", "--camara", url]).camara, url)

    def test_descarga_cli_configura_solo_yunet(self):
        import app
        import json
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
            cfg["artefactos"] = "salida"
            cfg["detector"] = None
            cfg["reconocedor"] = None
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            with patch("sys.argv", ["app.py", "--config", str(config_path), "descargar-modelos"]), patch("sistema.modelos.urllib.request.urlopen", return_value=io.BytesIO(b"x" * 100001)) as request:
                app.main()
            updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertTrue(Path(updated["detector"]).is_file())
            self.assertIsNone(updated["reconocedor"])
            self.assertEqual(request.call_count, 1)
            self.assertIn("face_detection_yunet", request.call_args.args[0])

    def test_recomendacion_usa_validacion_y_no_test(self):
        import app
        import json
        from reconocimiento.reportes import comparar
        reports = []
        for name, val_error, test_error in (("svm", .1, .9), ("knn", .9, .1)):
            stats = {split: {"muestras": 10, "error_clasificacion_conocidos": error,
                            "error_acceso": error}
                     for split, error in (("train", 0), ("val", val_error), ("test", test_error))}
            reports.append({"tarea": "identidad", "algoritmo": name, "metricas": stats, "inferencia_host_ms": 1})
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            comparar(reports, {"artefactos": temp})
            selected = json.loads((Path(temp) / "seleccion_modelo.json").read_text(encoding="utf-8"))
            self.assertEqual(selected["algoritmo"], "svm")

    def test_cnn_entrena_ambas_tareas_y_rechaza_otro_preprocesamiento(self):
        import app
        from clasificacion.entrenamiento import entrenar, cargar_artefacto
        from percepcion_computacional.humanos import entrenar_humanos, cargar_humanos
        rng = np.random.default_rng(40)
        crops = rng.random((12, 64, 64, 1), dtype=np.float32)
        vectors = np.array([descriptor(c) for c in crops])
        labels = np.repeat(["a", "b", "c", "d"], 3)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            cfg = {"seed": 42, "aumentos_por_imagen": 1, "autorizados": ["a", "b", "c", "d"],
                "artefactos": temp, "coseno_minimo": .6, "margen_minimo": .015,
                "detector": None, "reconocedor": None, "cnn": {"epochs": 1}}
            part = {"crops": crops, "vectors": vectors, "labels": labels, "paths": list(map(str, range(12)))}
            data = {split: part for split in ("train", "val", "test")}
            report = entrenar("cnn", data, cfg)
            self.assertEqual(report["muestras_ajuste"], 24)
            self.assertEqual(report["metricas"]["test"]["muestras"], 12)
            self.assertEqual(cargar_artefacto("cnn", cfg)["modelo"].predict(crops).shape, (12,))
            human_part = {**part, "labels": np.repeat(["humano", "objeto"], 6)}
            human_report, artifact = entrenar_humanos("cnn", {s: human_part for s in data}, cfg)
            self.assertEqual(human_report["muestras_ajuste"], 24)
            self.assertEqual(cargar_humanos("cnn", cfg)["version"], 2)
            cfg["preprocesamiento"] = {"bordes": "sobel"}
            with self.assertRaisesRegex(ValueError, "incompatible"):
                cargar_artefacto("cnn", cfg)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                cargar_humanos("cnn", cfg)


if __name__ == "__main__":
    unittest.main()
