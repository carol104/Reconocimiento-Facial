import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import cv2
from algoritmos import NOMBRES, crear
from sistema.acceso import Acceso, Puerta, verificar_hardware
from sistema.config import cargar
from sistema.entrenamiento import decidir, entrenar, identificar, cargar_artefacto
from sistema.vision import PAD
from sistema.datos import cargar_dataset
from sistema.entrenar_pad import entrenar_pad


def config():
    return cargar(Path(__file__).resolve().parents[1] / "config.json")


def face(pose):
    row = np.zeros(15)
    row[:4] = [0, 0, 100, 100]
    row[4], row[6] = 20, 80
    row[8] = {"CENTRO": 50, "IZQUIERDA": 30, "DERECHA": 70}[pose]
    return row


class SeguridadTests(unittest.TestCase):
    def test_rechazo_por_parecido_margen_y_desacuerdo(self):
        people = np.array(["a", "b", "c", "d"])
        centers = np.eye(4)
        vectors = np.array([[1, 0, 0, 0], [.71, .70, 0, 0], [0, 0, 0, -1], [1, 0, 0, 0]])
        decisions, _, _ = decidir(np.array(["a", "a", "a", "b"]), vectors, centers, people, .6, .12)
        self.assertEqual(list(decisions), ["a", "__desconocido__", "__desconocido__", "__desconocido__"])

    def test_pad_ausente_denegado(self):
        self.assertEqual(PAD(config()["pad"]).evaluar(None, None), (False, 0.0))

    def test_gpio_bloqueado(self):
        cfg = config()
        with self.assertRaises(ValueError):
            verificar_hardware(cfg, {"evaluacion_basica_aprobada": True}, True)
        cfg["despliegue_validado"] = True
        with self.assertRaises(ValueError):
            verificar_hardware(cfg, {"evaluacion_basica_aprobada": True}, False)
        with self.assertRaises(ValueError):
            verificar_hardware(cfg, {"evaluacion_basica_aprobada": False}, True)

    def test_foto_estatica_no_completa_reto(self):
        cfg = config()
        gate = Acceso(cfg)
        for t in np.arange(0, 30, .15):
            allowed, _ = gate.actualizar("persona_1", True, face("CENTRO"), float(t))
            self.assertFalse(allowed)

    def test_reto_completo_enfriamiento_y_cambio_de_persona(self):
        gate = Acceso(config())
        t = 1.0
        for _ in range(5):
            gate.actualizar("persona_1", True, face("CENTRO"), t)
            t += .15
        self.assertIsNotNone(gate.reto.steps)
        steps = gate.reto.steps.copy()
        allowed = False
        for pose in steps:
            for _ in range(4):
                allowed, _ = gate.actualizar("persona_1", True, face(pose), t)
                t += .15
                if allowed:
                    break
        self.assertTrue(allowed)
        self.assertFalse(gate.actualizar("persona_1", True, face("CENTRO"), t)[0])
        gate.actualizar("persona_2", True, face("CENTRO"), t + 20)
        self.assertEqual(gate.count, 1)
        gate.actualizar(None, False, None, t + 20.1)
        self.assertIsNone(gate.person)

    def test_rele_se_apaga_ante_excepcion(self):
        from unittest.mock import Mock
        cfg = config()["rele"]
        device = Mock()
        door = Puerta(cfg, True)
        door.device = device
        with patch("sistema.acceso.time.sleep", side_effect=RuntimeError("interrumpido")):
            with self.assertRaises(RuntimeError):
                door.abrir()
        device.off.assert_called_once()
        door.__exit__()
        device.close.assert_called_once()


class AlgoritmosTests(unittest.TestCase):
    def test_seis_modelos_entrenan_y_persisten(self):
        rng = np.random.default_rng(42)
        y = np.repeat(["a", "b", "c", "d"], 4)
        X = np.eye(4)[np.repeat(np.arange(4), 4)] + rng.normal(0, .01, (16, 4))
        crops = rng.random((16, 3, 64, 64), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp:
            for name in NOMBRES:
                with self.subTest(algoritmo=name):
                    model = crear(name)
                    if name == "cnn":
                        model.epochs = 1
                    features = crops if name == "cnn" else X
                    with contextlib.redirect_stdout(io.StringIO()):
                        model.fit(features, y)
                    before = model.predict(features)
                    path = Path(temp) / f"{name}.joblib"
                    joblib.dump(model, path)
                    np.testing.assert_array_equal(before, joblib.load(path).predict(features))

    def test_entrenamiento_reporte_y_artefacto(self):
        cfg = config()
        people = cfg["autorizados"]
        with tempfile.TemporaryDirectory() as temp:
            cfg["artefactos"] = temp
            for key in ("detector", "reconocedor"):
                path = Path(temp) / key
                path.write_bytes(b"modelo-de-prueba")
                cfg[key] = str(path)
            data = {}
            for split in ("train", "val", "test"):
                vectors = np.repeat(np.eye(8, dtype=np.float32)[:4], 3, axis=0)
                labels = np.repeat(people, 3)
                if split != "train":
                    vectors = np.vstack([vectors, np.eye(8, dtype=np.float32)[4]])
                    labels = np.concatenate([labels, ["__desconocido__"]])
                data[split] = {"vectors": vectors, "labels": labels,
                               "crops": np.zeros((len(labels), 3, 64, 64), dtype=np.float32),
                               "paths": [f"{split}/{i}" for i in range(len(labels))]}
            with contextlib.redirect_stdout(io.StringIO()):
                report = entrenar("knn", data, cfg)
            self.assertEqual(report["metricas"]["test"]["aceptacion_desconocidos"], 0)
            self.assertEqual(report["brecha_test_menos_train"], 0)
            artifact = cargar_artefacto("knn", cfg)
            self.assertTrue(artifact["evaluacion_basica_aprobada"])
            self.assertEqual(identificar(artifact, data["test"]["vectors"][0], None)[0], people[0])
            cfg["coseno_minimo"] = .7
            with self.assertRaises(ValueError):
                cargar_artefacto("knn", cfg)

    def test_error_cascada_cuenta_rechazos_y_exclusiones(self):
        from unittest.mock import Mock
        cfg = config()
        with tempfile.TemporaryDirectory() as temp:
            cfg["artefactos"] = temp
            for key in ("detector", "reconocedor"):
                path = Path(temp) / key
                path.write_bytes(b"test")
                cfg[key] = str(path)
            data = {}
            for split in ("train", "val", "test"):
                vectors = np.repeat(np.eye(4, dtype=np.float32), 3, axis=0)
                data[split] = {"vectors": vectors, "labels": np.repeat(cfg["autorizados"], 3),
                    "paths": [str(i) for i in range(12)], "human_vectors": vectors,
                    "n_excluidas": 2}
            human_model = Mock()
            human_model.predict.return_value = np.repeat("objeto", 12)
            with contextlib.redirect_stdout(io.StringIO()):
                report = entrenar("knn", data, cfg, humanos={"modelo": human_model})
            for split in ("train", "val", "test"):
                row = report["metricas"][split]
                self.assertEqual(row["error_clasificacion_conocidos"], 0)
                self.assertAlmostEqual(row["error_acceso_incluyendo_calidad"], 2 / 14)
                self.assertEqual(row["error_cascada_incluyendo_calidad"], 1)


class DatosYPADTests(unittest.TestCase):
    @staticmethod
    def vision_simulada():
        from unittest.mock import Mock
        vision = Mock()
        vision.extraer.return_value = (np.ones(8, dtype=np.float32) / np.sqrt(8),
                                       np.zeros((3, 64, 64), dtype=np.float32), face("CENTRO"))
        return vision

    def test_particiones_y_duplicados(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as temp:
            cfg["dataset"] = temp
            value = 0
            for split in ("train", "val", "test"):
                for person in cfg["autorizados"]:
                    directory = Path(temp) / split / person / f"sesion_{split}"
                    directory.mkdir(parents=True)
                    for i in range(3):
                        value += 1
                        cv2.imwrite(str(directory / f"{i}.png"), np.full((120, 120, 3), value, dtype=np.uint8))
            data = cargar_dataset(cfg, self.vision_simulada())
            self.assertEqual(len(data["train"]["labels"]), 12)
            source = Path(temp) / "train" / cfg["autorizados"][0] / "sesion_train" / "0.png"
            target = Path(temp) / "test" / cfg["autorizados"][0] / "sesion_test" / "duplicada.png"
            target.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicada"):
                cargar_dataset(cfg, self.vision_simulada())
            target.unlink()
            session = target.parent
            session.rename(session.with_name("sesion_train"))
            with self.assertRaisesRegex(ValueError, "Sesión compartida"):
                cargar_dataset(cfg, self.vision_simulada())

    def test_pad_entrena_calibra_persiste_y_predice(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as temp:
            cfg["artefactos"] = str(Path(temp) / "salida")
            root = Path(temp) / "datos"
            value = 0
            for split in ("train", "val", "test"):
                for kind in ("ataque", "real"):
                    directory = root / split / kind / f"sesion_{split}"
                    directory.mkdir(parents=True)
                    for i in range(3):
                        value += 1
                        image = np.full((120, 120, 3), value * 5, dtype=np.uint8)
                        cv2.imwrite(str(directory / f"{i}.png"), image)
            with contextlib.redirect_stdout(io.StringIO()):
                report = entrenar_pad(cfg, root, self.vision_simulada(), epochs=1)
            cfg["pad"]["modelo"] = str(Path(cfg["artefactos"]) / "pad.joblib")
            cfg["pad"]["umbral"] = report["umbral"]
            detector = PAD(cfg["pad"])
            passed, score = detector.evaluar(image, face("CENTRO"))
            self.assertTrue(0 <= score <= 1)
            self.assertEqual(passed, score >= report["umbral"])
            self.assertIn("APCER_ataques_aceptados", report["metricas"]["test"])

    def test_modelos_reales_no_detectan_en_imagen_vacia(self):
        from sistema.vision import Vision
        cfg = config()
        if not Path(cfg["detector"]).exists() or not Path(cfg["reconocedor"]).exists():
            self.skipTest("Modelos públicos aún no descargados")
        vision = Vision(cfg)
        with self.assertRaisesRegex(ValueError, "exactamente un rostro"):
            vision.extraer(np.zeros((480, 640, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
