import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import importlib

import cv2
import numpy as np

from sistema.preparacion import preparar_fuentes, cargar_manifiesto, cargar_fuentes
from sistema.humanos import entrenar_humanos, cargar_humanos, es_humano
from sistema.reportes import comparar


class FuentesTests(unittest.TestCase):
    def crear_fuentes(self, root):
        rng = np.random.default_rng(12)
        cfg = {"seed": 42, "aumentos_por_imagen": 0, "autorizados": ["a", "b", "c", "d"],
            "artefactos": str(root / "salida"), "rostro_minimo_px": 100,
            "fuentes": {"webcam": str(root / "webcam"), "humanos": str(root / "humanos"),
                "sesiones": {"train": ["s1", "s2", "s3"], "val": ["s4"], "test": ["s5"]}}}
        for p in cfg["autorizados"]:
            for s in ("s1", "s2", "s3", "s4", "s5"):
                directory = root / "webcam" / p / s
                directory.mkdir(parents=True)
                for i in range(3):
                    cv2.imwrite(str(directory / f"{i}.png"), rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        for folder in ("male_faces", "female_faces", "Objects"):
            directory = root / "humanos" / folder
            directory.mkdir(parents=True)
            for i in range(10):
                cv2.imwrite(str(directory / f"{i}.png"), rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        for key in ("detector", "reconocedor"):
            path = root / key
            path.write_bytes(b"fake-model")
            cfg[key] = str(path)
        return cfg

    def test_sesiones_reproducibles_duplicados_y_conflictos(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            cfg = self.crear_fuentes(root)
            source = root / "humanos" / "male_faces" / "0.png"
            (source.parent / "duplicado.png").write_bytes(source.read_bytes())
            conflict = root / "humanos" / "Objects" / "conflicto.png"
            conflict.write_bytes(source.read_bytes())
            first = preparar_fuentes(cfg)
            self.assertEqual(first, preparar_fuentes(cfg))
            self.assertEqual(len(first["duplicados_omitidos"]), 1)
            self.assertEqual(len(first["conflictos_etiquetas"]), 1)
            self.assertNotIn(str(source.resolve()), [r["path"] for r in first["humanos"]])
            self.assertNotIn(str(conflict.resolve()), [r["path"] for r in first["humanos"]])
            for row in first["identidad"]:
                self.assertIn(row["sesion"], cfg["fuentes"]["sesiones"][row["split"]])
            for task in ("identidad", "humanos"):
                hashes = [r["pixeles_sha256"] for r in first[task]]
                self.assertEqual(len(hashes), len(set(hashes)))
            row = first["humanos"][0]
            Path(row["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Cambió una imagen"):
                cargar_manifiesto(cfg)

    def test_sesion_repetida_y_duplicado_entre_sesiones_rechazados(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            cfg = self.crear_fuentes(root)
            cfg["fuentes"]["sesiones"]["test"] = ["s4"]
            with self.assertRaisesRegex(ValueError, "Sesión compartida"):
                preparar_fuentes(cfg)
            cfg["fuentes"]["sesiones"]["test"] = ["s5"]
            (root / "webcam/a/s5/0.png").write_bytes((root / "webcam/a/s1/0.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "Duplicado entre"):
                preparar_fuentes(cfg)

    def test_mezclado_por_persona_reproducible_sin_duplicados(self):
        # Inicializa los paquetes en el mismo orden que el punto de entrada CLI.
        import app

        for module in ("sistema.preparacion", "reconocimiento.preparacion"):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(temp)
                cfg = self.crear_fuentes(root)
                cfg["fuentes"]["reparto_identidad"] = "mezclado"
                prepare = importlib.import_module(module).preparar_fuentes
                first = prepare(cfg)
                self.assertEqual(first, prepare(cfg))
                for person in cfg["autorizados"]:
                    rows = [r for r in first["identidad"] if r["label"] == person]
                    self.assertEqual(len(rows), 15)
                    for split, count in (("train", 9), ("val", 3), ("test", 3)):
                        part = [r for r in rows if r["split"] == split]
                        self.assertEqual(len(part), count)
                    self.assertTrue(any(r["sesion"] not in cfg["fuentes"]["sesiones"][r["split"]] for r in rows))
                source = root / "webcam/a/s1/0.png"
                (root / "webcam/a/s5/duplicado.png").write_bytes(source.read_bytes())
                deduplicated = prepare(cfg)
                self.assertEqual(first["identidad"], deduplicated["identidad"])
                self.assertEqual(len(deduplicated["duplicados_omitidos"]), 1)
                (root / "webcam/b/s1/duplicado.png").write_bytes(source.read_bytes())
                with self.assertRaisesRegex(ValueError, "Duplicado entre"):
                    prepare(cfg)

    def test_exclusiones_auditadas_cache_y_modelo_binario(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            cfg = self.crear_fuentes(root)
            preparar_fuentes(cfg)
            vision = Mock()
            vector = np.ones(8, dtype=np.float32) / np.sqrt(8)
            crop = np.zeros((64, 64, 1), dtype=np.float32)
            calls = 0

            def extract(frame):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise ValueError("Rostro pequeño")
                return vector, crop, None

            vision.extraer.side_effect = extract
            vision.recortar_rostro.side_effect = lambda frame, face: frame
            data = cargar_fuentes(cfg, vision)
            audit = json.loads((root / "salida/calidad_datos.json").read_text(encoding="utf-8"))
            self.assertEqual(len(audit["exclusiones"]), 1)
            self.assertEqual(data["identidad"]["train"]["n_excluidas"], 1)
            vision.extraer.reset_mock()
            cached = cargar_fuentes(cfg, vision)
            vision.extraer.assert_not_called()
            np.testing.assert_array_equal(cached["humanos"]["train"]["vectors"], data["humanos"]["train"]["vectors"])
            report, artifact = entrenar_humanos("knn", data["humanos"], cfg)
            self.assertEqual(set(report["metricas"]), {"train", "val", "test"})
            loaded = cargar_humanos("knn", cfg)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            self.assertEqual(es_humano(artifact, image), es_humano(loaded, image))
            comparar([report], cfg)
            self.assertTrue((root / "salida/comparacion.csv").exists())


if __name__ == "__main__":
    unittest.main()
