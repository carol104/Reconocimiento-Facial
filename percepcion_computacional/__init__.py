"""Fase 1: Percepción Computacional.
Incluye visión por computador (Haar / HOG), filtro biológico (humano vs objeto)
y captura/renderizado visual con cámara en tiempo real.
"""
from percepcion_computacional.vision import Vision, FaceDetectorHaar, FeatureExtractorHOG
from percepcion_computacional.humanos import es_humano, entrenar_humanos, cargar_humanos
from percepcion_computacional.camara import ejecutar

__all__ = [
    "Vision", "FaceDetectorHaar", "FeatureExtractorHOG",
    "es_humano", "entrenar_humanos", "cargar_humanos",
    "ejecutar"
]
