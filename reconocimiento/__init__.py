"""Fase 2: Reconocimiento Facial.
Incluye preparación de particiones de autorizados, carga de datos y
generación de reportes de errores de entrenamiento y testeo.
"""
from reconocimiento.config import cargar as cargar_config
from reconocimiento.datos import cargar_dataset
from reconocimiento.preparacion import preparar_fuentes, cargar_fuentes
from reconocimiento.reportes import comparar

__all__ = [
    "cargar_config", "cargar_dataset",
    "preparar_fuentes", "cargar_fuentes",
    "comparar"
]
