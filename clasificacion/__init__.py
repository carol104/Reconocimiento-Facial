"""Fase 3: Clasificación y Control de Acceso.
Incluye los algoritmos de clasificación (SVM, KNN, ANN, CNN, Random Forest, Árbol de Decisión),
el motor de entrenamiento, la lógica de permitir/denegar y la interfaz con Arduino.
"""
from clasificacion.algoritmos import CREAR, NOMBRES
from clasificacion.entrenamiento import entrenar, cargar_artefacto, identificar
from clasificacion.acceso import Acceso, Puerta
from clasificacion.arduino import ArduinoControlador

__all__ = [
    "CREAR", "NOMBRES",
    "entrenar", "cargar_artefacto", "identificar",
    "Acceso", "Puerta",
    "ArduinoControlador"
]
