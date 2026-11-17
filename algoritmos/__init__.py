"""Clasificadores intercambiables para la misma tarea de identificación."""

NOMBRES = ("ann", "cnn", "svm", "knn", "random_forest", "arbol_decision")


def crear(nombre, seed=42):
    from importlib import import_module

    if nombre not in NOMBRES:
        raise ValueError(f"Algoritmo no válido: {nombre}")
    return import_module(f"algoritmos.{nombre}").crear(seed)
