"""Vecinos cercanos con distancia coseno y voto ponderado."""
from sklearn.neighbors import KNeighborsClassifier


def crear(seed=42):
    return KNeighborsClassifier(n_neighbors=3, weights="distance", metric="cosine")
