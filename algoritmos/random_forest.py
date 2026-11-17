"""Bosque aleatorio de árboles de clasificación."""
from sklearn.ensemble import RandomForestClassifier


def crear(seed=42):
    return RandomForestClassifier(n_estimators=200, max_depth=15,
                                  class_weight="balanced", random_state=seed, n_jobs=-1)
