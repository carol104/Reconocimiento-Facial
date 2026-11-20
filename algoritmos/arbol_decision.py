"""Árbol de decisión con regularización de profundidad y hojas."""
from sklearn.tree import DecisionTreeClassifier


def crear(seed=42):
    return DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=2,
                                  class_weight="balanced", random_state=seed)
