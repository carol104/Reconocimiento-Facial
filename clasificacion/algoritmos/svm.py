"""Máquina de vectores de soporte con kernel RBF."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def crear(seed=42):
    return make_pipeline(StandardScaler(), SVC(C=5, kernel="rbf", random_state=seed))
