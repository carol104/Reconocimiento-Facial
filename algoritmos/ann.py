"""Red neuronal artificial multicapa sobre descriptores faciales SFace."""
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def crear(seed=42):
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu", alpha=0.01,
        max_iter=500, random_state=seed))
