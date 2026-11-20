"""CNN Keras entrenada desde cero; imágenes NHWC 64x64 de un canal."""
import numpy as np


def red(n_clases, dropout=.3, l2=.001):
    from tensorflow import keras
    return keras.Sequential([
        keras.layers.Input(shape=(64, 64, 1)),
        keras.layers.Conv2D(16, 3, strides=1, padding="valid", activation="relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(32, 3, strides=1, padding="valid", activation="relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(dropout), keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.L2(l2)),
        keras.layers.Dense(n_clases, activation="softmax")])


class CNN:
    def __init__(self, seed=42, epochs=30, dropout=.3, l2=.001, patience=6):
        self.seed, self.epochs = seed, epochs
        self.dropout, self.l2, self.patience = dropout, l2, patience

    @staticmethod
    def comprobar(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 4 or X.shape[1:] != (64, 64, 1) or not np.isfinite(X).all():
            raise ValueError("CNN requiere imágenes NHWC (N,64,64,1) finitas. Reentrena con la nueva caché.")
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("CNN requiere píxeles normalizados entre 0 y 1.")
        return X

    def fit(self, X, y, validation_data=None):
        from tensorflow import keras
        keras.utils.set_random_seed(self.seed)
        X = self.comprobar(X)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.model = red(len(self.classes_), self.dropout, self.l2)
        self.model.compile(optimizer=keras.optimizers.Adam(.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.history_ = {"loss": [], "accuracy": [], "val_loss": []}
        rng = np.random.default_rng(self.seed)
        best, stale, weights = float("inf"), 0, None
        for epoch in range(self.epochs):
            self.model.reset_metrics()
            indices = rng.permutation(len(X))
            for start in range(0, len(X), 16):
                batch = indices[start:start + 16]
                result = self.model.train_on_batch(X[batch], encoded[batch], return_dict=True)
            loss = float(result["loss"])
            self.history_["loss"].append(loss)
            self.history_["accuracy"].append(float(result["accuracy"]))
            print(f"CNN época {epoch + 1}/{self.epochs}: loss={loss:.6f}", flush=True)
            if validation_data is not None:
                vx, vy = validation_data
                probs = self.predict_proba(vx)
                indices_val = np.array([list(self.classes_).index(v) for v in vy])
                val_loss = float(-np.log(np.maximum(probs[np.arange(len(vy)), indices_val], 1e-7)).mean())
                self.history_["val_loss"].append(val_loss)
                if val_loss < best:
                    best, stale, weights = val_loss, 0, self.model.get_weights()
                    self.best_epoch_ = epoch + 1
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
        if weights is not None:
            self.model.set_weights(weights)
        self.loss_history_ = self.history_["loss"]
        return self

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def predict_proba(self, X):
        X = self.comprobar(X)
        return np.concatenate([self.model(X[i:i + 32], training=False).numpy() for i in range(0, len(X), 32)])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["keras_json"] = self.model.to_json()
        state["keras_weights"] = self.model.get_weights()
        state.pop("model")
        return state

    def __setstate__(self, state):
        if "keras_json" not in state:
            raise ValueError("CNN antigua de PyTorch: reentrena con TensorFlow/Keras.")
        from tensorflow import keras
        state = state.copy()
        architecture, weights = state.pop("keras_json"), state.pop("keras_weights")
        self.__dict__.update(state)
        self.model = keras.models.model_from_json(architecture)
        self.model.set_weights(weights)


def crear(seed=42):
    return CNN(seed)
