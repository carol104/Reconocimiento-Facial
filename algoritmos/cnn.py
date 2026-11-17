"""CNN real sobre recortes RGB alineados; PyTorch se carga solo al usar CNN."""
import numpy as np


def red(n_clases):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Dropout(0.3),
        nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, n_clases))


class CNN:
    def __init__(self, seed=42, epochs=30):
        self.seed, self.epochs = seed, epochs

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.model = red(len(self.classes_))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X),
                                           torch.tensor(encoded, dtype=torch.long)),
            batch_size=16, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_history_ = []
        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            for images, labels in loader:
                # Aumento solamente en entrenamiento, nunca en validación/test.
                if torch.rand(()) > 0.5:
                    images = images.flip(-1)
                optimizer.zero_grad()
                loss = loss_fn(self.model(images), labels)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(labels)
            self.loss_history_.append(total / len(X))
            print(f"CNN época {epoch + 1}/{self.epochs}: loss={self.loss_history_[-1]:.6f}")
        self.model.eval()
        return self

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def predict_proba(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            batches = [self.model(torch.from_numpy(X[i:i + 32])).softmax(1).numpy()
                       for i in range(0, len(X), 32)]
        return np.concatenate(batches)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["weights"] = self.model.state_dict()
        del state["model"]
        return state

    def __setstate__(self, state):
        weights = state.pop("weights")
        self.__dict__.update(state)
        self.model = red(len(self.classes_))
        self.model.load_state_dict(weights)
        self.model.eval()


def crear(seed=42):
    return CNN(seed)
