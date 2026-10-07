"""Tiny trainable classifier on top of frozen CLIP embeddings."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

HEAD_PATH = Path(__file__).resolve().parent.parent / "models" / "head.joblib"


class FewShotHead:
    def __init__(self) -> None:
        self.clf: LogisticRegression | None = None
        self.classes: list[str] = []

    @property
    def ready(self) -> bool:
        return self.clf is not None

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> None:
        # Strong C: embeddings are L2-normalized, so features are small.
        self.clf = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced")
        self.clf.fit(embeddings, labels)
        self.classes = list(self.clf.classes_)

    def predict(self, embedding: np.ndarray) -> dict[str, float]:
        probs = self.clf.predict_proba(embedding.reshape(1, -1))[0]
        return {c: float(p) for c, p in zip(self.classes, probs)}

    def save(self, path: Path = HEAD_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "classes": self.classes}, path)

    def load(self, path: Path = HEAD_PATH) -> bool:
        if not path.exists():
            return False
        data = joblib.load(path)
        self.clf, self.classes = data["clf"], data["classes"]
        return True
