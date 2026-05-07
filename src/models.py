"""Model training and inference entry module."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression


def build_model(random_state: int = 42) -> LogisticRegression:
    """Create a baseline classifier for sentiment prediction."""
    return LogisticRegression(max_iter=1000, random_state=random_state)


def train_model(features: Any, labels: Any) -> LogisticRegression:
    """Train a sentiment model on vectorized features."""
    model = build_model()
    model.fit(features, labels)
    return model


def predict_labels(model: LogisticRegression, features: Any):
    """Predict class labels using a trained model."""
    return model.predict(features)

