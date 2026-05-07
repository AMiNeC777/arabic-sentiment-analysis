"""Evaluation entry module for sentiment models."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.metrics import accuracy_score, classification_report, f1_score


def evaluate_predictions(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """Return core classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def build_classification_report(y_true: Any, y_pred: Any) -> str:
    """Generate detailed per-class metrics text report."""
    return classification_report(y_true, y_pred)

