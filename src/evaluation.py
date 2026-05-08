"""Evaluation entry module for sentiment models."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import time 

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

def compute_accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))

def evaluate_predictions(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """Return core classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }

def compute_precision(y_true, y_pred) -> float:
    return float(precision_score(y_true, y_pred, average="macro", zero_division=0))

def build_classification_report(y_true: Any, y_pred: Any) -> str:
    """Generate detailed per-class metrics text report."""
    return classification_report(y_true, y_pred)

def compute_recall(y_true, y_pred) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))

def compute_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))    

def compute_roc_auc(y_true, y_proba, labels) -> float:
    """y_proba: shape (n_samples, n_classes) from clf.predict_proba(X_test)"""
    if len(labels) == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=labels)
    return float(roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr"))

def compute_confusion_matrix(y_true, y_pred, labels):
    """Returns a NumPy array of shape (n_classes, n_classes)."""
    return confusion_matrix(y_true, y_pred, labels=labels)

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def time_training(clf, X_train, y_train) -> float:
    start = time.perf_counter()
    clf.fit(X_train, y_train)
    return time.perf_counter() - start