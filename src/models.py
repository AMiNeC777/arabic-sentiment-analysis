"""Model training and inference."""

from __future__ import annotations

from typing import Any, Literal

from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.naive_bayes import MultinomialNB as SKMultinomialNB
from sklearn.svm import LinearSVC as SKLinearSVC


ClassifierKind = Literal["logistic", "lr", "nb", "multinomial_nb", "svc", "linear_svc"]


def fit_logistic_regression(X_train, y_train, **kwargs: Any) -> SKLogisticRegression:
    opts: dict[str, Any] = {"random_state": 42, "max_iter": 10000}
    opts.update(kwargs)
    model = SKLogisticRegression(**opts)
    model.fit(X_train, y_train)
    return model


def fit_multinomial_nb(X_train, y_train, **kwargs: Any) -> SKMultinomialNB:
    """MultinomialNB has no ``random_state``; extra kwargs go to sklearn."""
    model = SKMultinomialNB(**kwargs)
    model.fit(X_train, y_train)
    return model


def fit_linear_svc(X_train, y_train, **kwargs: Any) -> SKLinearSVC:
    opts: dict[str, Any] = {"random_state": 42}
    opts.update(kwargs)
    model = SKLinearSVC(**opts)
    model.fit(X_train, y_train)
    return model


def train_classifier(kind: ClassifierKind | str, X_train, y_train, **kwargs: Any):
    """
    Train one of the built-in classifiers.

    kind: ``logistic`` / ``lr``, ``nb`` / ``multinomial_nb``, ``svc`` / ``linear_svc``
    """
    k = kind.lower().strip().replace("-", "_")
    if k in {"logistic", "lr", "logistic_regression"}:
        return fit_logistic_regression(X_train, y_train, **kwargs)
    if k in {"nb", "multinomial_nb", "naive_bayes"}:
        nb_kwargs = dict(kwargs)
        nb_kwargs.pop("random_state", None)
        return fit_multinomial_nb(X_train, y_train, **nb_kwargs)
    if k in {"svc", "linear_svc", "svm"}:
        return fit_linear_svc(X_train, y_train, **kwargs)
    raise ValueError(
        f"Unknown classifier kind {kind!r}; use logistic, nb, or svc."
    )


def predict_labels(model: Any, X: Any):
    """Predict class labels using a trained sklearn-style estimator."""
    return model.predict(X)


# Backwards-compatible alias
predict = predict_labels


# PyTorch LSTM / CNN on whitespace-tokenized text (see ``nn_sequence_models``).
from .nn_sequence_models import SequenceModelBundle, fit_sequence_classifier
