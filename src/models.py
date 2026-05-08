"""Model training and inference entry module."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def LogisticRegression(X_train, y_train, random_state: int = 42, max_iter: int = 1000,) -> LogisticRegression:
    """Create a baseline classifier for sentiment prediction."""
    model = LogisticRegression(max_iter=max_iter, random_state=random_state,)
    model.fit(X_train, y_train)
    return model

def MultinomialNB(X_train, y_train, random_state: int = 42) -> MultinomialNB:
    """Create a baseline classifier for sentiment prediction."""
    model = MultinomialNB(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def LinearSVC(X_train, y_train, random_state: int = 42) -> LinearSVC:
    """Create a baseline classifier for sentiment prediction."""
    model = LinearSVC(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    """Predict class labels using a trained model."""
    return model.predict(X_test)

