"""Feature engineering entry module."""

from __future__ import annotations

from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Create a configurable TF-IDF vectorizer."""
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))


def extract_features(
    texts: Iterable[str],
    vectorizer: TfidfVectorizer | None = None,
):
    """Fit-transform text inputs into numeric features."""
    model = vectorizer or build_tfidf_vectorizer()
    features = model.fit_transform(texts)
    return features, model


def transform_features(texts: Iterable[str], vectorizer: TfidfVectorizer):
    """Transform texts with an already-fitted vectorizer."""
    return vectorizer.transform(texts)

