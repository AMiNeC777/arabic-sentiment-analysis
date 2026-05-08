"""Feature engineering entry module."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

def bow_vectorizer(
    X_train,
    X_test,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df:int =2,
    max_df:float = 0.95,
    ):
    Count_vectorizer = CountVectorizer(
        max_features=max_features, 
        ngram_range=ngram_range, 
        min_df=min_df,
        max_df=max_df,
        lowercase=False,)

    X_train_bow = Count_vectorizer.fit_transform(X_train)
    X_test_bow = Count_vectorizer.transform(X_test)

    return X_train_bow, X_test_bow

def tfidf(
    X_train,
    X_test,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        sublinear_tf=True,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf
