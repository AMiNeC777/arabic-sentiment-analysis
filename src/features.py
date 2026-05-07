"""Feature engineering entry module."""

from __future__ import annotations

from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

def build_bow_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df:int =2,
    max_df:float = 0.95,
) -> CountVectorizer:
    return CountVectorizer(
        max_features=max_features, 
        ngram_range=ngram_range, 
        min_df=min_df, 
        max_df=max_df,
        lowercase=False,
    )

def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int,int] = (1,2),
    min_df:int =2,
    max_df:float = 0.95,
    
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features, 
        ngram_range=ngram_range, 
        min_df=min_df, 
        max_df=max_df,
        lowercase = False,
        sublinear_tf = True,
        norm = 'l2',
        use_idf = True,
        smooth_idf = True,
    )

