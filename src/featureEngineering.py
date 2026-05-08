from __future__ import annotations
from typing import Dict, List
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif


DEFAULT_TOP_K: List[int] = [20, 40, 100, 200, 400, 600, 1000, 1500]

def chi2_selector(K:int ) -> SelectKBest:
    return SelectKBest(chi2, k=K)

def mutual_info_selector(K:int, random_state:int = 42) -> SelectKBest:

    def _mi_score(X,y):
        return mutual_info_classif(X,y,random_state=random_state)

    return SelectKBest(_mi_score, k=K)




