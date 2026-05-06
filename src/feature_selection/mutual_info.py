from sklearn.feature_selection import SelectKBest, mutual_info_classif


def select_mutual_info(X, y, k: int = 5000):
    """Select top-k features using mutual information."""
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    return selector.fit_transform(X, y), selector
