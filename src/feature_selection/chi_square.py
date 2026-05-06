from sklearn.feature_selection import SelectKBest, chi2


def select_chi_square(X, y, k: int = 5000):
    """Select top-k features using Chi-Square."""
    selector = SelectKBest(score_func=chi2, k=k)
    return selector.fit_transform(X, y), selector
