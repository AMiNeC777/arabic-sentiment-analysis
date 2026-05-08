import numpy as np
from scipy.sparse import issparse
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif


def chi2(X_train, y_train, X_test, k: int):
    """Select top-k features using Chi-square."""
    selector = SelectKBest(score_func=chi2, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel, selector


def mi(X_train, y_train, X_test, k: int):
    """Select top-k features using Mutual Information."""
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel, selector


def fisher_score(X, y):
    """
    Compute Fisher score for each feature.

    Higher score means better class separation.
    """
    if issparse(X):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)

    overall_mean = X.mean(axis=0)
    numerator = np.zeros(X.shape[1], dtype=np.float64)
    denominator = np.zeros(X.shape[1], dtype=np.float64)

    for cls in classes:
        Xc = X[y == cls]
        if Xc.shape[0] == 0:
            continue
        class_mean = Xc.mean(axis=0)
        class_var = Xc.var(axis=0)
        n_cls = Xc.shape[0]

        numerator += n_cls * (class_mean - overall_mean) ** 2
        denominator += n_cls * class_var

    scores = numerator / (denominator + 1e-12)
    return scores


def fisher(X_train, y_train, X_test, k: int):
    """Select top-k features using Fisher score ranking."""
    scores = fisher_score(X_train, y_train)
    top_idx = np.argsort(scores)[::-1][:k]
    X_train_sel = X_train[:, top_idx]
    X_test_sel = X_test[:, top_idx]
    return X_train_sel, X_test_sel, top_idx


def select_features(X_train, y_train, X_test, method: str, k: int):
    """
    Generic selector wrapper.

    method: "chi2" | "mi" | "fisher"
    """
    method = method.lower().strip()

    if method == "chi2":
        return chi2(X_train, y_train, X_test, k)
    if method in {"mi", "mutual_info", "mutual-information"}:
        return mi(X_train, y_train, X_test, k)
    if method == "fisher":
        return fisher(X_train, y_train, X_test, k)

    raise ValueError("method must be one of: 'chi2', 'mi', 'fisher'")
