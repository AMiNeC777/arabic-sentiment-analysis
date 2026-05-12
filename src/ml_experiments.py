"""
Reusable helpers for comparing vectorizers, feature selectors, and hyperparameters.

Intended for notebooks and ``scripts/compare_*.py`` / ``scripts/gridsearch_hyperparams.py``.

**Notebook setup** (run with cwd = ``notebooks/`` or adjust ``ROOT``)::

    import sys
    from pathlib import Path
    ROOT = Path("..").resolve()
    sys.path.insert(0, str(ROOT / "src"))
    from ml_experiments import (
        load_binary_xy,
        extract_features,
        apply_selection,
        cross_val_score_pipeline,
        holdout_evaluate,
        compare_methods_table,
        run_gridsearch_text_pipeline,
    )
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from Feature_Extraction import vectorize as fe_vectorize
from Feature_Selection import select_features

DatasetName = Literal["reviews", "tweets"]
ClassifierName = Literal["linear_svc", "logistic", "nb"]

SEED = 42


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_src_on_path() -> None:
    src = repo_root() / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_binary_reviews_xy(
    max_samples: int | None = None,
    *,
    random_state: int = SEED,
) -> tuple[pd.Series, pd.Series]:
    """Load hotel reviews as (text Series, int labels 0/1). UTF-16 file."""
    from preprocessing import preprocess_balanced_review, star_rating_to_binary

    p = repo_root() / "data" / "balanced-reviews.txt"
    raw = pd.read_csv(p, sep="\t", encoding="utf-16")
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw[["review", "rating"]].copy()
    df["review"] = df["review"].astype(str)
    df["lab"] = df["rating"].map(lambda r: star_rating_to_binary(r, neutral="drop"))
    df = df[df["lab"].notna() & df["lab"].isin(["POS", "NEG"])].copy()
    vc = df["lab"].value_counts()
    min_n = int(vc.min())
    pos = df[df["lab"] == "POS"].sample(min_n, random_state=random_state)
    neg = df[df["lab"] == "NEG"].sample(min_n, random_state=random_state)
    bal = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=random_state)
    x = bal["review"].map(preprocess_balanced_review)
    y = (bal["lab"] == "POS").astype(int)
    m = x.str.strip().ne("")
    x, y = x[m], y[m]
    if max_samples is not None and len(x) > max_samples:
        idx = np.random.default_rng(random_state).choice(len(x), size=max_samples, replace=False)
        x = x.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
    return x, y


def load_binary_tweets_xy(
    max_samples: int | None = None,
    *,
    random_state: int = SEED,
) -> tuple[pd.Series, pd.Series]:
    """Load ASTD Tweets.txt as (text Series, int labels 0=NEG, 1=POS)."""
    data_path = repo_root() / "ASTD" / "data" / "Tweets.txt"
    if not data_path.exists():
        data_path = repo_root() / "data" / "Tweets.txt"
    rows: list[tuple[str, str]] = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        text, label = line.rsplit("\t", 1)
        lab = label.strip().upper()
        if lab in {"POS", "NEG"}:
            rows.append((text.strip(), lab))
    df = pd.DataFrame(rows, columns=["text", "label"])
    vc = df["label"].value_counts()
    min_n = int(vc.min())
    pos = df[df["label"] == "POS"].sample(min_n, random_state=random_state)
    neg = df[df["label"] == "NEG"].sample(min_n, random_state=random_state)
    bal = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=random_state)
    x = bal["text"].astype(str)
    y = (bal["label"] == "POS").astype(int)
    if max_samples is not None and len(x) > max_samples:
        idx = np.random.default_rng(random_state).choice(len(x), size=max_samples, replace=False)
        x = x.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
    return x, y


def load_binary_xy(
    dataset: DatasetName,
    max_samples: int | None = None,
    *,
    random_state: int = SEED,
) -> tuple[pd.Series, pd.Series]:
    if dataset == "reviews":
        return load_binary_reviews_xy(max_samples=max_samples, random_state=random_state)
    if dataset == "tweets":
        return load_binary_tweets_xy(max_samples=max_samples, random_state=random_state)
    raise ValueError("dataset must be 'reviews' or 'tweets'")


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
    }


def default_vectorizer_kwargs() -> dict[str, Any]:
    return {
        "max_features": 20000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
    }


def extract_features(
    X_train,
    X_test,
    method: str,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Wrapper around ``Feature_Extraction.vectorize`` with shared defaults."""
    opts = default_vectorizer_kwargs()
    opts.update(kwargs)
    return fe_vectorize(X_train, X_test, method=method, **opts)


def apply_selection(
    X_train,
    y_train,
    X_test,
    method: str,
    k: int,
):
    """``Feature_Selection.select_features`` including ``none`` / ``identity``."""
    return select_features(X_train, y_train, X_test, method=method, k=k)


def build_classifier(name: ClassifierName | str, **kwargs: Any):
    """
    Instantiate a classifier (sklearn) without importing ``models`` — avoids
    package-relative imports when ``src`` is used as a flat path.
    """
    k = str(name).lower().strip().replace("-", "_")
    if k in {"logistic", "lr", "logistic_regression"}:
        opts: dict[str, Any] = {"random_state": SEED, "max_iter": 10000}
        opts.update(kwargs)
        return LogisticRegression(**opts)
    if k in {"nb", "multinomial_nb", "naive_bayes"}:
        nb_kwargs = dict(kwargs)
        nb_kwargs.pop("random_state", None)
        return MultinomialNB(**nb_kwargs)
    if k in {"svc", "linear_svc", "svm"}:
        opts = {"random_state": SEED, "max_iter": 10000, "dual": False}
        opts.update(kwargs)
        return LinearSVC(**opts)
    raise ValueError(f"Unknown classifier {name!r}; use linear_svc, logistic, or nb")


def cross_val_score_pipeline(
    X_text: pd.Series,
    y: pd.Series,
    *,
    extraction_method: str,
    selection_method: str,
    k_select: int,
    classifier_name: ClassifierName | str = "linear_svc",
    n_splits: int = 5,
    random_state: int = SEED,
    vec_kwargs: dict[str, Any] | None = None,
    clf_kwargs: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """
    Stratified K-fold on *X_text* (training pool): each fold refits vectorizer + selector + clf.

    Returns (mean f1, std f1).
    """
    vec_kwargs = vec_kwargs or {}
    clf_kwargs = dict(clf_kwargs or {})
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: list[float] = []
    X_arr = X_text.reset_index(drop=True).astype(str)
    y_arr = np.asarray(y)

    for train_idx, val_idx in skf.split(np.zeros(len(y_arr)), y_arr):
        Xtr, Xva = X_arr.iloc[train_idx], X_arr.iloc[val_idx]
        ytr, yva = y_arr[train_idx], y_arr[val_idx]

        Xvtr, Xvva = extract_features(Xtr, Xva, extraction_method, **vec_kwargs)
        Xvtr, Xvva = apply_selection(Xvtr, ytr, Xvva, selection_method, k_select)

        clf = build_classifier(classifier_name, **clf_kwargs)
        clf.fit(Xvtr, ytr)
        pred = clf.predict(Xvva)
        scores.append(f1_score(yva, pred, average="binary", pos_label=1, zero_division=0))

    return float(np.mean(scores)), float(np.std(scores))


def holdout_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    extraction_method: str,
    selection_method: str,
    k_select: int,
    classifier_name: ClassifierName | str = "linear_svc",
    vec_kwargs: dict[str, Any] | None = None,
    clf_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single train/test evaluation (already-split text)."""
    vec_kwargs = vec_kwargs or {}
    clf_kwargs = dict(clf_kwargs or {})
    Xvtr, Xvte = extract_features(X_train, X_test, extraction_method, **vec_kwargs)
    Xvtr, Xvte = apply_selection(Xvtr, y_train, Xvte, selection_method, k_select)
    clf = build_classifier(classifier_name, **clf_kwargs)
    clf.fit(Xvtr, y_train)
    pred = clf.predict(Xvte)
    metrics = binary_classification_metrics(np.asarray(y_test), pred)
    return {"metrics": metrics, "model": clf}


def compare_methods_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Sort by ``mean_f1`` descending when present, else ``f1``."""
    df = pd.DataFrame(rows)
    key = "mean_f1" if "mean_f1" in df.columns else "f1"
    if key in df.columns:
        df = df.sort_values(key, ascending=False).reset_index(drop=True)
    return df


def common_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", choices=["reviews", "tweets"], default="reviews")
    p.add_argument("--max-samples", type=int, default=None, help="Cap rows for faster runs")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv", type=int, default=5, help="CV folds for model selection")
    p.add_argument("--k-select", type=int, default=8000, help="Top-k features for selectors")
    p.add_argument("--classifier", default="linear_svc", help="linear_svc | logistic | nb")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--output-csv", type=str, default=None, help="Write results table to CSV")
    return p


def init_rng(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def run_gridsearch_text_pipeline(
    X_train,
    y_train,
    param_grid: dict[str, list[Any]],
    *,
    clf_kind: ClassifierName | str = "linear_svc",
    n_jobs: int = -1,
    cv: int = 5,
    scoring: str = "f1",
    verbose: int = 1,
    random_state: int = SEED,
) -> GridSearchCV:
    """
    ``GridSearchCV`` on ``Pipeline(vec=TfidfVectorizer, clf=...)`` fitted on *X_train*, *y_train*.

    *param_grid* keys use prefixes ``vec__`` and ``clf__`` (e.g. ``clf__C``, ``vec__max_features``).
    """
    k = str(clf_kind).lower().strip()
    if k in {"linear_svc", "svc", "svm"}:
        clf = LinearSVC(max_iter=10000, dual=False, random_state=random_state)
    elif k in {"logistic", "lr", "logistic_regression"}:
        clf = LogisticRegression(max_iter=10000, random_state=random_state)
    elif k in {"nb", "multinomial_nb", "naive_bayes"}:
        clf = MultinomialNB()
    else:
        raise ValueError(f"Unknown clf_kind {clf_kind!r}")

    pipe = Pipeline(
        [
            (
                "vec",
                TfidfVectorizer(
                    lowercase=False,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            ("clf", clf),
        ]
    )

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    grid.fit(X_train, y_train)
    return grid
