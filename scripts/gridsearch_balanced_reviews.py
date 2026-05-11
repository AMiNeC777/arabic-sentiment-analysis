"""GridSearchCV on balanced hotel reviews (binary stars → sentiment). Run from repo root: python scripts/gridsearch_balanced_reviews.py"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import preprocess_balanced_review, star_rating_to_binary


def load_xy():
    p = ROOT / "data" / "balanced-reviews.txt"
    raw = pd.read_csv(p, sep="	", encoding="utf-16")
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw[["review", "rating"]].copy()
    df["review"] = df["review"].astype(str)
    df["lab"] = df["rating"].map(lambda r: star_rating_to_binary(r, neutral="drop"))
    df = df[df["lab"].notna() & df["lab"].isin(["POS", "NEG"])].copy()
    vc = df["lab"].value_counts()
    min_n = int(vc.min())
    pos = df[df["lab"] == "POS"].sample(min_n, random_state=SEED)
    neg = df[df["lab"] == "NEG"].sample(min_n, random_state=SEED)
    bal = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=SEED)
    x = bal["review"].map(preprocess_balanced_review)
    y = (bal["lab"] == "POS").astype(int)
    m = x.str.strip().ne("")
    return x[m], y[m]


def main() -> None:
    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            ("clf", LinearSVC(max_iter=10000, dual=False)),
        ]
    )

    grid = GridSearchCV(
        pipe,
        {"clf__C": [0.1, 0.5, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    pred = grid.predict(X_test)

    print("Best params:", grid.best_params_)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, digits=4))


if __name__ == "__main__":
    main()
