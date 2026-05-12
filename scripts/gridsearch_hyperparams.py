#!/usr/bin/env python3
"""
GridSearchCV over TfidfVectorizer + classifier hyperparameters (text pipeline).

Run from repo root:

  python scripts/gridsearch_hyperparams.py --dataset reviews --max-samples 8000 --preset narrow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import classification_report, accuracy_score, f1_score

from ml_experiments import (
    SEED,
    init_rng,
    load_binary_xy,
    run_gridsearch_text_pipeline,
)


def param_grid_narrow() -> dict:
    return {
        "vec__max_features": [8000, 20000],
        "vec__ngram_range": [(1, 1), (1, 2)],
        "vec__min_df": [1, 2],
        "clf__C": [0.1, 1.0, 10.0],
    }


def param_grid_wide() -> dict:
    return {
        "vec__max_features": [5000, 15000, 30000],
        "vec__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "vec__min_df": [1, 2, 3],
        "vec__max_df": [0.9, 0.95, 0.99],
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["reviews", "tweets"], default="reviews")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--classifier", default="linear_svc", help="linear_svc | logistic | nb")
    p.add_argument("--preset", choices=["narrow", "wide"], default="narrow")
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--output-csv", type=str, default=None, help="Save cv_results_ to CSV")
    args = p.parse_args()
    init_rng(args.seed)

    X, y = load_binary_xy(args.dataset, max_samples=args.max_samples, random_state=args.seed)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    clf = args.classifier.lower().strip()
    if clf in {"nb", "multinomial_nb", "naive_bayes"}:
        grid_space = {
            "vec__max_features": [8000, 20000],
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [1, 2],
            "clf__alpha": [0.01, 0.1, 1.0],
        }
    elif args.preset == "narrow":
        grid_space = param_grid_narrow()
    else:
        grid_space = param_grid_wide()
    grid = run_gridsearch_text_pipeline(
        X_train,
        y_train,
        grid_space,
        clf_kind=args.classifier,
        cv=args.cv,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        random_state=args.seed,
    )

    pred = grid.predict(X_test)
    print("Best params:", grid.best_params_)
    print("Best CV score (f1):", round(float(grid.best_score_), 6))
    print("Test accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Test f1 (binary):", round(f1_score(y_test, pred, average="binary", pos_label=1), 4))
    print(classification_report(y_test, pred, digits=4))

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        grid.cv_results_["params_str"] = [str(p) for p in grid.cv_results_["params"]]
        import pandas as pd

        pd.DataFrame(grid.cv_results_).to_csv(out, index=False)
        print("Wrote", out)


if __name__ == "__main__":
    main()
