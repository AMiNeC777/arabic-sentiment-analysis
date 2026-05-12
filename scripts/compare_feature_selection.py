#!/usr/bin/env python3
"""
Compare feature selection methods (none, chi2, mi, fisher) with fixed extraction + classifier.

Uses stratified CV on the training split only, then refits the winner on full train and
reports hold-out metrics. Run from repo root:

  python scripts/compare_feature_selection.py --dataset reviews --max-samples 12000
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ml_experiments import (
    common_arg_parser,
    compare_methods_table,
    cross_val_score_pipeline,
    holdout_evaluate,
    init_rng,
    load_binary_xy,
)


def main() -> None:
    p = common_arg_parser(
        "Compare feature selection: none | chi2 | mi | fisher (CV mean F1, then hold-out)."
    )
    p.add_argument(
        "--extraction",
        default="tfidf",
        choices=["tfidf", "bow"],
        help="Fixed vectorization while sweeping selectors",
    )
    p.add_argument(
        "--methods",
        default="none,chi2,mi,fisher",
        help="Comma-separated subset of: none, chi2, mi, fisher",
    )
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

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    rows: list[dict] = []
    clf_kw = {"max_iter": 10000, "dual": False} if args.classifier in {"linear_svc", "svc"} else {}

    for method in methods:
        try:
            mean_f1, std_f1 = cross_val_score_pipeline(
                X_train,
                y_train,
                extraction_method=args.extraction,
                selection_method=method,
                k_select=args.k_select,
                classifier_name=args.classifier,
                n_splits=args.cv,
                random_state=args.seed,
                clf_kwargs=clf_kw,
            )
            rows.append(
                {
                    "selection": method,
                    "extraction": args.extraction,
                    "mean_f1": mean_f1,
                    "std_f1": std_f1,
                    "error": "",
                }
            )
        except Exception as e:  # noqa: BLE001 — surface MI / memory failures per method
            rows.append(
                {
                    "selection": method,
                    "extraction": args.extraction,
                    "mean_f1": float("nan"),
                    "std_f1": float("nan"),
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            traceback.print_exc()

    table = compare_methods_table(rows)
    print(table.to_string(index=False))
    valid = table[table["mean_f1"].notna()]
    if valid.empty:
        print("No successful runs.")
        return

    best = valid.iloc[0]
    print("\nBest by CV mean_f1:", best["selection"])
    out = holdout_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        extraction_method=args.extraction,
        selection_method=str(best["selection"]),
        k_select=args.k_select,
        classifier_name=args.classifier,
        clf_kwargs=clf_kw,
    )
    print("Hold-out:", out["metrics"])

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output_csv, index=False)
        print("Wrote", args.output_csv)


if __name__ == "__main__":
    main()
