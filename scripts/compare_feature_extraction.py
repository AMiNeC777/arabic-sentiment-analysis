#!/usr/bin/env python3
"""
Compare feature extraction methods (tfidf vs bow) with fixed selector + classifier.

Uses stratified CV on the training split, then hold-out evaluation for the winner.

  python scripts/compare_feature_extraction.py --dataset tweets --max-samples 5000
"""

from __future__ import annotations

import sys
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
        "Compare feature extraction: tfidf | bow (CV mean F1, then hold-out)."
    )
    p.add_argument(
        "--selection",
        default="none",
        help="Fixed selector: none | chi2 | mi | fisher",
    )
    p.add_argument(
        "--methods",
        default="tfidf,bow",
        help="Comma-separated subset of: tfidf, bow",
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

    for ext in methods:
        mean_f1, std_f1 = cross_val_score_pipeline(
            X_train,
            y_train,
            extraction_method=ext,
            selection_method=args.selection,
            k_select=args.k_select,
            classifier_name=args.classifier,
            n_splits=args.cv,
            random_state=args.seed,
            clf_kwargs=clf_kw,
        )
        rows.append(
            {
                "extraction": ext,
                "selection": args.selection,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
            }
        )

    table = compare_methods_table(rows)
    print(table.to_string(index=False))
    best = table.iloc[0]
    print("\nBest by CV mean_f1:", best["extraction"])
    out = holdout_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        extraction_method=str(best["extraction"]),
        selection_method=args.selection,
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
