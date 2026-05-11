"""Compute raw vs cleaned statistics for Tweets.txt and balanced-reviews.txt."""

from __future__ import annotations

import csv
import importlib.util
import statistics
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _load_preprocessing():
    path = _REPO / "src" / "preprocessing.py"
    spec = importlib.util.spec_from_file_location("preprocessing_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _tweet_rows():
    p = _REPO / "data" / "Tweets.txt"
    for raw in p.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or "\t" not in raw:
            continue
        text, _ = raw.rsplit("\t", 1)
        yield text.strip()


def _review_rows():
    p = _REPO / "data" / "balanced-reviews.txt"
    text = p.read_bytes().decode("utf-16-le")
    rdr = csv.reader(text.splitlines(), delimiter="\t")
    next(rdr, None)
    for row in rdr:
        if len(row) >= 7:
            yield row[6]


def main() -> None:
    prep = _load_preprocessing()

    # --- tweets ---
    tweets = list(_tweet_rows())
    clean_tweets = [prep.preprocess_tweet(t) for t in tweets]
    raw_lens_t = [len(t) for t in tweets]
    clean_lens_t = [len(c) for c in clean_tweets]
    tok_raw_t = [len(t.split()) for t in tweets]
    tok_clean_t = [len(c.split()) for c in clean_tweets]
    empty_t = sum(1 for c in clean_tweets if not c.strip())

    # --- reviews ---
    revs = list(_review_rows())
    clean_revs = [prep.preprocess_balanced_review(t) for t in revs]
    raw_lens_r = [len(t) for t in revs]
    clean_lens_r = [len(c) for c in clean_revs]
    tok_raw_r = [len(t.split()) for t in revs]
    tok_clean_r = [len(c.split()) for c in clean_revs]
    empty_r = sum(1 for c in clean_revs if not c.strip())

    def summarize(xs):
        return {
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        }

    def pct_shortening(raw, clean):
        pairs = [(a, b) for a, b in zip(raw, clean) if a > 0]
        return 100.0 * statistics.mean((a - b) / a for a, b in pairs)

    out = {
        "tweets_n": len(tweets),
        "reviews_n": len(revs),
        "tweet_chars_raw": summarize(raw_lens_t),
        "tweet_chars_clean": summarize(clean_lens_t),
        "tweet_tok_raw": summarize(tok_raw_t),
        "tweet_tok_clean": summarize(tok_clean_t),
        "tweet_pct_shorter": pct_shortening(raw_lens_t, clean_lens_t),
        "tweet_empty": empty_t,
        "review_chars_raw": summarize(raw_lens_r),
        "review_chars_clean": summarize(clean_lens_r),
        "review_tok_raw": summarize(tok_raw_r),
        "review_tok_clean": summarize(tok_clean_r),
        "review_pct_shorter": pct_shortening(raw_lens_r, clean_lens_r),
        "review_empty": empty_r,
    }

    for k, v in out.items():
        print(f"{k}: {v}")

    # Save JSON for TeX hand-off
    import json

    serializable = {}
    for k, v in out.items():
        if isinstance(v, dict):
            serializable[k] = {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
        else:
            serializable[k] = v
    (_REPO / "report" / "preprocessing_stats.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {_REPO / 'report' / 'preprocessing_stats.json'}")


if __name__ == "__main__":
    main()
