"""Write before/after samples to report/preprocessing_examples.utf8 for LaTeX."""

from __future__ import annotations

import csv
import importlib.util
from io import StringIO
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "report" / "preprocessing_examples.utf8.txt"


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "preprocessing_mod", _ROOT / "src" / "preprocessing.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(m)

    lines: list[str] = []

    tweets = (_ROOT / "data" / "Tweets.txt").read_text(encoding="utf-8").splitlines()
    for ln in tweets:
        if "\t" not in ln or "#" not in ln:
            continue
        raw = ln.rsplit("\t", 1)[0].strip()
        if len(raw) > 40:
            clean = m.preprocess_tweet(raw)
            lines.append(f"TWEET_RAW_LEN {len(raw)}")
            lines.append(f"TWEET_CLEAN_LEN {len(clean)}")
            lines.append("---RAW_TWEET---")
            lines.append(raw[:400])
            lines.append("---CLEAN_TWEET---")
            lines.append(clean[:400])
            break

    text = (_ROOT / "data" / "balanced-reviews.txt").read_bytes().decode("utf-16-le")
    for row in csv.reader(StringIO(text), delimiter="\t"):
        if len(row) < 7:
            continue
        rev = row[6]
        if '"' in rev or "\u201c" in rev:
            clean = m.preprocess_balanced_review(rev)
            lines.append(f"REV_RAW_LEN {len(rev)}")
            lines.append(f"REV_CLEAN_LEN {len(clean)}")
            lines.append("---RAW_REVIEW---")
            lines.append(rev[:400])
            lines.append("---CLEAN_REVIEW---")
            lines.append(clean[:400])
            break

    _OUT.write_text("\n".join(lines), encoding="utf-8")
    print(_OUT)


if __name__ == "__main__":
    main()
