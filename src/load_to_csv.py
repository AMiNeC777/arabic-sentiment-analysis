"""Parse Tweets.txt and export raw/clean CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

try:
    from src.preprocessing import clean_text
except ModuleNotFoundError:
    from preprocessing import clean_text


VALID_LABELS = {"POS", "NEG", "OBJ", "NEUTRAL"}


def parse_tweets_file(path: Path) -> Tuple[List[str], List[str], List[Tuple[int, str]]]:
    """Parse lines in format: tweet<TAB>label."""
    texts: List[str] = []
    labels: List[str] = []
    bad_lines: List[Tuple[int, str]] = []

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if "\t" not in line:
            bad_lines.append((line_no, "No tab separator"))
            continue

        text, label = line.rsplit("\t", 1)
        text = text.strip()
        label = label.strip().upper()

        if not text:
            bad_lines.append((line_no, "Empty text"))
            continue
        if label not in VALID_LABELS:
            bad_lines.append((line_no, f"Invalid label: {label}"))
            continue

        texts.append(text)
        labels.append(label)

    return texts, labels, bad_lines


def write_raw_csv(texts: List[str], labels: List[str], out_path: Path) -> None:
    """Write raw parsed tweets and labels."""
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["text", "label"])
        writer.writerows(zip(texts, labels))


def write_clean_csv(texts: List[str], labels: List[str], out_path: Path) -> None:
    """Write raw + cleaned tweets and labels."""
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["clean_text", "label"])
        for text, label in zip(texts, labels):
            writer.writerow([clean_text(text, use_stemming=True), label])


def main() -> None:
    """Run end-to-end conversion from Tweets.txt to CSV outputs."""
    repo_root = Path(__file__).resolve().parent.parent
    input_path = repo_root / "data" / "Tweets.txt"
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = output_dir / "tweets_raw.csv"
    clean_csv_path = output_dir / "tweets_clean.csv"

    texts, labels, bad_lines = parse_tweets_file(input_path)
    write_raw_csv(texts, labels, raw_csv_path)
    write_clean_csv(texts, labels, clean_csv_path)

    print(f"Parsed valid rows: {len(texts)}")
    print(f"Malformed rows skipped: {len(bad_lines)}")
    print(f"Saved raw CSV: {raw_csv_path}")
    print(f"Saved clean CSV: {clean_csv_path}")


if __name__ == "__main__":
    main()