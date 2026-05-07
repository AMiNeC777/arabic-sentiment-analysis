from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


INPUT_PDF = "Email des RH pour ceux qui seront a la recherche des stages de fin d'etude .pdf"
OUTPUT_CSV = "Email-des-RH-from-pdf-structured.csv"

TARGET_COLUMNS = [
    "SOCIETE",
    "GROUPE",
    "VILLE",
    "ACTIVITE",
    "PERSONNE",
    "FONCTION / SERVICE",
    "TELFAX",
    "GSM",
    "E-mail",
    "ADRESSE",
]

PAGE_MARKER_RE = re.compile(r"^--\s*\d+\s+of\s+\d+\s*--$")
MULTI_SPACE_RE = re.compile(r"\s+")
HEADER_TOKENS = [
    "SOCIETE",
    "GROUPE",
    "VILLE",
    "ACTIVITE",
    "PERSONNE",
    "FONCTION / SERVICE",
    "TEL",
    "FAX",
    "GSM",
    "E-mail",
    "ADRESSE",
]


def normalize(text: str) -> str:
    return MULTI_SPACE_RE.sub(" ", text).strip()


def extract_layout_lines(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    lines: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except KeyError:
            # Some pages may miss /Contents; fallback to standard extraction.
            text = page.extract_text() or ""
        for raw_line in text.splitlines():
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if PAGE_MARKER_RE.match(line.strip()):
                continue
            lines.append(line)
    return lines


def get_boundaries(lines: list[str]) -> list[tuple[str, int, int | None]]:
    header_line = ""
    for line in lines:
        if all(token in line for token in ("SOCIETE", "PERSONNE", "ADRESSE")):
            header_line = line
            break
    if not header_line:
        raise ValueError("Could not locate table header in PDF.")

    starts = []
    for token in HEADER_TOKENS:
        idx = header_line.find(token)
        if idx == -1:
            raise ValueError(f"Header token not found: {token}")
        starts.append((token, idx))

    starts = sorted(starts, key=lambda x: x[1])
    boundaries: list[tuple[str, int, int | None]] = []
    for idx, (token, start) in enumerate(starts):
        end = starts[idx + 1][1] if idx + 1 < len(starts) else None
        boundaries.append((token, start, end))
    return boundaries


def parse_line_by_boundaries(
    line: str,
    boundaries: list[tuple[str, int, int | None]],
) -> dict[str, str]:
    padded = line + " " * 20
    extracted: dict[str, str] = {}
    for token, start, end in boundaries:
        chunk = padded[start:end] if end is not None else padded[start:]
        extracted[token] = normalize(chunk)

    row = {
        "SOCIETE": extracted.get("SOCIETE", ""),
        "GROUPE": extracted.get("GROUPE", ""),
        "VILLE": extracted.get("VILLE", ""),
        "ACTIVITE": extracted.get("ACTIVITE", ""),
        "PERSONNE": extracted.get("PERSONNE", ""),
        "FONCTION / SERVICE": extracted.get("FONCTION / SERVICE", ""),
        "TELFAX": normalize(f"{extracted.get('TEL', '')} {extracted.get('FAX', '')}"),
        "GSM": extracted.get("GSM", ""),
        "E-mail": extracted.get("E-mail", ""),
        "ADRESSE": extracted.get("ADRESSE", ""),
    }
    return row


def is_continuation(row: dict[str, str]) -> bool:
    # Continuation lines often have no company/person but contain phone/email/address fragments.
    return not row["SOCIETE"] and not row["PERSONNE"]


def merge_continuations(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for row in rows:
        if is_continuation(row) and merged:
            prev = merged[-1]
            for col in TARGET_COLUMNS:
                if row[col]:
                    prev[col] = normalize(f"{prev[col]} {row[col]}")
            continue
        merged.append(row)
    return merged


def main() -> None:
    test_dir = Path(__file__).resolve().parent
    input_path = test_dir / INPUT_PDF
    output_path = test_dir / OUTPUT_CSV

    lines = extract_layout_lines(input_path)
    boundaries = get_boundaries(lines)

    rows: list[dict[str, str]] = []
    for line in lines:
        if all(token in line for token in ("SOCIETE", "PERSONNE", "ADRESSE")):
            continue
        parsed = parse_line_by_boundaries(line, boundaries)
        # Skip completely empty parsing artifacts.
        if not any(parsed.values()):
            continue
        rows.append(parsed)

    merged_rows = merge_continuations(rows)

    df = pd.DataFrame(merged_rows, columns=TARGET_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Extracted lines: {len(lines)}")
    print(f"Parsed lines: {len(rows)}")
    print(f"Merged records: {len(merged_rows)}")
    print(f"Saved structured CSV: {output_path}")


if __name__ == "__main__":
    main()
