from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


INPUT_FILE = "Email-des-RH-pour-ceux-qui-seront-a-la-recherche-des-stages-de-fin-d_etude-.csv"
OUTPUT_FILE = "Email-des-RH-cleaned-structured.csv"
OUTPUT_UNPARSED_FILE = "Email-des-RH-unparsed.csv"

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

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_ONLY_RE = re.compile(r"^[\d\s./()+-]+(?:STD)?$", re.IGNORECASE)
MULTI_SPACES_RE = re.compile(r"\s{2,}")


def normalize_row(value: str) -> str:
    return value.strip().strip('"').strip()


def is_noise_row(text: str) -> bool:
    # Drop blank rows, header row, and dangling phone/email-only rows.
    if not text:
        return True
    if "SOCIETE" in text and "E-mail" in text:
        return True
    if PHONE_ONLY_RE.fullmatch(text):
        return True
    if EMAIL_RE.fullmatch(text):
        return True
    return False


def parse_columns(text: str) -> dict[str, str]:
    """
    Parse an irregular fixed-width row into target columns.
    Split on 2+ spaces and align from the right where possible.
    """
    parts = [part.strip() for part in MULTI_SPACES_RE.split(text) if part.strip()]
    row = {col: "" for col in TARGET_COLUMNS}

    if not parts:
        return row

    # Last fields are usually most reliable.
    if parts and "@" in parts[-1]:
        row["E-mail"] = parts.pop()
    elif len(parts) >= 2 and "@" in parts[-2]:
        row["ADRESSE"] = parts.pop()
        row["E-mail"] = parts.pop()

    if parts:
        # Address tends to be at the end if email was extracted.
        if row["E-mail"]:
            row["ADRESSE"] = parts.pop() if parts else row["ADRESSE"]

    # Pull known phone-like fields from the right.
    if parts:
        row["GSM"] = parts.pop() if re.search(r"\d", parts[-1]) else row["GSM"]
    if parts:
        row["TELFAX"] = parts.pop() if re.search(r"\d", parts[-1]) else row["TELFAX"]

    # Remaining left-to-right fields.
    left_fields = ["SOCIETE", "GROUPE", "VILLE", "ACTIVITE", "PERSONNE", "FONCTION / SERVICE"]
    for field in left_fields:
        if parts:
            row[field] = parts.pop(0)

    # If anything remains, append to address.
    if parts:
        remainder = " ".join(parts).strip()
        row["ADRESSE"] = f"{row['ADRESSE']} {remainder}".strip()

    # Final normalization for each cell.
    for key in row:
        row[key] = MULTI_SPACES_RE.sub(" ", row[key]).strip()

    return row


def main() -> None:
    test_dir = Path(__file__).resolve().parent
    input_path = test_dir / INPUT_FILE
    output_path = test_dir / OUTPUT_FILE
    output_unparsed_path = test_dir / OUTPUT_UNPARSED_FILE

    raw_df = pd.read_csv(
        input_path,
        header=None,
        names=["raw"],
        encoding="utf-8",
        keep_default_na=False,
    )

    raw_df["raw"] = raw_df["raw"].astype(str).map(normalize_row)
    cleaned_df = raw_df[~raw_df["raw"].map(is_noise_row)].copy()
    parsed_df = cleaned_df["raw"].map(parse_columns).apply(pd.Series)
    parsed_df = parsed_df[TARGET_COLUMNS]
    parsed_df = parsed_df[parsed_df["SOCIETE"] != ""].copy()

    # Save rows with very low parse quality for manual inspection.
    unparsed_mask = (
        (parsed_df["E-mail"] == "")
        & (parsed_df["ADRESSE"] == "")
        & (parsed_df["TELFAX"] == "")
        & (parsed_df["GSM"] == "")
    )
    unparsed_df = parsed_df[unparsed_mask].copy()
    parsed_df = parsed_df[~unparsed_mask].copy()

    parsed_df.to_csv(output_path, index=False, encoding="utf-8")
    unparsed_df.to_csv(output_unparsed_path, index=False, encoding="utf-8")

    print(f"Input rows: {len(raw_df)}")
    print(f"Rows after noise removal: {len(cleaned_df)}")
    print(f"Parsed rows saved: {len(parsed_df)}")
    print(f"Unparsed rows saved: {len(unparsed_df)}")
    print(f"Saved cleaned file: {output_path}")
    print(f"Saved unparsed file: {output_unparsed_path}")


if __name__ == "__main__":
    main()