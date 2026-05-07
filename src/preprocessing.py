"""Preprocessing entry module for text cleaning and normalization."""

from __future__ import annotations

import re
from typing import Iterable, List


ARABIC_DIACRITICS_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
NON_ARABIC_LETTERS_PATTERN = re.compile(r"[^\u0600-\u06FF\s]")
EXTRA_SPACES_PATTERN = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Normalize common Arabic character variants."""
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ة": "ه",
        "ى": "ي",
    }
    normalized = text
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def clean_text(text: str) -> str:
    """Basic Arabic text cleaning: strip diacritics and non-Arabic symbols."""
    text = ARABIC_DIACRITICS_PATTERN.sub("", text)
    text = NON_ARABIC_LETTERS_PATTERN.sub(" ", text)
    text = EXTRA_SPACES_PATTERN.sub(" ", text).strip()
    return normalize_arabic(text)


def preprocess_texts(texts: Iterable[str]) -> List[str]:
    """Preprocess a batch of raw texts."""
    return [clean_text(text) for text in texts]

