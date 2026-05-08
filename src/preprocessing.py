"""Preprocessing entry module for text cleaning and normalization."""

from __future__ import annotations

import re
from typing import Iterable, List
try:
    from light_stemmer import LightStemmer  # type: ignore[import-not-found]
except ModuleNotFoundError:
    LightStemmer = None


ARABIC_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ك", "ل")
ARABIC_SUFFIXES = ("يات", "ات", "ون", "ين", "ان", "هم", "ها", "نا", "كم", "كن", "ه", "ة", "ي", "ك", "ا")

stemmer = LightStemmer("arabic") if LightStemmer is not None else None


def stem_arabic_token(token: str) -> str:
    """Fallback light stemmer for Arabic tokens if package is unavailable."""
    if len(token) <= 3:
        return token

    stemmed = token
    for prefix in ARABIC_PREFIXES:
        if stemmed.startswith(prefix) and len(stemmed) - len(prefix) >= 3:
            stemmed = stemmed[len(prefix):]
            break

    for suffix in ARABIC_SUFFIXES:
        if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= 3:
            stemmed = stemmed[:-len(suffix)]
            break

    return stemmed


def stem_arabic(text: str) -> str:
    """Apply stemming token by token using package or fallback stemmer."""
    if stemmer is not None:
        return " ".join(stemmer.stem(word) for word in text.split())
    return " ".join(stem_arabic_token(word) for word in text.split())

ARABIC_DIACRITICS_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
NON_ARABIC_LETTERS_PATTERN = re.compile(r"[^\u0600-\u06FF\s]")
EXTRA_SPACES_PATTERN = re.compile(r"\s+")
EMO_POS = {"😀","😁","😂","🤣","😊","😍","❤","❤️","👍","👏","🥰","😎"}
EMO_NEG = {"😡","😠","😢","😭","👎","💔","😞","😣","😖","🤮"}
ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
TATWEEL = re.compile(r"\u0640")
URLS = re.compile(r"http\S+|www\.\S+")
MENTIONS_HASHTAGS = re.compile(r"[@#]\w+")
NON_ARABIC_KEEP_TOKENS = re.compile(r"[^\u0600-\u06FF\s_]")
REPEAT_CHARS = re.compile(r"(.)\1{2,}")  # 3+ repeated chars
SPACES = re.compile(r"\s+")

def remove_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    return re.sub(r"[^\w\s]", "", text)


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

def map_emojis(text: str) -> str:
    chars = []
    for ch in text:
        if ch in EMO_POS:
            chars.append(" EMO_POS ")
        elif ch in EMO_NEG:
            chars.append(" EMO_NEG ")
        else:
            chars.append(ch)
    return "".join(chars)

def clean_arabic_text(text: str, keep_emoji_sentiment: bool = True) -> str:
    text = URLS.sub(" ", text)
    text = MENTIONS_HASHTAGS.sub(" ", text)
    if keep_emoji_sentiment:
        text = map_emojis(text)
    else:
        text = "".join(ch for ch in text if ch not in EMO_POS and ch not in EMO_NEG)
    text = normalize_arabic(text)
    text = ARABIC_DIACRITICS.sub("", text)
    text = TATWEEL.sub("", text)
    text = REPEAT_CHARS.sub(r"\1", text)
    text = NON_ARABIC_KEEP_TOKENS.sub(" ", text)
    text = SPACES.sub(" ", text).strip()
    return text

def clean_text(
    text: str,
    keep_emoji_sentiment: bool = True,
    use_stemming: bool = False,
) -> str:
    """Basic Arabic text cleaning: strip diacritics and non-Arabic symbols."""
    text = ARABIC_DIACRITICS_PATTERN.sub("", text)
    text = NON_ARABIC_LETTERS_PATTERN.sub(" ", text)
    text = EXTRA_SPACES_PATTERN.sub(" ", text).strip()
    if keep_emoji_sentiment:
        text = map_emojis(text)
    text = remove_punctuation(text)
    text = normalize_arabic(text)
    text = clean_arabic_text(text, keep_emoji_sentiment=keep_emoji_sentiment)
    if use_stemming:
        text = stem_arabic(text)
    return text


def preprocess_texts(
    texts: Iterable[str],
    keep_emoji_sentiment: bool = True,
    use_stemming: bool = False,
) -> List[str]:
    """Preprocess a batch of texts with optional stemming."""
    return [
        clean_text(
            text,
            keep_emoji_sentiment=keep_emoji_sentiment,
            use_stemming=use_stemming,
        )
        for text in texts
    ]
    