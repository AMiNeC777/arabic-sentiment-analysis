"""Arabic text cleaning: small steps + two pipelines (tweets vs hotel reviews)."""

from __future__ import annotations

import re
from typing import Iterable, List, Literal

BinarySentiment = Literal["POS", "NEG"]

try:
    from light_stemmer import LightStemmer  # type: ignore[import-not-found]
except ModuleNotFoundError:
    LightStemmer = None


# --- optional stemming ---------------------------------------------------------

ARABIC_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ك", "ل")
ARABIC_SUFFIXES = ("يات", "ات", "ون", "ين", "ان", "هم", "ها", "نا", "كم", "كن", "ه", "ة", "ي", "ك", "ا")

stemmer = LightStemmer("arabic") if LightStemmer is not None else None


def stem_arabic_token(token: str) -> str:
    if len(token) <= 3:
        return token
    stemmed = token
    for prefix in ARABIC_PREFIXES:
        if stemmed.startswith(prefix) and len(stemmed) - len(prefix) >= 3:
            stemmed = stemmed[len(prefix) :]
            break
    for suffix in ARABIC_SUFFIXES:
        if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= 3:
            stemmed = stemmed[: -len(suffix)]
            break
    return stemmed


def stem_arabic(text: str) -> str:
    if stemmer is not None:
        return " ".join(stemmer.stem(word) for word in text.split())
    return " ".join(stem_arabic_token(word) for word in text.split())


# --- regex --------------------------------------------------------------------

ARABIC_DIACRITICS_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
NON_ARABIC_LETTERS_PATTERN = re.compile(r"[^\u0600-\u06FF\s]")
EXTRA_SPACES_PATTERN = re.compile(r"\s+")

EMO_POS = {"😀", "😁", "😂", "🤣", "😊", "😍", "❤", "❤️", "👍", "👏", "🥰", "😎"}
EMO_NEG = {"😡", "😠", "😢", "😭", "👎", "💔", "😞", "😣", "😖", "🤮"}

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
TATWEEL = re.compile(r"\u0640")
URLS = re.compile(r"http\S+|www\.\S+")
MENTIONS_HASHTAGS = re.compile(r"[@#]\w+")
HASHTAG_BODY = re.compile(r"#([^\s#]+)")
MENTIONS_ONLY = re.compile(r"@\S+")
NON_ARABIC_KEEP_TOKENS = re.compile(r"[^\u0600-\u06FF\s_]")
REPEAT_CHARS = re.compile(r"(.)\1{2,}")
SPACES = re.compile(r"\s+")

SMART_AND_ASCII_QUOTES = re.compile(
    r'[\u201c\u201d\u2018\u2019\u00ab\u00bb\u2039\u203a"\'´`]+'
)
STRAY_SYMBOL_RUNS = re.compile(r"[_\-=]{5,}")


# --- tiny steps ----------------------------------------------------------------


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


def normalize_arabic(text: str) -> str:
    for src, dst in (
        ("أ", "ا"),
        ("إ", "ا"),
        ("آ", "ا"),
        ("ة", "ه"),
        ("ى", "ي"),
    ):
        text = text.replace(src, dst)
    return text


def map_emojis(text: str) -> str:
    out = []
    for ch in text:
        if ch in EMO_POS:
            out.append(" EMO_POS ")
        elif ch in EMO_NEG:
            out.append(" EMO_NEG ")
        else:
            out.append(ch)
    return "".join(out)


def strip_decorative_quotes(text: str) -> str:
    """Hotel exports often wrap snippets in smart quotes — remove them."""
    return SMART_AND_ASCII_QUOTES.sub(" ", text)


def _strip_urls(text: str) -> str:
    return URLS.sub(" ", text)


def _twitter_handles_hashtags_and_noise(text: str) -> str:
    """Turn #موضوع into موضوع, drop @user, kill long ____ runs (tweet-specific)."""
    text = _strip_urls(text)
    text = MENTIONS_ONLY.sub(" ", text)
    text = HASHTAG_BODY.sub(r"\1 ", text)
    text = STRAY_SYMBOL_RUNS.sub(" ", text)
    return text


def _coarse_arabic_pass(text: str) -> str:
    """Drop diacritics not meant for sentiment; keep Arabic letters + spaces."""
    text = ARABIC_DIACRITICS_PATTERN.sub("", text)
    text = NON_ARABIC_LETTERS_PATTERN.sub(" ", text)
    text = EXTRA_SPACES_PATTERN.sub(" ", text).strip()
    return text


def _emoji_then_punct_then_normalize(text: str, *, keep_emoji_sentiment: bool) -> str:
    if keep_emoji_sentiment:
        text = map_emojis(text)
    else:
        text = "".join(ch for ch in text if ch not in EMO_POS and ch not in EMO_NEG)
    text = remove_punctuation(text)
    text = normalize_arabic(text)
    return text


def _deep_clean_arabic_surface(
    text: str,
    *,
    keep_emoji_sentiment: bool,
    preserve_hashtag_words: bool,
) -> str:
    """Last pass: URLs/social rules, diacritics, elongation, keep underscores for tokens."""
    text = _strip_urls(text)
    if preserve_hashtag_words:
        text = MENTIONS_ONLY.sub(" ", text)
        text = HASHTAG_BODY.sub(r"\1 ", text)
    else:
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


def _maybe_stem(text: str, use_stemming: bool) -> str:
    return stem_arabic(text) if use_stemming else text


# --- main pipelines ------------------------------------------------------------


def preprocess_tweet(
    text: str,
    *,
    keep_emoji_sentiment: bool = True,
    use_stemming: bool = False,
) -> str:
    """Use for ``data/Tweets.txt``: hashtags become words, mentions gone, then Arabic cleanup."""
    t = _twitter_handles_hashtags_and_noise(text)
    t = _coarse_arabic_pass(t)
    t = _emoji_then_punct_then_normalize(t, keep_emoji_sentiment=keep_emoji_sentiment)
    t = _deep_clean_arabic_surface(
        t,
        keep_emoji_sentiment=keep_emoji_sentiment,
        preserve_hashtag_words=True,
    )
    return _maybe_stem(t, use_stemming)


def preprocess_balanced_review(
    text: str,
    *,
    keep_emoji_sentiment: bool = True,
    use_stemming: bool = False,
) -> str:
    """Use for ``data/balanced-reviews.txt`` review column: strip pasted quotes, then Arabic cleanup."""
    t = strip_decorative_quotes(text)
    t = _coarse_arabic_pass(t)
    t = _emoji_then_punct_then_normalize(t, keep_emoji_sentiment=keep_emoji_sentiment)
    t = _deep_clean_arabic_surface(
        t,
        keep_emoji_sentiment=keep_emoji_sentiment,
        preserve_hashtag_words=False,
    )
    return _maybe_stem(t, use_stemming)


def preprocess_plain_text(
    text: str,
    *,
    keep_emoji_sentiment: bool = True,
    use_stemming: bool = False,
) -> str:
    """Generic Arabic clean (no tweet/review-specific tricks)."""
    t = _coarse_arabic_pass(text)
    t = _emoji_then_punct_then_normalize(t, keep_emoji_sentiment=keep_emoji_sentiment)
    t = _deep_clean_arabic_surface(
        t,
        keep_emoji_sentiment=keep_emoji_sentiment,
        preserve_hashtag_words=False,
    )
    return _maybe_stem(t, use_stemming)


def preprocess_tweets(
    texts: Iterable[str],
    *,
    keep_emoji_sentiment: bool = True,
    use_stemming: bool = False,
) -> List[str]:
    return [preprocess_tweet(t, keep_emoji_sentiment=keep_emoji_sentiment, use_stemming=use_stemming) for t in texts]


def preprocess_balanced_reviews(
    texts: Iterable[str],
    *,
    keep_emoji_sentiment: bool = True,
    use_stemming: bool = False,
) -> List[str]:
    return [
        preprocess_balanced_review(
            t,
            keep_emoji_sentiment=keep_emoji_sentiment,
            use_stemming=use_stemming,
        )
        for t in texts
    ]


def star_rating_to_binary(
    rating: str | int,
    *,
    neutral: Literal["drop", "neg", "pos"] = "drop",
) -> BinarySentiment | None:
    """Stars 4–5 → POS, 1–2 → NEG; 3 → depends on *neutral*."""
    s = str(rating).strip()
    if not s or not s.isdigit():
        return None
    star = int(s)
    if star in (4, 5):
        return "POS"
    if star in (1, 2):
        return "NEG"
    if star == 3:
        if neutral == "drop":
            return None
        if neutral == "neg":
            return "NEG"
        return "POS"
    return None


# --- backwards compatibility ---------------------------------------------------


def clean_text(
    text: str,
    keep_emoji_sentiment: bool = True,
    use_stemming: bool = False,
    *,
    for_tweets: bool = False,
    for_reviews: bool = False,
) -> str:
    if for_tweets:
        return preprocess_tweet(
            text,
            keep_emoji_sentiment=keep_emoji_sentiment,
            use_stemming=use_stemming,
        )
    if for_reviews:
        return preprocess_balanced_review(
            text,
            keep_emoji_sentiment=keep_emoji_sentiment,
            use_stemming=use_stemming,
        )
    return preprocess_plain_text(
        text,
        keep_emoji_sentiment=keep_emoji_sentiment,
        use_stemming=use_stemming,
    )


def clean_tweet_text(
    text: str,
    keep_emoji_sentiment: bool = True,
    use_stemming: bool = False,
) -> str:
    return preprocess_tweet(
        text,
        keep_emoji_sentiment=keep_emoji_sentiment,
        use_stemming=use_stemming,
    )


def clean_review_text(
    text: str,
    keep_emoji_sentiment: bool = True,
    use_stemming: bool = False,
) -> str:
    return preprocess_balanced_review(
        text,
        keep_emoji_sentiment=keep_emoji_sentiment,
        use_stemming=use_stemming,
    )


def preprocess_texts(
    texts: Iterable[str],
    keep_emoji_sentiment: bool = True,
    use_stemming: bool = False,
    *,
    for_tweets: bool = False,
    for_reviews: bool = False,
) -> List[str]:
    return [
        clean_text(
            t,
            keep_emoji_sentiment=keep_emoji_sentiment,
            use_stemming=use_stemming,
            for_tweets=for_tweets,
            for_reviews=for_reviews,
        )
        for t in texts
    ]
