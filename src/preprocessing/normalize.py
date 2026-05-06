import re


def normalize_arabic(text: str) -> str:
    """Normalize common Arabic character variants."""
    if not isinstance(text, str):
        return ""

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ة": "ه",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Remove Arabic diacritics.
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
    return text
