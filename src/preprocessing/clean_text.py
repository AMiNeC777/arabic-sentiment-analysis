import re


def clean_text(text: str) -> str:
    """Basic Arabic text cleaning."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
