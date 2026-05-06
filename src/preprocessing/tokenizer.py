from typing import List


def simple_tokenize(text: str) -> List[str]:
    """Whitespace tokenizer baseline."""
    if not isinstance(text, str):
        return []
    return [tok for tok in text.split() if tok]
