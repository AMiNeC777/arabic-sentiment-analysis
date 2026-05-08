from __future__ import annotations

from pathlib import Path

from gensim.models import FastText
import numpy as np
from typing import Iterable
from transformers import AutoModel, AutoTokenizer
import torch

ARABERT_NAME = "aubmindlab/bert-base-arabertv02"
_ARABERT_TOKENIZER = None
_ARABERT_MODEL = None

# ---------- FastText  ----------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = _PROJECT_ROOT / "data" / "embeddings" / "aravec" / "full_grams_cbow_300_twitter.mdl"


def load_aravec_model(path: Path | str | None = None) -> FastText:
    p = Path(path) if path is not None else MODEL_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"Model not found: {p.resolve()}. Place AraVec files under data/embeddings/aravec/"
        )
    return FastText.load(str(p))

def _tokenize_arabic(text: str) -> list[str]:
    return text.split()

def _document_vector(model: FastText,tokens: list[str]) -> np.ndarray:
    dim=model.vector_size
    if not tokens:
        return np.zeros(dim, dtype=np.float32)

    vectors = []
    for token in tokens:
        try:
            vectors.append(model.wv[token])
        except KeyError:
            continue
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

def extract_fasttext_features(
    texts: Iterable[str],
    model: FastText | None = None,
) -> np.ndarray:
    model = model or load_aravec_model()
    
    rows = [_document_vector(model, _tokenize_arabic(text)) for text in texts]
    return np.vstack(rows)

# ---------- AraBERT  ----------

def _load_arabert():
    global _ARABERT_TOKENIZER, _ARABERT_MODEL
    if _ARABERT_TOKENIZER is None or _ARABERT_MODEL is None:
        _ARABERT_TOKENIZER = AutoTokenizer.from_pretrained(ARABERT_NAME)
        _ARABERT_MODEL = AutoModel.from_pretrained(ARABERT_NAME)
        _ARABERT_MODEL.eval()
        if torch.cuda.is_available():
            _ARABERT_MODEL= _ARABERT_MODEL.to("cuda")
    return _ARABERT_TOKENIZER, _ARABERT_MODEL

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token vectors per sentence, ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()                 # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)              # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                    # (B, 1)
    return summed / counts   

def extract_arabert_features(
    texts: Iterable[str],
    batch_size: int = 16,
    max_length: int = 128,
) -> np.ndarray:
    """Encode texts with AraBERT (mean-pooled). Returns (n_texts, 768) float32."""
    tokenizer, model = _load_arabert()
    device = next(model.parameters()).device
    texts = list(texts)
    all_vecs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            all_vecs.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(all_vecs)


if __name__ == "__main__":
    sample_texts = [
        "الفيلم جميل جدا",
        "لم يعجبني هذا الكتاب",
        "",
    ]
    X_ft = extract_fasttext_features(sample_texts)
    print("fasttext shape:", X_ft.shape)            # (3, 300)
    X_bert = extract_arabert_features(sample_texts)
    print("arabert shape: ", X_bert.shape)          # (3, 768)