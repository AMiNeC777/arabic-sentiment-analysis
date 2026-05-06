"""ghofyy de moi"""
import numpy as np


class AraBERTEmbedding:
    """Placeholder AraBERT embedding extractor."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def transform(self, texts):
        # Replace with Hugging Face tokenizer/model forward pass.
        return np.zeros((len(texts), self.embedding_dim), dtype=float)
