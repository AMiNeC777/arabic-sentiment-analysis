import numpy as np


class FastTextEmbedding:
    """Placeholder fastText embedding extractor."""

    def __init__(self, embedding_dim: int = 300):
        self.embedding_dim = embedding_dim

    def transform(self, texts):
        # Replace with loading real fastText vectors and averaging tokens.
        return np.zeros((len(texts), self.embedding_dim), dtype=float)
