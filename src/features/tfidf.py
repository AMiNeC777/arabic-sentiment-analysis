from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFVectorizer:
    """TF-IDF vectorizer wrapper."""

    def __init__(self, max_features: int = 20000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)
