from __future__ import annotations

import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class BaselineNaiveBayes:
    def __init__(self, alpha: float = 1.0, min_df: int = 2, max_features: int | None = None):
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b[A-Za-z]{2,}\b", min_df=min_df, max_features=max_features)
        self.model = MultinomialNB(alpha=alpha)

    def fit(self, texts, labels):
        start = time.perf_counter()
        matrix = self.vectorizer.fit_transform(texts)
        self.model.fit(matrix, labels)
        self.fit_seconds = time.perf_counter() - start
        return self

    def predict(self, texts):
        return self.model.predict(self.vectorizer.transform(texts))
