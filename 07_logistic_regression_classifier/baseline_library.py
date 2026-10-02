from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class BaselineLogisticRegression:
    def __init__(self, min_df: int = 2, max_features: int | None = None):
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[A-Za-z]{2,}\b", min_df=min_df, max_features=max_features, smooth_idf=True, norm="l2")
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def fit(self, texts, labels):
        matrix = self.vectorizer.fit_transform(texts)
        self.model.fit(matrix, labels)
        return self

    def predict(self, texts):
        matrix = self.vectorizer.transform(texts)
        return self.model.predict(matrix)

    def predict_proba(self, texts):
        matrix = self.vectorizer.transform(texts)
        return self.model.predict_proba(matrix)[:, 1]
