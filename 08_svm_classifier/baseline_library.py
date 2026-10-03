from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC


class BaselineSVMs:
    def __init__(self, min_df: int = 2, max_features: int | None = None, c: float = 1.0):
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[A-Za-z]{2,}\b", min_df=min_df, max_features=max_features, smooth_idf=True, norm="l2")
        self.linear_model = LinearSVC(C=c, random_state=42)
        self.rbf_model = SVC(C=c, kernel="rbf", gamma="scale")

    def fit(self, texts, labels):
        matrix = self.vectorizer.fit_transform(texts)
        self.linear_model.fit(matrix, labels)
        self.rbf_model.fit(matrix, labels)
        return self

    def predict_linear(self, texts):
        return self.linear_model.predict(self.vectorizer.transform(texts))

    def predict_rbf(self, texts):
        return self.rbf_model.predict(self.vectorizer.transform(texts))
