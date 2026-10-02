from __future__ import annotations

from collections import Counter

import numpy as np

from text_utils import tokenize


class TfidfVectorizerScratch:
    def __init__(self, min_df: int = 1, max_features: int | None = None):
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}
        self.idf = None

    def fit(self, texts: list[str]):
        document_frequency = Counter()
        term_frequency = Counter()
        for text in texts:
            tokens = tokenize(text)
            document_frequency.update(set(tokens))
            term_frequency.update(tokens)
        terms = [term for term, frequency in document_frequency.items() if frequency >= self.min_df]
        terms.sort(key=lambda term: (-term_frequency[term], term))
        if self.max_features:
            terms = terms[:self.max_features]
        self.vocabulary = {term: index for index, term in enumerate(terms)}
        self.idf = np.zeros(len(terms), dtype=np.float64)
        document_count = len(texts)
        for term, column in self.vocabulary.items():
            self.idf[column] = np.log((1 + document_count) / (1 + document_frequency[term])) + 1
        return self

    def transform(self, texts: list[str]):
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float64)
        for row, text in enumerate(texts):
            counts = Counter(tokenize(text))
            for term, count in counts.items():
                column = self.vocabulary.get(term)
                if column is not None:
                    matrix[row, column] = count * self.idf[column]
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix

    def fit_transform(self, texts: list[str]):
        return self.fit(texts).transform(texts)
