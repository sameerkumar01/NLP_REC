from __future__ import annotations

from collections import Counter

import numpy as np

from text_utils import tokenize


class CountVectorizerScratch:
    def __init__(self, min_df: int = 1, max_features: int | None = None):
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}

    def fit(self, texts: list[str]):
        document_frequency = Counter()
        term_frequency = Counter()
        for text in texts:
            tokens = tokenize(text)
            term_frequency.update(tokens)
            document_frequency.update(set(tokens))
        terms = [term for term, frequency in document_frequency.items() if frequency >= self.min_df]
        terms.sort(key=lambda term: (-term_frequency[term], term))
        if self.max_features:
            terms = terms[:self.max_features]
        self.vocabulary = {term: index for index, term in enumerate(terms)}
        return self

    def transform(self, texts: list[str]):
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=np.int32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                column = self.vocabulary.get(token)
                if column is not None:
                    matrix[row, column] += 1
        return matrix

    def fit_transform(self, texts: list[str]):
        return self.fit(texts).transform(texts)
