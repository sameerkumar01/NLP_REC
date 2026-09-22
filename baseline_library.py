from __future__ import annotations

import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def load_texts(csv_path: str | None, limit: int, examples: list[str]) -> list[str]:
    if not csv_path:
        return examples
    frame = pd.read_csv(csv_path, nrows=limit)
    return frame["review"].fillna("").astype(str).tolist()


def run_baseline(texts: list[str], ngram_range: tuple[int, int] = (1, 1)):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    start = time.perf_counter()
    matrix = vectorizer.fit_transform(texts)
    elapsed = time.perf_counter() - start
    vocabulary = vectorizer.vocabulary_
    return matrix, vocabulary, elapsed
