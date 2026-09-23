from __future__ import annotations

import time

from sklearn.feature_extraction.text import TfidfVectorizer


def build_library_index(documents: list[str]):
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[A-Za-z]{2,}\b", lowercase=True, smooth_idf=True, norm="l2")
    start = time.perf_counter()
    matrix = vectorizer.fit_transform(documents)
    elapsed = time.perf_counter() - start
    return vectorizer, matrix, elapsed


def rank_library(query: str, vectorizer, matrix, top_k: int = 5):
    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    order = scores.argsort()[::-1][:top_k]
    return [(int(index), float(scores[index])) for index in order]
