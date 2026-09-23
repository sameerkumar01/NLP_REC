from collections import Counter

import numpy as np


def build_vocabulary(documents: list[list[str]], min_df: int = 1):
    document_frequency = Counter()
    for document in documents:
        document_frequency.update(set(document))
    terms = sorted(term for term, frequency in document_frequency.items() if frequency >= min_df)
    vocabulary = {term: index for index, term in enumerate(terms)}
    return vocabulary, document_frequency


def compute_idf(document_frequency: Counter, vocabulary: dict[str, int], document_count: int):
    idf = np.zeros(len(vocabulary), dtype=np.float64)
    for term, column in vocabulary.items():
        frequency = document_frequency[term]
        idf[column] = np.log((1 + document_count) / (1 + frequency)) + 1
    return idf


def build_tfidf_matrix(documents: list[list[str]], vocabulary: dict[str, int], idf):
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    for row, document in enumerate(documents):
        counts = Counter(document)
        for term, count in counts.items():
            column = vocabulary.get(term)
            if column is not None:
                matrix[row, column] = count * idf[column]
        norm = np.linalg.norm(matrix[row])
        if norm > 0:
            matrix[row] /= norm
    return matrix


def vectorize_query(tokens: list[str], vocabulary: dict[str, int], idf):
    vector = np.zeros(len(vocabulary), dtype=np.float64)
    counts = Counter(tokens)
    for term, count in counts.items():
        column = vocabulary.get(term)
        if column is not None:
            vector[column] = count * idf[column]
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def cosine_scores(query_vector, document_matrix):
    return document_matrix @ query_vector


def matrix_sparsity(matrix) -> float:
    return float(np.mean(matrix == 0)) if matrix.size else 0.0
