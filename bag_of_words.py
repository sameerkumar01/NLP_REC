from collections.abc import Iterable


def build_vocabulary(documents: list[list[str]], min_count: int = 1) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for document in documents:
        for token in document:
            frequencies[token] = frequencies.get(token, 0) + 1
    vocabulary = sorted(token for token, count in frequencies.items() if count >= min_count)
    return {token: index for index, token in enumerate(vocabulary)}


def document_term_matrix(documents: list[list[str]], vocabulary: dict[str, int]):
    import numpy as np

    matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.int64)
    for row, document in enumerate(documents):
        for token in document:
            column = vocabulary.get(token)
            if column is not None:
                matrix[row, column] += 1
    return matrix


def matrix_sparsity(matrix) -> float:
    total = matrix.size
    zeros = int((matrix == 0).sum())
    return zeros / total if total else 0.0


def top_features(matrix, vocabulary: dict[str, int], limit: int = 20):
    totals = matrix.sum(axis=0)
    ordered = sorted(vocabulary.items(), key=lambda item: totals[item[1]], reverse=True)
    return [(token, int(totals[index])) for token, index in ordered[:limit]]
