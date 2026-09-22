from __future__ import annotations

import argparse
import time
from collections import Counter

import pandas as pd

from bag_of_words import build_vocabulary, document_term_matrix, matrix_sparsity, top_features
from ngrams import format_ngram, ngram_counts

EXAMPLES = [
    "this movie is good and the acting is good",
    "this movie is not good but the ending is interesting",
    "the acting is interesting and the story is good",
]


def simple_tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.split()]


def load_texts(csv_path: str | None, limit: int) -> list[str]:
    if not csv_path:
        return EXAMPLES
    frame = pd.read_csv(csv_path, nrows=limit)
    if "review" not in frame.columns:
        raise ValueError("Expected a review column")
    return frame["review"].fillna("").astype(str).tolist()


def summarize(texts: list[str], documents: list[list[str]], matrix, vocabulary: dict[str, int], ngram_results: dict[int, Counter]):
    print(f"documents: {len(texts)}")
    print(f"vocabulary_size: {len(vocabulary)}")
    print(f"matrix_shape: {matrix.shape}")
    print(f"matrix_sparsity: {matrix_sparsity(matrix):.4f}")
    print(f"average_document_length: {sum(map(len, documents)) / max(len(documents), 1):.2f}")
    print("top_unigrams:")
    print(top_features(matrix, vocabulary, limit=10))
    for n, counts in ngram_results.items():
        top = [(format_ngram(gram), count) for gram, count in counts.most_common(10)]
        print(f"top_{n}grams:")
        print(top)


def build_representation(texts: list[str]):
    documents = [simple_tokenize(text) for text in texts]
    vocabulary = build_vocabulary(documents)
    matrix = document_term_matrix(documents, vocabulary)
    ngram_results = {}
    for n in (1, 2, 3):
        combined = Counter()
        for document in documents:
            combined.update(ngram_counts(document, n))
        ngram_results[n] = combined
    return documents, vocabulary, matrix, ngram_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    texts = load_texts(args.csv, args.limit)
    start = time.perf_counter()
    documents, vocabulary, matrix, ngram_results = build_representation(texts)
    elapsed = time.perf_counter() - start
    print(f"runtime_seconds: {elapsed:.4f}")
    summarize(texts, documents, matrix, vocabulary, ngram_results)
    print("first_document_tokens:")
    print(documents[0][:30])
    print("first_document_bigrams:")
    print([format_ngram(gram) for gram in list(ngram_counts(documents[0], 2))[:15]])


if __name__ == "__main__":
    main()
