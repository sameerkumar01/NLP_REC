from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from retrieval import rank_documents
from text_utils import tokenize
from tfidf import build_tfidf_matrix, build_vocabulary, compute_idf, matrix_sparsity

EXAMPLES = [
    "Scientists study renewable energy and climate policy for a cleaner future.",
    "The football team won after a late goal in the championship match.",
    "Technology companies release new artificial intelligence models and chips.",
    "Markets reacted to inflation data and the central bank interest rate decision.",
]


def load_documents(json_path: str | None, limit: int) -> list[str]:
    if not json_path:
        return EXAMPLES
    path = Path(json_path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, nrows=None if limit == 0 else limit)
        headline = frame.get("headline", "").fillna("").astype(str)
        description = frame.get("short_description", "").fillna("").astype(str)
        return (headline + " " + description).tolist()
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if limit and line_number >= limit:
                break
            record = json.loads(line)
            rows.append(f"{record.get('headline', '')} {record.get('short_description', '')}".strip())
    return rows


def build_index(documents: list[str]):
    tokenized_documents = [tokenize(document) for document in documents]
    vocabulary, document_frequency = build_vocabulary(tokenized_documents)
    idf = compute_idf(document_frequency, vocabulary, len(documents))
    matrix = build_tfidf_matrix(tokenized_documents, vocabulary, idf)
    return tokenized_documents, vocabulary, idf, matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    documents = load_documents(args.json, args.limit)
    start = time.perf_counter()
    tokenized_documents, vocabulary, idf, matrix = build_index(documents)
    elapsed = time.perf_counter() - start
    query = args.query or documents[0]
    results = rank_documents(query, documents, tokenized_documents, vocabulary, idf, matrix, args.top_k)
    print(f"documents: {len(documents)}")
    print(f"vocabulary_size: {len(vocabulary)}")
    print(f"matrix_shape: {matrix.shape}")
    print(f"matrix_sparsity: {matrix_sparsity(matrix):.4f}")
    print(f"runtime_seconds: {elapsed:.4f}")
    print(f"query: {query}")
    print("top_results:")
    for rank, (index, score, document) in enumerate(results, 1):
        print(f"{rank}. score={score:.4f} index={index} text={document[:240]}")


if __name__ == "__main__":
    main()
