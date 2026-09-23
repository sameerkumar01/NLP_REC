from __future__ import annotations

import argparse

from baseline_library import build_library_index, rank_library
from from_scratch import build_index, load_documents
from retrieval import rank_documents
from text_utils import tokenize
from tfidf import matrix_sparsity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    documents = load_documents(args.json, args.limit)
    tokenized_documents, vocabulary, idf, scratch_matrix = build_index(documents)
    query = args.query or documents[0]
    scratch_results = rank_documents(query, documents, tokenized_documents, vocabulary, idf, scratch_matrix, args.top_k)
    vectorizer, library_matrix, library_time = build_library_index(documents)
    library_results = rank_library(query, vectorizer, library_matrix, args.top_k)
    print("metric                         from_scratch       sklearn")
    print("-" * 65)
    print(f"documents                     {len(documents):>16} {library_matrix.shape[0]:>16}")
    print(f"vocabulary_size               {len(vocabulary):>16} {len(vectorizer.vocabulary_):>16}")
    print(f"matrix_shape                  {str(scratch_matrix.shape):>16} {str(library_matrix.shape):>16}")
    print(f"matrix_sparsity               {matrix_sparsity(scratch_matrix):>16.4f} {1 - library_matrix.nnz / max(library_matrix.shape[0] * library_matrix.shape[1], 1):>16.4f}")
    print(f"runtime_seconds               {'see scratch run':>16} {library_time:>16.4f}")
    print(f"query: {query}")
    print("scratch_results:")
    for rank, (index, score, document) in enumerate(scratch_results, 1):
        print(f"{rank}. score={score:.4f} index={index} text={document[:180]}")
    print("library_results:")
    for rank, (index, score) in enumerate(library_results, 1):
        print(f"{rank}. score={score:.4f} index={index}")


if __name__ == "__main__":
    main()
