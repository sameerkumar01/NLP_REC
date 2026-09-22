from __future__ import annotations

import argparse

from baseline_library import load_texts, run_baseline
from from_scratch import EXAMPLES, build_representation, load_texts as load_scratch_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    texts = load_scratch_texts(args.csv, args.limit)
    documents, scratch_vocabulary, scratch_matrix, _ = build_representation(texts)
    library_texts = load_texts(args.csv, args.limit, EXAMPLES)
    library_matrix, library_vocabulary, library_time = run_baseline(library_texts)
    scratch_nnz = int((scratch_matrix != 0).sum())
    library_nnz = int(library_matrix.nnz)
    print("metric                         from_scratch       sklearn")
    print("-" * 65)
    print(f"documents                     {len(texts):>16} {len(library_texts):>16}")
    print(f"unigram_vocabulary            {len(scratch_vocabulary):>16} {len(library_vocabulary):>16}")
    print(f"matrix_shape                  {str(scratch_matrix.shape):>16} {str(library_matrix.shape):>16}")
    print(f"nonzero_entries               {scratch_nnz:>16} {library_nnz:>16}")
    print(f"runtime_seconds               {'see scratch run':>16} {library_time:>16.4f}")
    print("scratch_first_document_tokens:")
    print(documents[0][:20])
    print("library_first_document_counts:")
    print(library_matrix[0].toarray()[0][:20])


if __name__ == "__main__":
    main()
