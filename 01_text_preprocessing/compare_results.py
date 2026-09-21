"""Compare diagnostics from the scratch pipeline and NLTK baseline."""
from __future__ import annotations

import argparse
import time

from from_scratch import EXAMPLES, load_texts, preprocess, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    texts = load_texts(args.csv, args.limit)

    start = time.perf_counter()
    scratch = [preprocess(text) for text in texts]
    scratch_time = time.perf_counter() - start

    try:
        from baseline_library import preprocess_with_nltk
        start = time.perf_counter()
        library = [preprocess_with_nltk(text) for text in texts]
        library_time = time.perf_counter() - start
    except Exception as error:
        print(f"NLTK baseline could not run: {error}")
        print("Install requirements and allow NLTK data downloads, then retry.")
        return

    print("Metric                         From scratch       NLTK baseline")
    print("-" * 70)
    left, right = summarize(scratch), summarize(library)
    for key in left:
        print(f"{key:30s} {str(round(left[key], 3)):>16s} {str(round(right[key], 3)):>16s}")
    print(f"{'runtime_seconds':30s} {scratch_time:>16.4f} {library_time:>16.4f}")

    print("\nOne-document token comparison:")
    print("Scratch filtered:", scratch[0]["filtered_tokens"][:20])
    print("NLTK filtered:  ", library[0]["filtered_tokens"][:20])
    print("\nInterpretation: differences are expected because the scratch tokenizer, stopword list, and rule lemmatizer are intentionally small and readable.")


if __name__ == "__main__":
    main()
