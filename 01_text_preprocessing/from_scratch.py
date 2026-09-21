"""Run the transparent preprocessing pipeline and print diagnostics."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

try:
    from regex_cleaner import clean_text
    from tokenizer import tokenize
    from stopwords import remove_stopwords
    from porter_stemmer import PorterStemmer
    from lemmatizer import lemmatize
except ImportError:  # Allows importing as a package from the repository root.
    from .regex_cleaner import clean_text
    from .tokenizer import tokenize
    from .stopwords import remove_stopwords
    from .porter_stemmer import PorterStemmer
    from .lemmatizer import lemmatize

EXAMPLES = [
    "<b>This movie was AMAZING!</b> I wasn't expecting it to be so moving. https://example.com",
    "The actors' performances were better than I expected, and the story kept me watching.",
    "A slow start... but children, mice, and other details made the ending memorable.",
]


def preprocess(text: str, stemmer: PorterStemmer | None = None) -> dict[str, list[str] | str]:
    """Apply each transformation separately so intermediate state is inspectable."""
    stemmer = stemmer or PorterStemmer()
    cleaned = clean_text(text)
    raw_tokens = tokenize(cleaned)
    filtered_tokens = remove_stopwords(raw_tokens)
    stems = [stemmer.stem(token) for token in filtered_tokens]
    lemmas = [lemmatize(token) for token in filtered_tokens]
    return {
        "cleaned": cleaned,
        "raw_tokens": raw_tokens,
        "filtered_tokens": filtered_tokens,
        "stems": stems,
        "lemmas": lemmas,
    }


def load_texts(csv_path: str | None, limit: int) -> list[str]:
    if not csv_path:
        return EXAMPLES
    frame = pd.read_csv(csv_path, nrows=limit)
    if "review" not in frame.columns:
        raise ValueError("Expected a 'review' column in the IMDB CSV")
    return frame["review"].fillna("").astype(str).tolist()


def summarize(results: list[dict]) -> dict[str, float]:
    raw = sum(len(item["raw_tokens"]) for item in results)
    filtered = sum(len(item["filtered_tokens"]) for item in results)
    stems = {token for item in results for token in item["stems"]}
    lemmas = {token for item in results for token in item["lemmas"]}
    return {
        "documents": len(results),
        "raw_tokens": raw,
        "filtered_tokens": filtered,
        "stopword_removal_percent": 100 * (raw - filtered) / max(raw, 1),
        "stem_vocabulary": len(stems),
        "lemma_vocabulary": len(lemmas),
        "avg_filtered_tokens": filtered / max(len(results), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to IMDB Dataset.csv; omit for built-in examples")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    texts = load_texts(args.csv, args.limit)
    start = time.perf_counter()
    results = [preprocess(text) for text in texts]
    elapsed = time.perf_counter() - start

    print(f"Processed {len(results)} document(s) in {elapsed:.4f}s")
    print("Diagnostics:")
    for key, value in summarize(results).items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    print("\nFirst examples (cleaned -> filtered -> stemmed -> lemmatized):")
    for original, result in list(zip(texts, results))[:3]:
        print(f"\nOriginal: {original[:160]}")
        print("Cleaned:", result["cleaned"][:160])
        print("Filtered:", result["filtered_tokens"][:20])
        print("Stems:", result["stems"][:20])
        print("Lemmas:", result["lemmas"][:20])


if __name__ == "__main__":
    main()
