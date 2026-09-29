from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from language_model import NGramLanguageModel
from tokenization import prepare_sentences

EXAMPLES = [
    "The city council approved the new housing plan. The city will build more homes.",
    "The football team won the final match. The team celebrated the championship.",
    "Scientists developed a new climate model. The model predicts rising temperatures.",
    "Technology companies released new artificial intelligence tools. The tools help developers.",
    "The central bank discussed interest rates. Markets watched the bank decision.",
    "Scientists study renewable energy. Renewable energy can reduce emissions.",
    "The football team prepared for the match. The match begins tomorrow.",
    "The city announced a new transport plan. The plan includes electric buses.",
]


def load_texts(json_path: str | None, limit: int) -> list[str]:
    if not json_path:
        return EXAMPLES
    rows = []
    with Path(json_path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit and index >= limit:
                break
            record = json.loads(line)
            rows.append(f"{record.get('headline', '')}. {record.get('short_description', '')}")
    return rows


def split_sentences(sentences: list[list[str]], test_fraction: float = 0.2, seed: int = 42):
    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)
    split = max(1, int(len(shuffled) * (1 - test_fraction)))
    split = min(split, len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
    return shuffled[:split], shuffled[split:] or shuffled[:1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--discount", type=float, default=0.75)
    parser.add_argument("--generate-length", type=int, default=30)
    args = parser.parse_args()
    texts = load_texts(args.json, args.limit)
    sentences = prepare_sentences(texts)
    train_sentences, test_sentences = split_sentences(sentences)
    model = NGramLanguageModel(args.order, args.min_count, args.alpha, args.discount).fit(train_sentences)
    print(f"documents: {len(texts)}")
    print(f"train_sentences: {len(train_sentences)}")
    print(f"test_sentences: {len(test_sentences)}")
    print(f"order: {args.order}")
    print(f"vocabulary_size: {len(model.counts.prediction_vocabulary)}")
    for smoothing in ("laplace", "kneser_ney"):
        perplexity = model.perplexity(test_sentences, smoothing)
        generated = model.generate(smoothing, args.generate_length, seed=5)
        print(f"{smoothing}_perplexity: {perplexity:.4f}")
        print(f"{smoothing}_generation: {generated}")


if __name__ == "__main__":
    main()
