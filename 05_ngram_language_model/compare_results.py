from __future__ import annotations

import argparse

from baseline_library import baseline_perplexity, train_baseline
from from_scratch import load_texts, split_sentences
from language_model import NGramLanguageModel
from tokenization import prepare_sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--min-count", type=int, default=2)
    args = parser.parse_args()
    sentences = prepare_sentences(load_texts(args.json, args.limit))
    train_sentences, test_sentences = split_sentences(sentences)
    scratch = NGramLanguageModel(args.order, args.min_count).fit(train_sentences)
    mapped_train = scratch.counts.training_sentences
    mapped_test = [scratch.counts.encode(sentence) for sentence in test_sentences]
    print("smoothing                      from_scratch             nltk")
    print("-" * 68)
    for smoothing in ("laplace", "kneser_ney"):
        scratch_value = scratch.perplexity(test_sentences, smoothing)
        baseline = train_baseline(mapped_train, args.order, smoothing)
        baseline_value = baseline_perplexity(baseline, mapped_test, args.order)
        print(f"{smoothing:28s} {scratch_value:>16.4f} {baseline_value:>16.4f}")
    print("scratch_laplace_generation:")
    print(scratch.generate("laplace", 30, seed=5))
    print("scratch_kneser_ney_generation:")
    print(scratch.generate("kneser_ney", 30, seed=5))


if __name__ == "__main__":
    main()
