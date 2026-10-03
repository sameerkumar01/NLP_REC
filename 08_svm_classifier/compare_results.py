from __future__ import annotations

import argparse

from baseline_library import BaselineSVMs
from from_scratch import evaluate_model, load_dataset, stratified_split, train_model
from metrics import classification_metrics


def format_metric(name, scratch, linear, rbf):
    print(f"{name:24s} {scratch:>14.4f} {linear:>14.4f} {rbf:>14.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int)
    args = parser.parse_args()
    rows = load_dataset(args.csv, args.limit)
    train_rows, test_rows = stratified_split(rows)
    train_texts = [text for _, text in train_rows]
    train_labels = [1 if label == "spam" else -1 for label, _ in train_rows]
    test_texts = [text for _, text in test_rows]
    test_labels = [1 if label == "spam" else -1 for label, _ in test_rows]
    scratch_vectorizer, scratch_model = train_model(train_rows, args.learning_rate, args.epochs, args.regularization, args.c, args.min_df, args.max_features)
    scratch_metrics, _, _ = evaluate_model(test_rows, scratch_vectorizer, scratch_model)
    baseline = BaselineSVMs(args.min_df, args.max_features, args.c).fit(train_texts, train_labels)
    linear_metrics = classification_metrics(test_labels, baseline.predict_linear(test_texts).tolist())
    rbf_metrics = classification_metrics(test_labels, baseline.predict_rbf(test_texts).tolist())
    print("metric                     scratch_linear sklearn_linear    sklearn_rbf")
    print("-" * 76)
    print(f"vocabulary_size            {len(scratch_vectorizer.vocabulary):>14} {len(baseline.vectorizer.vocabulary_):>14} {len(baseline.vectorizer.vocabulary_):>14}")
    for name in ("accuracy", "precision", "recall", "f1"):
        format_metric(name, scratch_metrics[name], linear_metrics[name], rbf_metrics[name])
    print("scratch_confusion:")
    print([[scratch_metrics["true_negative"], scratch_metrics["false_positive"]], [scratch_metrics["false_negative"], scratch_metrics["true_positive"]]])
    print("sklearn_linear_confusion:")
    print([[linear_metrics["true_negative"], linear_metrics["false_positive"]], [linear_metrics["false_negative"], linear_metrics["true_positive"]]])
    print("sklearn_rbf_confusion:")
    print([[rbf_metrics["true_negative"], rbf_metrics["false_positive"]], [rbf_metrics["false_negative"], rbf_metrics["true_positive"]]])


if __name__ == "__main__":
    main()
