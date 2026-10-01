from __future__ import annotations

import argparse

from baseline_library import BaselineNaiveBayes
from from_scratch import evaluate_model, load_dataset, stratified_split, train_model
from metrics import classification_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int)
    args = parser.parse_args()
    rows = load_dataset(args.csv, args.limit)
    train_rows, test_rows = stratified_split(rows)
    train_labels = [label for label, _ in train_rows]
    train_texts = [text for _, text in train_rows]
    test_labels = [label for label, _ in test_rows]
    test_texts = [text for _, text in test_rows]
    scratch_vectorizer, scratch_model = train_model(train_rows, args.alpha, args.min_df, args.max_features)
    scratch_metrics, _ = evaluate_model(test_rows, scratch_vectorizer, scratch_model)
    baseline = BaselineNaiveBayes(args.alpha, args.min_df, args.max_features).fit(train_texts, train_labels)
    baseline_predictions = baseline.predict(test_texts).tolist()
    baseline_metrics = classification_metrics(test_labels, baseline_predictions)
    print("metric                         from_scratch          sklearn")
    print("-" * 68)
    print(f"vocabulary_size               {len(scratch_vectorizer.vocabulary):>16} {len(baseline.vectorizer.vocabulary_):>16}")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"{name:30s} {scratch_metrics[name]:>16.4f} {baseline_metrics[name]:>16.4f}")
    print("from_scratch_confusion:")
    print([[scratch_metrics["true_negative"], scratch_metrics["false_positive"]], [scratch_metrics["false_negative"], scratch_metrics["true_positive"]]])
    print("sklearn_confusion:")
    print([[baseline_metrics["true_negative"], baseline_metrics["false_positive"]], [baseline_metrics["false_negative"], baseline_metrics["true_positive"]]])


if __name__ == "__main__":
    main()
