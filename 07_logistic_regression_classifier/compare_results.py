from __future__ import annotations

import argparse

from baseline_library import BaselineLogisticRegression
from from_scratch import evaluate_model, load_dataset, stratified_split, train_model
from metrics import classification_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int)
    args = parser.parse_args()
    rows = load_dataset(args.csv, args.limit)
    train_rows, test_rows = stratified_split(rows)
    train_texts = [text for _, text in train_rows]
    train_labels = [1 if label == "spam" else 0 for label, _ in train_rows]
    test_texts = [text for _, text in test_rows]
    test_labels = [1 if label == "spam" else 0 for label, _ in test_rows]
    scratch_vectorizer, scratch_model = train_model(train_rows, args.learning_rate, args.epochs, args.l2, args.min_df, args.max_features)
    scratch_metrics, _, _ = evaluate_model(test_rows, scratch_vectorizer, scratch_model)
    baseline = BaselineLogisticRegression(args.min_df, args.max_features).fit(train_texts, train_labels)
    baseline_predictions = baseline.predict(test_texts).tolist()
    baseline_metrics = classification_metrics(test_labels, baseline_predictions)
    print("metric                         from_scratch          sklearn")
    print("-" * 68)
    print(f"vocabulary_size               {len(scratch_vectorizer.vocabulary):>16} {len(baseline.vectorizer.vocabulary_):>16}")
    print(f"training_iterations           {len(scratch_model.loss_history):>16} {'library solver':>16}")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"{name:30s} {scratch_metrics[name]:>16.4f} {baseline_metrics[name]:>16.4f}")
    print("from_scratch_confusion:")
    print([[scratch_metrics["true_negative"], scratch_metrics["false_positive"]], [scratch_metrics["false_negative"], scratch_metrics["true_positive"]]])
    print("sklearn_confusion:")
    print([[baseline_metrics["true_negative"], baseline_metrics["false_positive"]], [baseline_metrics["false_negative"], baseline_metrics["true_positive"]]])


if __name__ == "__main__":
    main()
