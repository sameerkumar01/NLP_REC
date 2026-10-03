from __future__ import annotations

import argparse
import random
import time

import pandas as pd

from linear_svm import LinearSVMScratch
from metrics import classification_metrics
from tfidf_vectorizer import TfidfVectorizerScratch

EXAMPLES = [
    ("ham", "Are we still meeting for lunch today"),
    ("ham", "Please call me when you reach home"),
    ("ham", "Can you send the project report tonight"),
    ("ham", "Happy birthday hope you have a wonderful day"),
    ("ham", "The meeting has moved to three pm"),
    ("ham", "I will pick up groceries after work"),
    ("ham", "Thanks for helping me with the assignment"),
    ("ham", "Your appointment is confirmed for Monday"),
    ("ham", "Dinner is ready when will you arrive"),
    ("ham", "Let us review the notes tomorrow morning"),
    ("spam", "Win cash now claim your free prize"),
    ("spam", "Congratulations you won a free holiday click now"),
    ("spam", "Urgent claim your reward before it expires"),
    ("spam", "Free entry in our weekly prize draw"),
    ("spam", "You have won cash text WIN to claim"),
    ("spam", "Exclusive offer get free credit today"),
    ("spam", "Claim your bonus prize immediately"),
    ("spam", "Winner call now to receive your reward"),
    ("spam", "Free cash waiting reply now to collect"),
    ("spam", "Limited offer claim your free reward today"),
]


def find_columns(frame: pd.DataFrame):
    names = {column: str(column).lower() for column in frame.columns}
    label_candidates = [column for column, name in names.items() if name in {"v1", "label", "category", "class", "target"}]
    text_candidates = [column for column, name in names.items() if name in {"v2", "text", "message", "sms", "content"}]
    if label_candidates and text_candidates:
        return label_candidates[0], text_candidates[0]
    usable = [column for column in frame.columns if not str(column).lower().startswith("unnamed")]
    if len(usable) < 2:
        raise ValueError("The CSV must contain label and message columns")
    return usable[0], usable[1]


def load_dataset(csv_path: str | None, limit: int):
    if not csv_path:
        return EXAMPLES
    frame = pd.read_csv(csv_path, encoding="latin-1", nrows=None if limit == 0 else limit)
    label_column, text_column = find_columns(frame)
    rows = []
    for label, text in zip(frame[label_column], frame[text_column]):
        label = str(label).strip().lower()
        text = str(text).strip()
        if label in {"ham", "spam"} and text:
            rows.append((label, text))
    return rows


def stratified_split(rows, test_size: float = 0.2, seed: int = 42):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)
    train = []
    test = []
    generator = random.Random(seed)
    for group in grouped.values():
        generator.shuffle(group)
        test_count = max(1, int(len(group) * test_size))
        test.extend(group[:test_count])
        train.extend(group[test_count:])
    generator.shuffle(train)
    generator.shuffle(test)
    return train, test


def train_model(train_rows, learning_rate: float = 0.1, epochs: int = 500, regularization: float = 0.01, c: float = 1.0, min_df: int = 2, max_features: int | None = None):
    texts = [text for _, text in train_rows]
    labels = [1 if label == "spam" else -1 for label, _ in train_rows]
    vectorizer = TfidfVectorizerScratch(min_df, max_features)
    matrix = vectorizer.fit_transform(texts)
    model = LinearSVMScratch(learning_rate, epochs, regularization, c).fit(matrix, labels)
    return vectorizer, model


def evaluate_model(test_rows, vectorizer, model):
    texts = [text for _, text in test_rows]
    labels = [1 if label == "spam" else -1 for label, _ in test_rows]
    matrix = vectorizer.transform(texts)
    scores = model.decision_function(matrix)
    predictions = model.predict(matrix).tolist()
    return classification_metrics(labels, predictions), predictions, scores.tolist()


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
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    rows = load_dataset(args.csv, args.limit)
    train_rows, test_rows = stratified_split(rows, args.test_size)
    start = time.perf_counter()
    vectorizer, model = train_model(train_rows, args.learning_rate, args.epochs, args.regularization, args.c, args.min_df, args.max_features)
    metrics, predictions, scores = evaluate_model(test_rows, vectorizer, model)
    elapsed = time.perf_counter() - start
    print(f"rows: {len(rows)}")
    print(f"train_rows: {len(train_rows)}")
    print(f"test_rows: {len(test_rows)}")
    print(f"vocabulary_size: {len(vectorizer.vocabulary)}")
    print(f"epochs_completed: {len(model.loss_history)}")
    print(f"initial_loss: {model.loss_history[0]:.6f}")
    print(f"final_loss: {model.loss_history[-1]:.6f}")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"{name}: {metrics[name]:.4f}")
    print(f"confusion_matrix: [[{metrics['true_negative']}, {metrics['false_positive']}], [{metrics['false_negative']}, {metrics['true_positive']}]]")
    print(f"runtime_seconds: {elapsed:.4f}")
    print("predictions:")
    for (label, text), prediction, score in zip(test_rows[:20], predictions[:20], scores[:20]):
        predicted_label = "spam" if prediction == 1 else "ham"
        print(f"score={score:.4f} predicted={predicted_label} actual={label} text={text[:100]}")


if __name__ == "__main__":
    main()
