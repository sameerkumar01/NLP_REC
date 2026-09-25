from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from language_model import UnigramLanguageModel
from spelling_corrector import NoisyChannelCorrector

EXAMPLES = [
    ("speling", "spelling"),
    ("recieve", "receive"),
    ("adress", "address"),
    ("seperate", "separate"),
    ("occured", "occurred"),
    ("definately", "definitely"),
    ("enviroment", "environment"),
]


def find_columns(frame: pd.DataFrame):
    names = {column: str(column).lower() for column in frame.columns}
    misspelled = [column for column, name in names.items() if any(word in name for word in ("miss", "wrong", "incorrect", "error"))]
    correct = [column for column, name in names.items() if any(word in name for word in ("correct", "right", "label", "target", "standard"))]
    if misspelled and correct and misspelled[0] != correct[0]:
        return misspelled[0], correct[0]
    if len(frame.columns) < 2:
        raise ValueError("The CSV must contain at least two columns")
    return frame.columns[0], frame.columns[1]


def load_pairs(csv_path: str | None, limit: int):
    if not csv_path:
        return EXAMPLES
    frame = pd.read_csv(csv_path, nrows=None if limit == 0 else limit)
    misspelled_column, correct_column = find_columns(frame)
    pairs = []
    for observed, target in zip(frame[misspelled_column], frame[correct_column]):
        observed = str(observed).strip().lower()
        target = str(target).strip().lower()
        if observed and target and observed.isalpha() and target.isalpha():
            pairs.append((observed, target))
    return pairs


def train_corrector(pairs, distance_penalty: float = 1.0, max_distance: int = 2):
    language_model = UnigramLanguageModel(alpha=1.0).fit([target for _, target in pairs])
    return NoisyChannelCorrector(language_model, distance_penalty, max_distance)


def evaluate(pairs, corrector):
    predictions = [corrector.correct(observed) for observed, _ in pairs]
    correct = sum(prediction == target for prediction, (_, target) in zip(predictions, pairs))
    return correct / max(len(pairs), 1), predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-distance", type=int, default=2)
    parser.add_argument("--distance-penalty", type=float, default=1.0)
    args = parser.parse_args()
    pairs = load_pairs(args.csv, args.limit)
    start = time.perf_counter()
    corrector = train_corrector(pairs, args.distance_penalty, args.max_distance)
    accuracy, predictions = evaluate(pairs, corrector)
    elapsed = time.perf_counter() - start
    print(f"pairs: {len(pairs)}")
    print(f"vocabulary_size: {len(corrector.language_model.vocabulary)}")
    print(f"exact_accuracy: {accuracy:.4f}")
    print(f"runtime_seconds: {elapsed:.4f}")
    print("examples:")
    for (observed, target), prediction in list(zip(pairs, predictions))[:20]:
        print(f"{observed} -> {prediction} expected={target}")


if __name__ == "__main__":
    main()
