from __future__ import annotations

import time

from spellchecker import SpellChecker


def run_baseline(pairs):
    checker = SpellChecker()
    start = time.perf_counter()
    predictions = [checker.correction(observed) or observed for observed, _ in pairs]
    elapsed = time.perf_counter() - start
    correct = sum(prediction == target for prediction, (_, target) in zip(predictions, pairs))
    accuracy = correct / max(len(pairs), 1)
    return accuracy, predictions, elapsed
