from __future__ import annotations

from counts import NGramCounts


def laplace_probability(model: NGramCounts, history: tuple[str, ...], word: str, alpha: float = 1.0) -> float:
    word = word if word in model.prediction_vocabulary else "<UNK>"
    history = tuple(item if item in model.vocabulary or item == "<s>" else "<UNK>" for item in history)
    history = history[-(model.order - 1):] if model.order > 1 else ()
    vocabulary_size = len(model.prediction_vocabulary)
    count = model.ngram_count(history, word)
    context_total = model.context_count(history)
    return (count + alpha) / (context_total + alpha * vocabulary_size)
