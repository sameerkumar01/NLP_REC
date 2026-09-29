from __future__ import annotations

from counts import NGramCounts


def continuation_probability(model: NGramCounts, word: str) -> float:
    if model.total_bigram_types == 0:
        total = sum(model.ngrams[1].values())
        return model.ngrams[1][(word,)] / max(total, 1)
    return len(model.predecessors[word]) / model.total_bigram_types


def kneser_ney_probability(model: NGramCounts, history: tuple[str, ...], word: str, discount: float = 0.75) -> float:
    word = word if word in model.prediction_vocabulary else "<UNK>"
    history = tuple(item if item in model.vocabulary or item == "<s>" else "<UNK>" for item in history)
    history = history[-(model.order - 1):] if model.order > 1 else ()
    return max(_recursive_probability(model, history, word, discount), 1e-12)


def _recursive_probability(model: NGramCounts, history: tuple[str, ...], word: str, discount: float) -> float:
    if not history:
        return continuation_probability(model, word)
    level = len(history) + 1
    context_total = model.context_totals[level][history]
    lower_history = history[1:]
    if context_total == 0:
        return _recursive_probability(model, lower_history, word, discount)
    count = model.ngrams[level][history + (word,)]
    first_term = max(count - discount, 0) / context_total
    unique_followers = len(model.followers[level][history])
    interpolation_weight = discount * unique_followers / context_total
    return first_term + interpolation_weight * _recursive_probability(model, lower_history, word, discount)
