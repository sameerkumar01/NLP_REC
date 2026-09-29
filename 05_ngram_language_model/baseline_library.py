from __future__ import annotations

import math

from nltk.lm import KneserNeyInterpolated, Laplace
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.util import everygrams


def train_baseline(sentences: list[list[str]], order: int = 3, smoothing: str = "laplace"):
    training_data, vocabulary_text = padded_everygram_pipeline(order, sentences)
    model = Laplace(order) if smoothing == "laplace" else KneserNeyInterpolated(order)
    model.fit(training_data, vocabulary_text)
    return model


def baseline_perplexity(model, sentences: list[list[str]], order: int):
    ngrams = []
    for sentence in sentences:
        padded = list(pad_both_ends(sentence, n=order))
        ngrams.extend(everygrams(padded, max_len=order))
    try:
        return model.perplexity(ngrams)
    except ZeroDivisionError:
        return math.inf
