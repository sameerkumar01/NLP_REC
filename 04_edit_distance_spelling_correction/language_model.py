from __future__ import annotations

import math
from collections import Counter


class UnigramLanguageModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.counts: Counter[str] = Counter()
        self.total = 0

    def fit(self, words: list[str]):
        self.counts = Counter(words)
        self.total = sum(self.counts.values())
        return self

    @property
    def vocabulary(self) -> set[str]:
        return set(self.counts)

    def probability(self, word: str) -> float:
        vocabulary_size = max(len(self.counts), 1)
        return (self.counts[word] + self.alpha) / (self.total + self.alpha * vocabulary_size)

    def log_probability(self, word: str) -> float:
        return math.log(self.probability(word))
