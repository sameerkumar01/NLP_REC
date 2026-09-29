from __future__ import annotations

import math
import random

from counts import NGramCounts
from kneser_ney import kneser_ney_probability
from laplace import laplace_probability


class NGramLanguageModel:
    def __init__(self, order: int = 3, min_count: int = 2, alpha: float = 1.0, discount: float = 0.75):
        self.counts = NGramCounts(order, min_count)
        self.alpha = alpha
        self.discount = discount

    @property
    def order(self):
        return self.counts.order

    def fit(self, sentences: list[list[str]]):
        self.counts.fit(sentences)
        return self

    def probability(self, history: list[str] | tuple[str, ...], word: str, smoothing: str = "laplace") -> float:
        history_tuple = tuple(history)[-(self.order - 1):] if self.order > 1 else ()
        if smoothing == "laplace":
            return laplace_probability(self.counts, history_tuple, word, self.alpha)
        if smoothing == "kneser_ney":
            return kneser_ney_probability(self.counts, history_tuple, word, self.discount)
        raise ValueError("smoothing must be laplace or kneser_ney")

    def perplexity(self, sentences: list[list[str]], smoothing: str = "laplace") -> float:
        log_probability = 0.0
        prediction_count = 0
        for sentence in sentences:
            encoded = self.counts.encode(sentence)
            padded = ["<s>"] * (self.order - 1) + encoded + ["</s>"]
            for index in range(self.order - 1, len(padded)):
                history = padded[index - self.order + 1:index] if self.order > 1 else []
                word = padded[index]
                probability = max(self.probability(history, word, smoothing), 1e-12)
                log_probability += math.log(probability)
                prediction_count += 1
        return math.exp(-log_probability / max(prediction_count, 1))

    def generate(self, smoothing: str = "laplace", max_tokens: int = 30, seed: int = 42) -> str:
        generator = random.Random(seed)
        history = ["<s>"] * (self.order - 1)
        output = []
        candidates = sorted(self.counts.prediction_vocabulary - {"<UNK>"})
        for _ in range(max_tokens):
            probabilities = [self.probability(history, word, smoothing) for word in candidates]
            total = sum(probabilities)
            if total <= 0:
                break
            normalized = [probability / total for probability in probabilities]
            word = generator.choices(candidates, weights=normalized, k=1)[0]
            if word == "</s>":
                break
            output.append(word)
            if self.order > 1:
                history = (history + [word])[-(self.order - 1):]
        return " ".join(output)
