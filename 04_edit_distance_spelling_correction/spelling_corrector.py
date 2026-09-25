from __future__ import annotations

from edit_distance import edits_within_distance, levenshtein_distance
from language_model import UnigramLanguageModel


class NoisyChannelCorrector:
    def __init__(self, language_model: UnigramLanguageModel, distance_penalty: float = 1.0, max_distance: int = 2):
        self.language_model = language_model
        self.distance_penalty = distance_penalty
        self.max_distance = max_distance

    def candidates(self, observed: str) -> set[str]:
        word = observed.lower()
        return edits_within_distance(word, self.language_model.vocabulary, self.max_distance)

    def score(self, observed: str, candidate: str) -> float:
        distance = levenshtein_distance(observed.lower(), candidate)
        return self.language_model.log_probability(candidate) - self.distance_penalty * distance

    def rank(self, observed: str):
        candidates = self.candidates(observed)
        ranked = sorted(candidates, key=lambda candidate: self.score(observed, candidate), reverse=True)
        return [(candidate, self.score(observed, candidate)) for candidate in ranked]

    def correct(self, observed: str) -> str:
        ranked = self.rank(observed)
        return ranked[0][0] if ranked else observed.lower()
