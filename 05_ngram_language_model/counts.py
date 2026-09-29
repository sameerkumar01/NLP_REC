from __future__ import annotations

from collections import Counter, defaultdict


class NGramCounts:
    def __init__(self, order: int = 3, min_count: int = 2):
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.min_count = min_count
        self.ngrams = {level: Counter() for level in range(1, order + 1)}
        self.context_totals = {level: Counter() for level in range(2, order + 1)}
        self.followers = {level: defaultdict(set) for level in range(2, order + 1)}
        self.predecessors = defaultdict(set)
        self.vocabulary = {"<UNK>", "</s>"}
        self.prediction_vocabulary = {"<UNK>", "</s>"}
        self.total_bigram_types = 0
        self.training_sentences = []

    def fit(self, sentences: list[list[str]]):
        word_counts = Counter(word for sentence in sentences for word in sentence)
        self.vocabulary.update(word for word, count in word_counts.items() if count >= self.min_count)
        self.prediction_vocabulary = self.vocabulary - {"<s>"}
        self.training_sentences = [self.encode(sentence) for sentence in sentences]
        for sentence in self.training_sentences:
            padded = ["<s>"] * (self.order - 1) + sentence + ["</s>"]
            for level in range(1, self.order + 1):
                for index in range(len(padded) - level + 1):
                    gram = tuple(padded[index:index + level])
                    if level == 1 and gram[0] == "<s>":
                        continue
                    self.ngrams[level][gram] += 1
                    if level >= 2:
                        history = gram[:-1]
                        word = gram[-1]
                        self.context_totals[level][history] += 1
                        self.followers[level][history].add(word)
                        if level == 2:
                            self.predecessors[word].add(history[0])
        self.total_bigram_types = len(self.ngrams.get(2, {}))
        return self

    def encode(self, sentence: list[str]) -> list[str]:
        return [word if word in self.vocabulary else "<UNK>" for word in sentence]

    def ngram_count(self, history: tuple[str, ...], word: str) -> int:
        level = len(history) + 1
        return self.ngrams[level][history + (word,)]

    def context_count(self, history: tuple[str, ...]) -> int:
        if not history:
            return sum(self.ngrams[1].values())
        return self.context_totals[len(history) + 1][history]
