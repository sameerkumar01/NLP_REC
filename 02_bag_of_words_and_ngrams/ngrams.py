from collections.abc import Iterable


def generate_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be at least 1")
    return [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]


def ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for gram in generate_ngrams(tokens, n):
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def format_ngram(gram: tuple[str, ...]) -> str:
    return " ".join(gram)
