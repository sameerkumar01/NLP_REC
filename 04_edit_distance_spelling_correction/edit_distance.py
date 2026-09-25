from __future__ import annotations

import string


def levenshtein_table(source: str, target: str):
    rows = len(source) + 1
    columns = len(target) + 1
    table = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        table[row][0] = row
    for column in range(columns):
        table[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            insertion = table[row][column - 1] + 1
            deletion = table[row - 1][column] + 1
            substitution = table[row - 1][column - 1] + (source[row - 1] != target[column - 1])
            table[row][column] = min(insertion, deletion, substitution)
    return table


def levenshtein_distance(source: str, target: str) -> int:
    return levenshtein_table(source, target)[-1][-1]


def edits_one(word: str) -> set[str]:
    alphabet = string.ascii_lowercase
    splits = [(word[:index], word[index:]) for index in range(len(word) + 1)]
    deletes = {left + right[1:] for left, right in splits if right}
    transposes = {left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1}
    replaces = {left + letter + right[1:] for left, right in splits if right for letter in alphabet}
    inserts = {left + letter + right for left, right in splits for letter in alphabet}
    return deletes | transposes | replaces | inserts


def edits_within_distance(word: str, vocabulary: set[str], max_distance: int = 2) -> set[str]:
    known = {word} & vocabulary
    if max_distance < 1:
        return known
    current = {word}
    for _ in range(max_distance):
        current = {candidate for item in current for candidate in edits_one(item)}
        known.update(current & vocabulary)
        if known:
            break
    return known
