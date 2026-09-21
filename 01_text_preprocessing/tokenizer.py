"""A small regex tokenizer written for inspection, not linguistic completeness."""
import re

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+", re.IGNORECASE)


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Extract words, contractions, and integer-like numbers.

    A regex makes the token boundary rule explicit. It intentionally does not
    pretend to solve every language, emoji, or punctuation-tokenization case.
    """
    if lowercase:
        text = text.lower()
    return TOKEN_PATTERN.findall(text)
