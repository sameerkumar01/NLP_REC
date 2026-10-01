import re

TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text))]
