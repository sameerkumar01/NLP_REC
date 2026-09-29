import re

SENTENCE_PATTERN = re.compile(r"[^.!?]+")
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def tokenize_sentence(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text))]


def text_to_sentences(text: str) -> list[list[str]]:
    sentences = []
    for piece in SENTENCE_PATTERN.findall(str(text)):
        tokens = tokenize_sentence(piece)
        if tokens:
            sentences.append(tokens)
    return sentences


def prepare_sentences(texts: list[str]) -> list[list[str]]:
    return [sentence for text in texts for sentence in text_to_sentences(text)]
