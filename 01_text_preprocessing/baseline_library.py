"""NLTK baseline for comparison; library calls live only in this file."""
from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_data():
    """Download the small comparison corpora if they are not already present."""
    resources = [("corpora/stopwords", "stopwords"), ("tokenizers/punkt", "punkt"), ("corpora/wordnet", "wordnet")]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


def preprocess_with_nltk(text: str) -> dict[str, list[str] | str]:
    ensure_nltk_data()
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    tokens = [token.lower() for token in word_tokenize(cleaned, preserve_line=True) if token.isalpha()]
    filtered = [token for token in tokens if token not in set(stopwords.words("english"))]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    return {
        "cleaned": cleaned,
        "raw_tokens": tokens,
        "filtered_tokens": filtered,
        "stems": [stemmer.stem(token) for token in filtered],
        "lemmas": [lemmatizer.lemmatize(token) for token in filtered],
    }
