"""Visible text-cleaning steps used by the from-scratch pipeline."""
import html
import re

HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_TEXT = re.compile(r"[^a-zA-Z0-9'!?.,;:\s-]")
WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Return readable text while preserving useful word boundaries.

    The order matters: decode HTML entities before removing tags, then remove
    URLs and unusual symbols, and finally collapse repeated whitespace.
    """
    text = html.unescape(str(text))
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" ", text)
    text = NON_TEXT.sub(" ", text)
    text = WHITESPACE.sub(" ", text)
    return text.strip()
