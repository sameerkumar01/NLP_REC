"""Small rule-based lemmatizer for learning morphology decisions."""

IRREGULAR = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be",
    "has": "have", "had": "have", "does": "do", "did": "do",
    "went": "go", "gone": "go", "better": "good", "best": "good",
    "mice": "mouse", "children": "child", "men": "man", "women": "woman",
}


def lemmatize(token: str) -> str:
    """Return a conservative lemma using ordered suffix rules.

    Because this has no POS tagger or dictionary, it will not handle every
    inflection. That limitation is intentional and documented for comparison.
    """
    if token in IRREGULAR:
        return IRREGULAR[token]
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ves") and len(token) > 4:
        return token[:-3] + "f"
    if token.endswith("ing") and len(token) > 5:
        root = token[:-3]
        if root.endswith(root[-1] * 2):
            root = root[:-1]
        return root
    if token.endswith("ed") and len(token) > 4:
        root = token[:-2]
        if root.endswith(root[-1] * 2):
            root = root[:-1]
        return root
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token
