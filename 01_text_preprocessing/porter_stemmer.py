"""Educational Porter stemmer implementation.

The implementation follows the classic suffix families in steps 1a through 5.
It is intentionally written as small predicates instead of using NLTK's
PorterStemmer, so each rule can be inspected and modified.
"""

VOWELS = set("aeiou")


class PorterStemmer:
    """Apply the main English Porter suffix-reduction rules."""

    @staticmethod
    def _contains_vowel(word: str) -> bool:
        return any(char in VOWELS for char in word)

    @staticmethod
    def _measure(word: str) -> int:
        """Approximate Porter measure m: the number of VC transitions."""
        measure = 0
        previous_was_vowel = False
        for char in word:
            current_is_vowel = char in VOWELS
            if previous_was_vowel and not current_is_vowel:
                measure += 1
            previous_was_vowel = current_is_vowel
        return measure

    @staticmethod
    def _ends_double_consonant(word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and word[-1] not in VOWELS

    @staticmethod
    def _cvc(word: str) -> bool:
        """Whether the end looks consonant-vowel-consonant (not w/x/y)."""
        if len(word) < 3:
            return False
        a, b, c = word[-3:]
        return a not in VOWELS and b in VOWELS and c not in VOWELS and c not in "wxy"

    def stem(self, word: str) -> str:
        w = word.lower()
        if len(w) <= 2 or not w.isalpha():
            return w
        w = self._step_1a(w)
        w = self._step_1b(w)
        w = self._step_1c(w)
        w = self._replace_suffixes(w, {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
            "izer": "ize", "bli": "ble", "alli": "al", "entli": "ent", "eli": "e",
            "ousli": "ous", "ization": "ize", "ation": "ate", "ator": "ate",
            "alism": "al", "iveness": "ive", "fulness": "ful", "ousness": "ous",
            "aliti": "al", "iviti": "ive", "biliti": "ble", "logi": "log",
        }, minimum_measure=0)
        w = self._replace_suffixes(w, {
            "icate": "ic", "ative": "", "alize": "al", "iciti": "ic",
            "ical": "ic", "ful": "", "ness": "",
        }, minimum_measure=0)
        for suffix in ("ement", "ment", "able", "ible", "ant", "ent", "al", "er", "ic", "ive", "ous", "ize", "ion"):
            if w.endswith(suffix):
                stem = w[:-len(suffix)]
                if self._measure(stem) > 1 and (suffix != "ion" or (stem.endswith("s") or stem.endswith("t"))):
                    w = stem
                break
        if w.endswith("e"):
            stem = w[:-1]
            if self._measure(stem) > 1 or (self._measure(stem) == 1 and not self._cvc(stem)):
                w = stem
        if w.endswith("ll") and self._measure(w) > 1:
            w = w[:-1]
        return w

    def _step_1a(self, word: str) -> str:
        if word.endswith("sses"):
            return word[:-2]       # sses -> ss
        if word.endswith("ies"):
            return word[:-2]        # ies -> i
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step_1b(self, word: str) -> str:
        if word.endswith("eed"):
            stem = word[:-3]
            return stem + "ee" if self._measure(stem) > 0 else word
        for suffix in ("ed", "ing"):
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._contains_vowel(stem):
                    word = stem
                    if word.endswith(("at", "bl", "iz")):
                        return word + "e"
                    if self._ends_double_consonant(word) and word[-1] not in "lsz":
                        return word[:-1]
                    if self._measure(word) == 1 and self._cvc(word):
                        return word + "e"
                return word
        return word

    def _step_1c(self, word: str) -> str:
        if word.endswith("y") and self._contains_vowel(word[:-1]):
            return word[:-1] + "i"
        return word

    def _replace_suffixes(self, word: str, replacements: dict[str, str], minimum_measure: int) -> str:
        for suffix, replacement in replacements.items():
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > minimum_measure:
                    return stem + replacement
                return word
        return word
