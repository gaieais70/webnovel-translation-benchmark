"""Deterministic diagnostics computed from output text alone.

No judge is involved, so these cost nothing and are perfectly reproducible.
They are the cheapest defense against a judge that rewards fluent-sounding
filler: a model can score well on the rubric and still read like machine
translation, and these two numbers show it.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# Digits are part of a word: web serials are full of numerals ("3,000 years",
# "Level 7"). Dropping them also undercounts the denominator in slop_score's
# per-1,000-words normalization, which would inflate every slop rate.
_WORD_RE = re.compile(r"[a-z0-9']+")
_TOP_N = 20


def load_slop_list(path: Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [
        line.strip().lower()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def slop_score(text: str, slop_list: list[str]) -> float:
    """Slop-phrase occurrences per 1,000 words."""
    words = _words(text)
    if not words:
        return 0.0
    lowered = text.lower()
    hits = sum(lowered.count(phrase) for phrase in slop_list)
    return hits * 1000.0 / len(words)


def repetition_score(text: str) -> float:
    """Share of output accounted for by its most frequent n-grams.

    Averages the top-20 concentration of unigrams, bigrams, and trigrams.
    Higher means more repetitive. Bounded to 0.0-1.0.
    """
    words = _words(text)
    if not words:
        return 0.0

    shares: list[float] = []
    for n in (1, 2, 3):
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if not grams:
            continue
        top = Counter(grams).most_common(_TOP_N)
        shares.append(sum(count for _, count in top) / len(grams))
    if not shares:
        return 0.0
    return min(1.0, sum(shares) / len(shares))
