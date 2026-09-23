"""Map CLTK 2 word annotations to the flat string columns written by ``annotate``."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WordAnalysis:
    lemma: str
    upos: str
    xpos: str
    morph_features: str


def format_ud_features(features: Any) -> str:
    """Render a CLTK ``UDFeatureTagSet`` as a UD FEATS string, e.g. ``Case=Nom|Number=Sing``.

    Keys are sorted and repeated keys are merged into sorted, comma-separated values, following
    the CoNLL-U FEATS convention.
    """
    values_by_key: defaultdict[str, set[str]] = defaultdict(set)
    for tag in getattr(features, "features", None) or []:
        values_by_key[str(tag.key)].add(str(tag.value))
    return "|".join(f"{key}={','.join(sorted(values))}" for key, values in sorted(values_by_key.items()))


def analyze_cltk_word(word: Any) -> WordAnalysis:
    """Extract lemma, UPOS, XPOS, and UD features from a CLTK 2 ``Word``.

    CLTK 2 does not carry Stanza's XPOS through its Stanza backend, so ``xpos`` is usually empty.
    """
    upos = getattr(word, "upos", None)
    return WordAnalysis(
        lemma=str(getattr(word, "lemma", None) or ""),
        upos=str(getattr(upos, "tag", None) or ""),
        xpos=str(getattr(word, "xpos", None) or ""),
        morph_features=format_ud_features(getattr(word, "features", None)),
    )
