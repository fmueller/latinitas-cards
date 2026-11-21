"""Newly composed development wording for finite Bennett-grounded propositions.

These are whole utterances, not keywords. Each tuple denotes one proposition;
unlisted wording has no interpretation here. No naturalness/exhaustivity claim
or general Latin/German/English language evaluator follows from this vocabulary.
See docs/meaning-reference-review.md for the initial claims and review record.
"""

from .claim_review import Evidence
from .latin_library import bennett_sections

VERSION = "bennett-bilingual-meaning/2"

GERMAN = {"gate": "Tor", "garden": "Garten", "boy": "Junge", "good": "gut", "love": "lieben"}

# The isolated finite predicate has no supplied object. Experiential paraphrases
# were rejected by independent review; see the retained initial claims.
LOVE: dict[str, tuple[tuple[str, ...], ...]] = {
    "en": (
        ("I love",),
        ("you love",),
        ("he/she/it loves",),
        ("we love",),
        ("you (plural) love",),
        ("they love",),
    ),
    "de": (
        ("ich liebe",),
        ("du liebst",),
        ("er/sie/es liebt",),
        ("wir lieben",),
        ("ihr liebt",),
        ("sie lieben",),
    ),
}

# Fixed constructed phrases; generic genitive relation does not select possession
# over other possible relations. Surface/tuple applicability is checked separately.
PHRASES: dict[str, dict[str, tuple[str, ...]]] = {
    "preposition-noun": {
        "en": ("with a gate", "together with a gate"),
        "de": ("mit einem Tor", "zusammen mit einem Tor"),
    },
    "noun-genitive": {
        "en": ("the gate of the garden", "the garden's gate"),
        "de": ("das Tor des Gartens", "des Gartens Tor"),
    },
    "adjective-noun": {
        "en": ("a good garden", "a garden that is good"),
        "de": ("ein guter Garten", "ein Garten, der gut ist"),
    },
    "verb-noun": {
        "en": ("the boy loves the gate", "the gate is loved by the boy"),
        "de": ("der Junge liebt das Tor", "das Tor wird vom Jungen geliebt"),
    },
}

CONTEXTS: dict[str, dict[str, str]] = {
    "preposition-noun": {
        "en": (
            "Suppose someone has set out (profectus est …). "
            "Translate only the fragment, using indefinite noun reference."
        ),
        "de": (
            "Nimm an, jemand ist aufgebrochen (profectus est …). "
            "Übersetze nur das Fragment und verwende eine unbestimmte Nominalgruppe."
        ),
    },
    "noun-genitive": {
        "en": "Use definite reference for both nouns.",
        "de": "Verwende für beide Substantive eine bestimmte Referenz.",
    },
    "adjective-noun": {
        "en": "Use indefinite reference for the noun.",
        "de": "Verwende für das Substantiv eine unbestimmte Referenz.",
    },
    "verb-noun": {
        "en": "Use definite reference for both nouns.",
        "de": "Verwende für beide Substantive eine bestimmte Referenz.",
    },
}

# Interpreted negative candidates remain excluded from the equivalence vocabulary.
COMPATIBLE_LOVE: dict[str, tuple[str, ...]] = {
    "en": (
        "I feel love",
        "you feel love",
        "he/she/it feels love",
        "we feel love",
        "you (plural) feel love",
        "they feel love",
    ),
    "de": (
        "ich empfinde Liebe",
        "du empfindest Liebe",
        "er/sie/es empfindet Liebe",
        "wir empfinden Liebe",
        "ihr empfindet Liebe",
        "sie empfinden Liebe",
    ),
}
ROLE_CONFLICT: dict[str, tuple[str, ...]] = {
    "en": ("the gate loves the boy",),
    "de": ("das Tor liebt den Jungen",),
}

CLAUSES: dict[str, dict[str, tuple[str, ...]]] = {
    "present-active": {
        "en": ("The verb is present indicative active.", "This is an active present indicative form."),
        "de": ("Das Verb steht im Präsens Indikativ Aktiv.", "Dies ist eine aktive Präsensform im Indikativ."),
    },
    "bounded": {
        "en": ("No claim of exhaustive contextual interpretation is made.",),
        "de": ("Eine vollständige kontextuelle Deutung wird nicht behauptet.",),
    },
    "preposition-noun": {
        "en": ("Cum governs the ablative and expresses accompaniment here.",),
        "de": ("Cum regiert den Ablativ und drückt hier Begleitung aus.",),
    },
    "noun-genitive": {
        "en": ("The genitive relates the gate to the garden without specifying the kind of relation.",),
        "de": ("Der Genitiv setzt das Tor zum Garten in Beziehung, ohne die Art der Beziehung festzulegen.",),
    },
    "adjective-noun": {
        "en": ("Bonus agrees with hortus in case, number and gender.",),
        "de": ("Bonus stimmt mit hortus in Kasus, Numerus und Genus überein.",),
    },
    "verb-noun": {
        "en": ("Puer is the subject and portam is the object of amat.",),
        "de": ("Puer ist das Subjekt und portam das Objekt von amat.",),
    },
}

ROLES: dict[str, dict[str, dict[str, str]]] = {
    "preposition-noun": {"object": {"en": "accompaniment: gate", "de": "Begleitung: Tor"}},
    "noun-genitive": {
        "head": {"en": "head: gate", "de": "Bezugswort: Tor"},
        "dependent": {"en": "genitive relation: garden", "de": "Genitivbeziehung: Garten"},
    },
    "adjective-noun": {
        "head": {"en": "head: garden", "de": "Bezugswort: Garten"},
        "modifier": {"en": "quality: good", "de": "Eigenschaft: gut"},
    },
    "verb-noun": {
        "subject": {"en": "lover: boy", "de": "Liebender: Junge"},
        "object": {"en": "loved entity: gate", "de": "geliebtes Objekt: Tor"},
        "verb": {"en": "present active love", "de": "lieben im Präsens Aktiv"},
    },
}


def meaning_references() -> tuple[Evidence, ...]:
    return (
        *bennett_sections(20, 23, 63, 101, 141, 142, 143, 170, 172, 174, 175, 195, 222, 234, 254),
        Evidence(
            "newly-composed-development-language",
            "docs/meaning-reference-review.md",
            VERSION,
            "Finite truth-conditional propositions and whole utterances; no commercial translation, "
            "human curated-card approval, naturalness rating or general evaluator authority.",
        ),
    )
