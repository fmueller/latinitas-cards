"""Finite inflection proposals from bound source anchors, not acceptance authority.

The consumer must corroborate anchor identity, paradigm applicability and every
rendered assertion. Endings alone never establish a source identity or PPP role.
"""

from dataclasses import dataclass

from .claim_review import Evidence
from .phrase_inputs import Features

RULE_VERSION = "bounded-inflection/1"
REFERENCE = "https://dcc.dickinson.edu/grammar/latin/"


def form_translations(pos: str, features: Features, meaning: str, language: str, role: str) -> tuple[str, ...]:
    """Finite reference glosses; unknown inflected meanings need explicit evidence.

    These are proposals covered by the same complete translation assertion as an
    authored reference. They never establish semantic acceptance or grading rules.
    """
    if role == "ppp":
        choices = {
            ("en", "love"): ("loved", "having been loved"),
            ("en", "loved"): ("loved", "having been loved"),
            ("de", "lieben"): ("geliebt",),
            ("de", "geliebt"): ("geliebt",),
        }
        if (language, meaning) in choices:
            return choices[language, meaning]
    elif pos == "adjective" or (pos == "noun" and (features.case, features.number) == ("nominative", "singular")):
        return (meaning,)
    elif pos == "verb":
        if role == "present_infinitive":
            choices = {("en", "love"): ("to love",), ("de", "lieben"): ("lieben",)}
            if (language, meaning) in choices:
                return choices[language, meaning]
        if (features.tense, features.mood, features.voice) == ("present", "indicative", "active"):
            cells = {
                ("1", "singular"): ("I love", "ich liebe"),
                ("2", "singular"): ("you love", "du liebst"),
                ("3", "singular"): ("he/she/it loves", "er/sie/es liebt"),
                ("1", "plural"): ("we love", "wir lieben"),
                ("2", "plural"): ("you (plural) love", "ihr liebt"),
                ("3", "plural"): ("they love", "sie lieben"),
            }
            if (language, meaning) in (("en", "love"), ("de", "lieben")) and (
                features.person,
                features.number,
            ) in cells:
                return (cells[str(features.person), str(features.number)][0 if language == "en" else 1],)
    elif pos == "noun" and (language, meaning) in (("en", "girl"), ("de", "Mädchen")):
        singular = features.number == "singular"
        noun_cells = {
            "nominative": ("girl" if singular else "girls", "das Mädchen" if singular else "die Mädchen"),
            "accusative": ("a girl" if singular else "girls", "das Mädchen" if singular else "die Mädchen"),
            "genitive": ("of a girl" if singular else "of girls", "des Mädchens" if singular else "der Mädchen"),
            "dative": ("to/for a girl" if singular else "to/for girls", "dem Mädchen" if singular else "den Mädchen"),
            "ablative": (
                "by/with/from a girl" if singular else "by/with/from girls",
                "mit/von einem Mädchen" if singular else "mit/von Mädchen",
            ),
            "vocative": ("girl!" if singular else "girls!", "Mädchen!"),
        }
        if features.case in noun_cells and features.number in ("singular", "plural"):
            return (noun_cells[str(features.case)][0 if language == "en" else 1],)
    raise ValueError("missing_form_translation_evidence; supply a supported explicit form reference")


VERB_ENDINGS = {
    ("1", "singular"): "ō",
    ("2", "singular"): "ās",
    ("3", "singular"): "at",
    ("1", "plural"): "āmus",
    ("2", "plural"): "ātis",
    ("3", "plural"): "ant",
}
NOUN_ENDINGS = {
    ("nominative", "singular"): "a",
    ("genitive", "singular"): "ae",
    ("dative", "singular"): "ae",
    ("accusative", "singular"): "am",
    ("ablative", "singular"): "ā",
    ("vocative", "singular"): "a",
    ("nominative", "plural"): "ae",
    ("genitive", "plural"): "ārum",
    ("dative", "plural"): "īs",
    ("accusative", "plural"): "ās",
    ("ablative", "plural"): "īs",
    ("vocative", "plural"): "ae",
}
SECOND_ENDINGS = {
    ("nominative", "singular"): "us",
    ("genitive", "singular"): "ī",
    ("dative", "singular"): "ō",
    ("accusative", "singular"): "um",
    ("ablative", "singular"): "ō",
    ("vocative", "singular"): "e",
    ("nominative", "plural"): "ī",
    ("genitive", "plural"): "ōrum",
    ("dative", "plural"): "īs",
    ("accusative", "plural"): "ōs",
    ("ablative", "plural"): "īs",
    ("vocative", "plural"): "ī",
}


@dataclass(frozen=True)
class DerivedForm:
    surface: str
    stem: str
    rule: str
    evidence: Evidence


def derive_form(pos: str, anchors: tuple[str, ...], target: Features) -> DerivedForm:
    """Propose one explicitly requested regular cell, never expand a paradigm.

    Applicability is a separate assertion at the full-card gate. These shape
    checks reject incompatible anchors even before that corroboration is read.
    """
    required = ("person", "number", "tense", "mood", "voice") if pos == "verb" else ("case", "number", "gender")
    if any(getattr(target, name) is None for name in required):
        raise ValueError("incomplete production target")
    if any(value is not None for name, value in target.model_dump().items() if name not in required):
        raise ValueError("unsupported cross-paradigm target")
    if pos == "verb":
        if len(anchors) != 2 or not anchors[1].endswith("āre") or anchors[0] != anchors[1][:-3] + "ō":
            raise ValueError("unsupported first-conjugation anchors")
        if (target.tense, target.mood, target.voice) != ("present", "indicative", "active"):
            raise ValueError("unsupported verb cell")
        base = anchors[1][:-3]
        ending = VERB_ENDINGS.get((str(target.person), str(target.number)))
        stem, rule, reference = base + "ā", "regular-first-present-active/1", "1st-conjugation"
    elif pos == "noun":
        if len(anchors) != 2 or not anchors[0].endswith("a") or anchors[1] != anchors[0][:-1] + "ae":
            raise ValueError("unsupported first-declension nominative/genitive anchors")
        if target.gender not in ("masculine", "feminine"):
            raise ValueError("unsupported noun gender")
        base = anchors[0][:-1]
        ending = NOUN_ENDINGS.get((str(target.case), str(target.number)))
        stem, rule, reference = base + "ā", "regular-first-declension/1", "1st-declension-stem-paradigm-and-gender"
    elif pos == "adjective":
        if len(anchors) != 3 or not anchors[0].endswith("us"):
            raise ValueError("unsupported adjective anchors")
        base = anchors[0][:-2]
        if anchors != (base + "us", base + "a", base + "um"):
            raise ValueError("unsupported adjective gender anchors")
        # The pronominal -īus class is not the regular bonus paradigm.
        if anchors[0] in ("alius", "alter", "ūnus", "ūllus", "nūllus", "sōlus", "tōtus", "neuter", "uter"):
            raise ValueError("unsupported pronominal adjective")
        endings = NOUN_ENDINGS if target.gender == "feminine" else SECOND_ENDINGS
        ending = endings.get((str(target.case), str(target.number)))
        if target.gender == "neuter" and target.case in ("nominative", "accusative", "vocative"):
            ending = "um" if target.number == "singular" else "a"
        stem, rule, reference = base, "regular-us-a-um/1", "1st-and-2nd-declension-adjectives-%C4%81-o-stems"
    else:
        raise ValueError("unsupported paradigm")
    if ending is None:
        raise ValueError("unsupported cell")
    return DerivedForm(
        base + ending,
        stem,
        rule,
        Evidence(
            "Allen and Greenough; DCC, Meagan Ayer",
            REFERENCE + reference,
            "2014",
            f"{rule}; {anchors}; {target.model_dump(exclude_none=True)}",
        ),
    )
