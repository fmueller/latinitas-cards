"""Finite lexical readings from the retained Bennett edition, never deck glosses."""

from __future__ import annotations

from .claim_review import Evidence
from .latin_library import LibraryReading, attested_paradigms, bennett_sections
from .meaning_language import GERMAN, meaning_references
from .phrase_inputs import Features, Sense

VERSION = "bennett-lexical-constructions/1"


class BennettLexicon:
    """Constructed readings, not an exhaustive Latin parser or a translation engine."""

    version = VERSION
    ambiguity_complete = False

    def __init__(self) -> None:
        numbers = (20, 23, 63, 101, 170, 172, 174, 175)
        sections = dict(zip(numbers, bennett_sections(*numbers), strict=True))
        self.readings = [
            (cell.surface, LibraryReading(paradigm.lemma, paradigm.pos, cell.features))
            for paradigm in attested_paradigms()
            for cell in paradigm.cells
        ]
        cases = {
            "Nom.": "nominative",
            "Gen.": "genitive",
            "Dat.": "dative",
            "Acc.": "accusative",
            "Voc.": "vocative",
            "Abl.": "ablative",
        }
        for lemma, text in (
            ("hortus", sections[23].text.split("Nouns in -er and -ir", 1)[0]),
            ("puer", sections[23].text.split("Puer, boy;", 1)[1]),
        ):
            rows = text.split("SINGULAR.\n", 1)[1].split("\n1.", 1)[0].splitlines()
            number = "singular"
            for index, row in enumerate(rows):
                if row == "PLURAL.":
                    number = "plural"
                if row in cases:
                    self.readings.append(
                        (
                            rows[index + 1],
                            LibraryReading(
                                lemma,
                                "noun",
                                Features.model_validate({"case": cases[row], "number": number, "gender": "masculine"}),
                            ),
                        )
                    )
        self.lexemes: dict[str, tuple[Sense, ...]] = {}
        for lemma, section, heading in (
            ("porta", 20, "Porta, "),
            ("hortus", 23, "Hortus, "),
            ("puer", 23, "Puer, "),
            ("bonus", 63, "Bonus, "),
        ):
            gloss = sections[section].text.split(heading, 1)[1].split(";", 1)[0].split(".", 1)[0]
            self.lexemes[lemma] = (Sense(key=gloss, meaning=gloss, evidence=(sections[section],)),)
        if (
            "Amō, I\nlove." not in sections[101].text
            or "parentēs amāmus, we love our parents" not in sections[175].text
        ):
            raise ValueError("retained active love sense/valency locator missing")
        self.lexemes["amō"] = (
            Sense(
                key="love",
                meaning="love",
                valency={"subject": "nominative", "object": "accusative"},
                evidence=(sections[101], sections[170], sections[172], sections[174], sections[175]),
            ),
        )

    def analyze(self, text: str) -> tuple[LibraryReading, ...]:
        return tuple(dict.fromkeys(reading for surface, reading in self.readings if surface == text))

    def senses(self, lemma: str, language: str) -> tuple[Sense, ...]:
        senses = self.lexemes.get(lemma, ())
        if language == "en":
            return senses
        if language == "de":
            references = meaning_references()
            return tuple(
                s.model_copy(update={"meaning": GERMAN[s.key], "evidence": (*s.evidence, *references)})
                for s in senses
                if s.key in GERMAN
            )
        return ()


def construction_references(family: str) -> tuple[Evidence, ...]:
    sections = {
        "preposition-noun": (141, 142, 143),
        "noun-genitive": (195,),
        "adjective-noun": (234,),
        "verb-noun": (170, 172, 174, 175, 254),
    }
    return bennett_sections(*sections[family])
