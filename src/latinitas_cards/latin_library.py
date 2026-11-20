"""Opt-in local Latin analysis. Construction never downloads models or calls a service."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from .acceptance_records import digest
from .claim_review import Evidence
from .html_text import source_html_to_text
from .phrase_inputs import POS, Features, Sense

BENNETT_PATH = Path(__file__).parent / "references" / "bennett-1918.html"
BENNETT_SHA256 = "13f80dd1202e958854117c9c9cb2914630e92e37de2967791a45237f571327e6"


class ProcessingUnavailable(RuntimeError):
    """Processing prerequisites failed; this is not a linguistic judgment."""


@dataclass(frozen=True)
class LibraryReading:
    lemma: str
    pos: POS
    features: Features


@dataclass(frozen=True)
class AttestedCell:
    surface: str
    features: Features


@dataclass(frozen=True)
class AttestedParadigm:
    """One explicitly attested lexeme, not a suffix-based productive class."""

    pos: POS
    lemma: str
    anchors: tuple[AttestedCell, ...]
    cells: tuple[AttestedCell, ...]
    rule: str
    evidence: Evidence


def bennett_sections(*sections: int) -> tuple[Evidence, ...]:
    """Read exact sections from the retained edition, refusing altered bytes."""
    try:
        raw = BENNETT_PATH.read_bytes()
    except OSError as exc:
        raise ProcessingUnavailable("attested morphology reference absent or unreadable") from exc
    if hashlib.sha256(raw).hexdigest() != BENNETT_SHA256:
        raise ProcessingUnavailable("attested morphology reference digest mismatch")
    document = raw.decode("latin1")
    evidence = []
    for section in sections:
        start = document.index(f'<a name="sect{section}">')
        end = document.index(f'<a name="sect{section + 1}">', start)
        evidence.append(
            Evidence(
                "Charles E. Bennett, New Latin Grammar",
                f"https://www.gutenberg.org/ebooks/15665#sect{section}",
                "1918; GITenberg HTML; sha256:" + BENNETT_SHA256,
                source_html_to_text(document[start:end]),
            )
        )
    return tuple(evidence)


def attested_paradigms() -> tuple[AttestedParadigm, ...]:
    """Read the installed, digest-pinned source tables without using the generator.

    This is bounded generation evidence, not exhaustive analysis of an arbitrary
    surface or contextual disambiguation. No sense or translation is certified.
    """
    cases = {
        "Nom.": "nominative",
        "Gen.": "genitive",
        "Dat.": "dative",
        "Acc.": "accusative",
        "Voc.": "vocative",
        "Abl.": "ablative",
    }
    paradigms = []
    tables: tuple[tuple[int, POS, str], ...] = (
        (20, "noun", "regular-first-declension/1"),
        (63, "adjective", "regular-us-a-um/1"),
        (101, "verb", "regular-first-present-active/1"),
    )
    for (_, pos, rule), evidence in zip(tables, bennett_sections(*(t[0] for t in tables)), strict=True):
        text = evidence.text
        cells = []
        anchors: tuple[AttestedCell, ...]
        if pos == "verb":
            anchor_texts = text.split("PERF. PASS. PARTIC.\n", 1)[1].splitlines()[:2]
            rows = text.split("PRESENT TENSE.\nSINGULAR.\nPLURAL.\n", 1)[1].split("\nIMPERFECT.", 1)[0].splitlines()
            for index, row in enumerate(rows):
                cells.append(
                    AttestedCell(
                        row.split(",", 1)[0],
                        Features.model_validate(
                            {
                                "person": str(index // 2 + 1),
                                "number": "singular" if index % 2 == 0 else "plural",
                                "tense": "present",
                                "mood": "indicative",
                                "voice": "active",
                            }
                        ),
                    )
                )
            anchors = (
                AttestedCell(anchor_texts[0], cells[0].features),
                AttestedCell(anchor_texts[1], Features(tense="present", voice="active")),
            )
        else:
            rows = text.split("SINGULAR.\n", 1)[1].split("\n1.", 1)[0].splitlines()
            number = "singular"
            genders = ("feminine",) if pos == "noun" else ("masculine", "feminine", "neuter")
            for index, row in enumerate(rows):
                if row == "PLURAL.":
                    number = "plural"
                if row not in cases:
                    continue
                for column, gender in enumerate(genders, start=1):
                    cells.append(
                        AttestedCell(
                            rows[index + column],
                            Features.model_validate({"case": cases[row], "number": number, "gender": gender}),
                        )
                    )
            anchor_count = 2 if pos == "noun" else 3
            anchors = tuple(cells[:anchor_count])
        paradigms.append(
            AttestedParadigm(
                pos=pos,
                lemma=anchors[0].surface,
                anchors=anchors,
                cells=tuple(cells),
                rule=rule,
                evidence=evidence,
            )
        )
    return tuple(paradigms)


class LatinLibrary(Protocol):
    @property
    def version(self) -> str: ...

    @property
    def ambiguity_complete(self) -> bool: ...

    def analyze(self, text: str) -> tuple[LibraryReading, ...]: ...

    def senses(self, lemma: str, language: str) -> tuple[Sense, ...]: ...


class LocalStanza:
    """Use already-installed Latin models, with Stanza downloads explicitly disabled.

    A contextual tagger supplies one hypothesis, not an exhaustive ambiguity set.
    The evidence adapter consequently never treats this backend alone as a complete
    disambiguation proof.
    """

    ambiguity_complete = False

    def senses(self, lemma: str, language: str) -> tuple[Sense, ...]:
        """Stanza supplies no lexical meaning/valency reference."""
        return ()

    def __init__(self, resources: Path) -> None:
        self._pipeline: Any = None
        self._resources = resources
        self._setup_error: str | None = None
        try:
            package_version = version("stanza")
        except PackageNotFoundError:
            package_version = "absent"
        try:
            self._resource_digest = self._fingerprint()
        except OSError as exc:
            self._setup_error = str(exc)
            self._resource_digest = "unavailable"
        self.version = f"stanza-local/1:{package_version}:{self._resource_digest}"

    def _fingerprint(self) -> str:
        files = [self._resources / "resources.json", *sorted((self._resources / "la").rglob("*.pt"))]
        return digest(
            [
                {
                    "path": p.relative_to(self._resources).as_posix(),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in files
                if p.is_file()
            ]
        )

    def analyze(self, text: str) -> tuple[LibraryReading, ...]:
        try:
            if self._setup_error is not None:
                raise ProcessingUnavailable(self._setup_error)
            if self._fingerprint() != self._resource_digest:
                raise ProcessingUnavailable("local resources changed since version capture")
            if not (self._resources / "resources.json").is_file() or not (self._resources / "la").is_dir():
                raise ProcessingUnavailable("local Latin Stanza resources absent")
            if self._pipeline is None:
                import stanza

                self._pipeline = stanza.Pipeline(
                    lang="la",
                    dir=str(self._resources),
                    processors="tokenize,pos,lemma",
                    download_method=None,
                    use_gpu=False,
                    tokenize_no_ssplit=True,
                )
            doc = self._pipeline(text)
            if self._fingerprint() != self._resource_digest:
                raise ProcessingUnavailable("local resources changed during processing")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise ProcessingUnavailable(str(exc)) from exc
        words = [w for s in doc.sentences for w in s.words]
        if len(words) != 1 or words[0].text != text:
            return ()
        word = words[0]
        pos: dict[str, POS] = {"NOUN": "noun", "ADJ": "adjective", "VERB": "verb", "AUX": "verb"}
        if word.upos not in pos:
            return ()
        ud = dict(part.split("=", 1) for part in (word.feats or "").split("|") if "=" in part)
        names = {
            "Case": "case",
            "Number": "number",
            "Gender": "gender",
            "Person": "person",
            "Tense": "tense",
            "Mood": "mood",
            "Voice": "voice",
        }
        values = {
            "Nom": "nominative",
            "Acc": "accusative",
            "Gen": "genitive",
            "Dat": "dative",
            "Abl": "ablative",
            "Voc": "vocative",
            "Loc": "locative",
            "Sing": "singular",
            "Plur": "plural",
            "Masc": "masculine",
            "Fem": "feminine",
            "Neut": "neuter",
            "Pres": "present",
            "Perf": "perfect",
            "Imp": "imperfect",
            "Fut": "future",
            "Pqp": "pluperfect",
            "Ind": "indicative",
            "Sub": "subjunctive",
            "Impv": "imperative",
            "Act": "active",
            "Pass": "passive",
            "1": "1",
            "2": "2",
            "3": "3",
        }
        features = Features.model_validate({names[k]: values[v] for k, v in ud.items() if k in names and v in values})
        return (LibraryReading(word.lemma or "", pos[word.upos], features),)
