"""Opt-in local Latin analysis. Construction never downloads models or calls a service."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from .acceptance_records import digest
from .phrase_inputs import POS, Features, Sense


class ProcessingUnavailable(RuntimeError):
    """Processing prerequisites failed; this is not a linguistic judgment."""


@dataclass(frozen=True)
class LibraryReading:
    lemma: str
    pos: POS
    features: Features


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
