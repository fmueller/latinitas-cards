"""Read-only accepted-fact view over a corpus, its annotation sidecar and corpus claims.

v0.3.0 annotation validation is unchanged: ``latinitas-morph/1`` values are checked
only here, when a fact is consumed.
"""

import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from .annotation_sidecar import Annotation, Retraction, Segmentation, Translation, read_sidecar
from .annotation_state import annotation_states
from .cli_contract import Diagnostic
from .corpus_claims import current_annotations, join_corpus_decisions
from .corpus_io import CorpusFinding, read_corpus
from .corpus_model import CorpusPassage, CorpusRecord
from .corpus_validation import validate_records
from .terminology import FEATURE_VALUES

type FactReason = Literal[
    "missing_fact",
    "stale_fact",
    "open_fact",
    "withheld_fact",
    "conflicting_facts",
    "unsupported_morphology",
    "wrong_language",
    "no_features",
]
type FactLayer = Literal["morphology", "translation"]
type ReviewStatus = Literal["accepted", "open", "withheld"]
type Span = tuple[int, int]
type AnchorKey = tuple[FactLayer, Span | None]

MORPH_TAGSET = "latinitas-morph/1"
PARTS_OF_SPEECH = frozenset(
    {
        "noun",
        "verb",
        "adjective",
        "pronoun",
        "adverb",
        "preposition",
        "conjunction",
        "particle",
        "numeral",
        "interjection",
    }
)
# Non-finite forms are encoded only by ``verbform``, so ``mood`` takes the finite values alone.
MORPH_FEATURES: dict[str, frozenset[str]] = {name: frozenset(values) for name, values in FEATURE_VALUES.items()} | {
    "mood": frozenset({"indicativus", "coniunctivus", "imperativus"})
}
_REVIEW_REASONS: dict[ReviewStatus, FactReason | None] = {
    "accepted": None,
    "open": "open_fact",
    "withheld": "withheld_fact",
}


@dataclass(frozen=True)
class Morphology:
    lemma: str
    pos: str
    features: tuple[tuple[str, str], ...]


def parse_morphology(value: JsonValue) -> Morphology | None:
    """Return a strict ``latinitas-morph/1`` value, or None when it is unsupported."""
    if not isinstance(value, dict) or value.get("tagset") != MORPH_TAGSET:
        return None
    if not set(value) <= {"tagset", "lemma", "pos", "features"}:
        return None
    lemma, pos, features = value.get("lemma"), value.get("pos"), value.get("features", {})
    if not isinstance(lemma, str) or not lemma or not unicodedata.is_normalized("NFC", lemma):
        return None
    if not isinstance(pos, str) or pos not in PARTS_OF_SPEECH or not isinstance(features, dict):
        return None
    pairs = tuple(sorted((name, item) for name, item in features.items() if isinstance(item, str)))
    if len(pairs) != len(features) or any(item not in MORPH_FEATURES.get(name, ()) for name, item in pairs):
        return None
    return Morphology(lemma, pos, pairs)


@dataclass(frozen=True)
class Fact:
    """A current annotation on an anchor; ``reason`` is None only for an accepted, usable value."""

    annotation_id: str
    layer: FactLayer
    review_status: ReviewStatus
    value: Morphology | str | None
    reason: FactReason | None


@dataclass(frozen=True)
class Resolution:
    """The single accepted value serving one requirement, or the reason it is withheld."""

    reason: FactReason | None
    value: Morphology | str | None = None
    facts: tuple[Fact, ...] = ()


def resolve(facts: tuple[Fact, ...], *, stale: bool = False) -> Resolution:
    """Filter by language, then withhold on unsupported or disagreeing accepted values."""
    usable = [fact for fact in facts if fact.reason != "wrong_language"]
    accepted = [fact for fact in usable if fact.review_status == "accepted"]
    if any(fact.reason == "unsupported_morphology" for fact in accepted):
        return Resolution("unsupported_morphology", facts=facts)
    values = list(dict.fromkeys(fact.value for fact in accepted))
    if len(values) > 1:
        return Resolution("conflicting_facts", facts=facts)
    if values:
        return Resolution(None, values[0], facts)
    reasons = {fact.reason for fact in usable}
    for reason in ("open_fact", "withheld_fact", "unsupported_morphology"):
        if reason in reasons:
            return Resolution(reason, facts=facts)
    if facts:
        return Resolution("wrong_language", facts=facts)
    return Resolution("stale_fact" if stale else "missing_fact")


@dataclass(frozen=True)
class Sentence:
    index: int
    span: Span
    tokens: Span
    continued: bool
    continues: bool

    @property
    def fragment(self) -> bool:
        return self.continued or self.continues


@dataclass(frozen=True)
class PassageFacts:
    """One passage's current segmentation and its facts keyed by layer and exact anchor span."""

    passage: CorpusPassage
    segmentation_id: str | None
    segmentation: Segmentation | None
    sentences: tuple[Sentence, ...]
    facts: Mapping[AnchorKey, tuple[Fact, ...]]
    stale: frozenset[AnchorKey]

    def anchor(self, layer: FactLayer, span: Span | None) -> Resolution:
        return self._resolve((layer, span))

    def morphology(self, token: int) -> Resolution:
        assert self.segmentation is not None
        start, end = self.segmentation.tokens[token].span
        return self.anchor("morphology", (start, end))

    def analysis(self, token: int) -> Resolution:
        """Morphology for guided analysis, which needs at least one feature."""
        resolved = self.morphology(token)
        if isinstance(resolved.value, Morphology) and not resolved.value.features:
            return Resolution("no_features", facts=resolved.facts)
        return resolved

    def translation(self, span: Span) -> Resolution:
        return self.anchor("translation", span)

    def sentence_translation(self, index: int) -> Resolution:
        """A one-sentence passage's span-less translation also serves that sentence."""
        keys: list[AnchorKey] = [("translation", self.sentences[index].span)]
        if len(self.sentences) == 1:
            keys.append(("translation", None))
        return self._resolve(*keys)

    def passage_translation(self) -> Resolution:
        return self.anchor("translation", None)

    def _resolve(self, *keys: AnchorKey) -> Resolution:
        facts = tuple(fact for key in keys for fact in self.facts.get(key, ()))
        return resolve(facts, stale=any(key in self.stale for key in keys))


@dataclass(frozen=True)
class FactView:
    target_language: str
    passages: Mapping[str, PassageFacts]
    diagnostics: tuple[Diagnostic, ...]


def accepted_facts(corpus: Path, sidecar: Path, claims: Path, *, target_language: str) -> FactView:
    """Build the view without writes or locks; translations must match the primary subtag.

    An absent sidecar or claims file means no annotations or decisions.
    """
    diagnostics: list[Diagnostic] = []
    if not sidecar.exists():
        statuses: dict[str, str] = {}
    elif not claims.exists():
        statuses = dict.fromkeys((row.annotation_id for row in current_annotations(corpus, sidecar)), "open")
    else:
        statuses, diagnostics = join_corpus_decisions(corpus, sidecar, claims)
    passages: list[CorpusPassage] = []

    def corpus_rows() -> Iterator[CorpusRecord | CorpusFinding]:
        for row in read_corpus(corpus):
            if isinstance(row, CorpusPassage):
                passages.append(row)
            yield row

    # Validate this read too: its passage text and hashes are what the view serves.
    _, findings = validate_records(corpus_rows(), path=str(corpus))
    if findings:
        raise ValueError("; ".join(f"{d.code}: {d.message}" for d in findings))
    rows: list[Annotation | Retraction] = []
    if sidecar.exists():
        rows = [row for row in read_sidecar(sidecar) if isinstance(row, Annotation | Retraction)]
    states = annotation_states({passage.passage_id: passage.passage_hash for passage in passages}, rows)
    by_passage: dict[str, list[Annotation]] = {}
    for row in rows:
        if isinstance(row, Annotation):
            by_passage.setdefault(row.anchor.passage_id, []).append(row)
    language = _primary_subtag(target_language)
    # A decision counts only for a row live in this read, so a write between reads never surfaces as accepted.
    live = {
        annotation: statuses[annotation]
        for annotation, state in states.items()
        if annotation in statuses and state.state == "current" and not (state.superseded or state.retracted)
    }
    result: dict[str, PassageFacts] = {}
    for passage in passages:
        records = by_passage.get(passage.passage_id, [])
        segmentations = [row for row in records if row.layer == "segmentation" and row.annotation_id in live]
        # Without exactly one current segmentation record a passage has no units and no spanned facts.
        record = segmentations[0] if len(segmentations) == 1 else None
        segmentation = Segmentation.model_validate(record.value) if record is not None else None
        words = {
            (token.span[0], token.span[1])
            for token in (segmentation.tokens if segmentation else [])
            if token.kind == "word"
        }
        facts: dict[AnchorKey, list[Fact]] = {}
        stale: set[AnchorKey] = set()
        for row in records:
            if row.layer not in ("morphology", "translation"):
                continue
            span = (row.anchor.span[0], row.anchor.span[1]) if row.anchor.span is not None else None
            key: AnchorKey = ("morphology" if row.layer == "morphology" else "translation", span)
            status = live.get(row.annotation_id)
            if status is None:
                state = states[row.annotation_id]
                if state.state in ("stale", "orphaned") and not (state.superseded or state.retracted):
                    stale.add(key)
            elif span is not None and (record is None or row.anchor.segmentation_id != record.annotation_id):
                stale.add(key)
            else:
                facts.setdefault(key, []).append(_fact(row, key, _review_status(status), words, language))
        result[passage.passage_id] = PassageFacts(
            passage=passage,
            segmentation_id=record.annotation_id if record is not None else None,
            segmentation=segmentation,
            sentences=_sentences(segmentation),
            facts={key: tuple(items) for key, items in facts.items()},
            stale=frozenset(stale),
        )
    return FactView(target_language, result, tuple(diagnostics))


def _review_status(status: str) -> ReviewStatus:
    for review in _REVIEW_REASONS:
        if review == status:
            return review
    raise ValueError(f"unknown review status {status!r}")


def _fact(row: Annotation, key: AnchorKey, status: ReviewStatus, words: set[Span], language: str) -> Fact:
    layer, span = key
    value: Morphology | str | None
    reason: FactReason | None
    if layer == "morphology":
        value = parse_morphology(row.value)
        # The anchor must cover exactly one word token of the current segmentation.
        reason = "unsupported_morphology" if value is None or span not in words else None
    else:
        translation = Translation.model_validate(row.value)
        value = translation.text
        reason = "wrong_language" if _primary_subtag(translation.language) != language else None
    return Fact(row.annotation_id, layer, status, value, reason or _REVIEW_REASONS[status])


def _primary_subtag(language: str) -> str:
    return language.lower().split("-")[0]


def _sentences(segmentation: Segmentation | None) -> tuple[Sentence, ...]:
    return tuple(
        Sentence(
            index,
            (sentence.span[0], sentence.span[1]),
            (sentence.tokens[0], sentence.tokens[1]),
            sentence.continued,
            sentence.continues,
        )
        for index, sentence in enumerate(segmentation.sentences if segmentation else [])
    )
