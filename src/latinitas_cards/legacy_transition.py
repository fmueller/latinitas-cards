"""Explicit, review-first guards for legacy note-model transitions.

The pre-release per-exercise note models — note schema ``1``/``2``, the
``latinitas-v1-`` identities, and the legacy split-path clones — cannot be
consolidated into the current learning-object model by ordinary CSV Update
with any promise of preserved card histories.  This module never mutates a
collection: it classifies observed destination evidence and returns explicit,
read-only decisions.  The only supported legacy option is an explicitly
approved fresh start of disposable data into a new dedicated note type;
valuable review history and conflicting personal annotations retain the old
collection, incompatible targets are rejected outright, and old per-exercise
identities are never reinterpreted as new object identities.  Compatible
regeneration of the current model is a separate decision that preserves note
identity, surviving cards, and review history.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .identity import LATINITAS_ID_VERSION
from .notes import NOTE_SCHEMA_VERSION
from .profile import DeckProfile

LEGACY_NOTE_SCHEMA_VERSIONS: tuple[str, ...] = ("1", "2")
LEGACY_LATINITAS_ID_VERSION = "v1"
LEGACY_SPLIT_GUID_PREFIX = "legacy-split-guid-v1-"

_LEGACY_ID_PREFIX = f"latinitas-{LEGACY_LATINITAS_ID_VERSION}-"
_CURRENT_ID_PREFIX = f"latinitas-{LATINITAS_ID_VERSION}-"

#: The published guarantee boundary for any structural consolidation of the
#: legacy models.  It is quoted verbatim in ``docs/legacy-transition.md`` and
#: a test fails when the published copy drifts.
CSV_CONSOLIDATION_HISTORY_STATEMENT = (
    "Ordinary CSV structural consolidation has no demonstrated history guarantee: "
    "a future history-preserving migration requires separate approval, destination-aware "
    "per-card mapping, explicit annotation and tag reconciliation, backup and rollback, "
    "and verified native scheduling and history preservation evidence."
)

_REGENERATION_REASON = (
    "Current learning-object model: regeneration re-exports the unchanged LatinitasID, "
    "so the first-field match updates managed fields in place and preserves note identity, "
    "surviving cards, and review history; eligibility loss is withheld by the "
    "prior-export checkpoint instead of deleting cards."
)
_UNRECOGNIZED_REASON = (
    "The destination note model cannot be recognized safely. Inventory the note type, "
    "schema, identities, personal fields, tags, and histories before any transition."
)
_HISTORY_REASON = (
    "Legacy note with valuable review history: retain the old collection and defer "
    "conversion. " + CSV_CONSOLIDATION_HISTORY_STATEMENT
)
_PERSONAL_REASON = (
    "Legacy note carries a personal annotation: retain the old collection and defer "
    "conversion; personal annotations are never combined by last-write-wins."
)
_FRESH_START_REASON = (
    "Explicitly approved fresh start: export the disposable data into the new dedicated "
    "note type with new identities and new schedules; the old collection stays intact "
    "and is never cleaned up automatically."
)
_RECONFIRMATION_REASON = (
    "Legacy pre-release note model: old per-exercise and split-clone identities are never "
    "reinterpreted as new object identities; an explicitly approved fresh start is the "
    "only supported option."
)

_BACKUP_REQUIREMENT = "an existing non-empty backup file of the old collection (checked for existence and size only)"
_DEDICATED_TYPE_REQUIREMENT = "a new dedicated note type distinct from every legacy note type"
_DISPOSABLE_REQUIREMENT = "explicit confirmation that the legacy data is disposable"
_NEW_SCHEDULES_REQUIREMENT = "acknowledgement that every fresh-start card schedule is new"
_MATCHING_TYPE_REQUIREMENT = (
    "a fresh-start approval whose dedicated note type matches the export note type '{intended}'"
)

LegacyModelKind = Literal["current_learning_object", "legacy_per_exercise", "legacy_split_clone", "unrecognized"]
TransitionDecision = Literal["compatible_regeneration", "fresh_start", "retain_and_defer", "legacy_review_required"]


class LegacyTransitionError(ValueError):
    """Raised when a transition request is incompatible and must be rejected outright."""


@dataclass(frozen=True, slots=True)
class DestinationNoteEvidence:
    """One observed destination note relevant to a transition review.

    Evidence comes from a reviewed inventory of the destination collection
    (for example Anki's browser export or the ``inspect`` command); nothing
    here reads or mutates a collection.
    """

    first_field: str
    note_type: str = ""
    note_schema: str | None = None
    personal_notes: str = ""
    tags: tuple[str, ...] = ()
    has_review_history: bool = False
    card_count: int = 0


@dataclass(frozen=True, slots=True)
class FreshStartApproval:
    """The explicit owner confirmation that alone unlocks a legacy fresh start."""

    backup_path: Path
    dedicated_note_type: str
    data_is_disposable: bool = False
    acknowledges_new_schedules: bool = False

    def missing_requirements(self, legacy_note_types: Iterable[str] = ()) -> tuple[str, ...]:
        """Return every explicit confirmation the approval still lacks."""

        missing: list[str] = []
        if not self.backup_path.is_file() or self.backup_path.stat().st_size == 0:
            missing.append(_BACKUP_REQUIREMENT)
        dedicated = self.dedicated_note_type.strip()
        legacy_names = {name.strip() for name in legacy_note_types}
        if not dedicated or dedicated in legacy_names:
            missing.append(_DEDICATED_TYPE_REQUIREMENT)
        if not self.data_is_disposable:
            missing.append(_DISPOSABLE_REQUIREMENT)
        if not self.acknowledges_new_schedules:
            missing.append(_NEW_SCHEDULES_REQUIREMENT)
        return tuple(missing)


@dataclass(frozen=True, slots=True)
class LegacyTransitionReview:
    """One read-only transition decision for a single destination note."""

    decision: TransitionDecision
    model_kind: LegacyModelKind
    first_field: str
    reason: str
    requires: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LegacyTransitionPlan:
    """The aggregated review of one intended export against legacy evidence."""

    intended_note_type: str
    legacy_note_types: tuple[str, ...]
    decisions: tuple[LegacyTransitionReview, ...]
    old_collections_retained: bool = True

    @property
    def review_only(self) -> bool:
        """Whether every decision stays a no-write review outcome."""

        return all(review.decision in {"legacy_review_required", "retain_and_defer"} for review in self.decisions)


def is_legacy_note_schema(note_schema: str | None) -> bool:
    """Return whether a destination note schema belongs to a legacy model."""

    return note_schema in LEGACY_NOTE_SCHEMA_VERSIONS


def classify_legacy_model(first_field: str, note_schema: str | None) -> LegacyModelKind:
    """Classify one destination note by its identity prefix and schema.

    Contradictory evidence — a current identity with a legacy schema or the
    reverse — is never reinterpreted; it is unrecognized and stays a review
    outcome.
    """

    legacy_schema = is_legacy_note_schema(note_schema)
    if first_field.startswith(_LEGACY_ID_PREFIX):
        if note_schema is None or legacy_schema:
            return "legacy_per_exercise"
        return "unrecognized"
    if first_field.startswith(_CURRENT_ID_PREFIX):
        if note_schema is None or note_schema == NOTE_SCHEMA_VERSION:
            return "current_learning_object"
        return "unrecognized"
    if first_field.startswith(LEGACY_SPLIT_GUID_PREFIX):
        if note_schema is None:
            return "legacy_split_clone"
        return "unrecognized"
    if legacy_schema:
        return "legacy_per_exercise"
    return "unrecognized"


def evaluate_legacy_transition(
    evidence: DestinationNoteEvidence,
    *,
    intended_note_type: str,
    legacy_note_types: Iterable[str] = (),
    fresh_start: FreshStartApproval | None = None,
) -> LegacyTransitionReview:
    """Return the explicit, no-write decision for one destination note."""

    kind = classify_legacy_model(evidence.first_field, evidence.note_schema)
    if kind == "current_learning_object":
        return LegacyTransitionReview(
            decision="compatible_regeneration",
            model_kind=kind,
            first_field=evidence.first_field,
            reason=_REGENERATION_REASON,
        )
    if kind == "unrecognized":
        return LegacyTransitionReview(
            decision="legacy_review_required",
            model_kind=kind,
            first_field=evidence.first_field,
            reason=_UNRECOGNIZED_REASON,
            requires=("a reviewed inventory of the destination note model",),
        )
    if evidence.has_review_history:
        return _retain_decision(kind, evidence, _HISTORY_REASON)
    if evidence.personal_notes.strip():
        return _retain_decision(kind, evidence, _PERSONAL_REASON)
    if fresh_start is None:
        requires: tuple[str, ...] = (
            _BACKUP_REQUIREMENT,
            _DEDICATED_TYPE_REQUIREMENT,
            _DISPOSABLE_REQUIREMENT,
            _NEW_SCHEDULES_REQUIREMENT,
        )
    else:
        requires = fresh_start.missing_requirements(legacy_note_types)
        intended = intended_note_type.strip()
        if fresh_start.dedicated_note_type.strip() != intended:
            requires = (*requires, _MATCHING_TYPE_REQUIREMENT.format(intended=intended))
    if requires:
        return LegacyTransitionReview(
            decision="legacy_review_required",
            model_kind=kind,
            first_field=evidence.first_field,
            reason=_RECONFIRMATION_REASON,
            requires=requires,
        )
    return LegacyTransitionReview(
        decision="fresh_start",
        model_kind=kind,
        first_field=evidence.first_field,
        reason=_FRESH_START_REASON,
    )


def _retain_decision(
    kind: LegacyModelKind,
    evidence: DestinationNoteEvidence,
    reason: str,
) -> LegacyTransitionReview:
    return LegacyTransitionReview(
        decision="retain_and_defer",
        model_kind=kind,
        first_field=evidence.first_field,
        reason=reason,
    )


def plan_legacy_transition(
    profile: DeckProfile,
    evidences: Sequence[DestinationNoteEvidence],
    *,
    legacy_note_types: Iterable[str] = (),
    fresh_start: FreshStartApproval | None = None,
) -> LegacyTransitionPlan:
    """Review one intended export profile against destination legacy evidence.

    A profile that would write new-model rows into a legacy note type is
    rejected outright instead of silently reinterpreting the legacy model.
    """

    intended = profile.generated_note_type
    normalized_legacy = tuple(dict.fromkeys(name.strip() for name in legacy_note_types if name.strip()))
    if intended in normalized_legacy:
        raise LegacyTransitionError(
            f"The profile's generated note type '{intended}' is a legacy note model; "
            "exporting new-model rows into it would silently reinterpret the legacy model. "
            "Create a new dedicated note type instead."
        )
    decisions = tuple(
        evaluate_legacy_transition(
            evidence,
            intended_note_type=intended,
            legacy_note_types=normalized_legacy,
            fresh_start=fresh_start,
        )
        for evidence in evidences
    )
    return LegacyTransitionPlan(
        intended_note_type=intended,
        legacy_note_types=normalized_legacy,
        decisions=decisions,
    )


__all__ = [
    "CSV_CONSOLIDATION_HISTORY_STATEMENT",
    "DestinationNoteEvidence",
    "FreshStartApproval",
    "LEGACY_LATINITAS_ID_VERSION",
    "LEGACY_NOTE_SCHEMA_VERSIONS",
    "LEGACY_SPLIT_GUID_PREFIX",
    "LegacyTransitionError",
    "LegacyTransitionPlan",
    "LegacyTransitionReview",
    "classify_legacy_model",
    "evaluate_legacy_transition",
    "is_legacy_note_schema",
    "plan_legacy_transition",
]
