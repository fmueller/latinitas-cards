"""Versioned prior-export eligibility checkpoints for scoped CSV sources.

A checkpoint records exported state only: which learning-object note IDs were
last serialized with which eligible card keys, under which source scope, note
family/schema, and template registry, together with the fingerprint of the
committed CSV output.  It is never evidence about a live destination
collection, and generating a file does not establish that it was imported.

The checkpoint advances only through the recoverable export boundary together
with the output and the identity manifest; a read-only preview never touches
it.  Missing, corrupt, or incompatible checkpoint state requires recovery,
review, or an explicitly confirmed fresh import; it never implies an empty
prior card set.  Withheld rows, absent objects, parser failures, and partial
exports retain their last safe evidence instead of asserting that cards
disappeared.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .cards import TEMPLATE_REGISTRY_DIGEST, TEMPLATE_REGISTRY_VERSION
from .notes import NOTE_SCHEMA_VERSION, GeneratedNote

CHECKPOINT_SCHEMA_VERSION = 1
NOTE_FAMILY = "latinitas-note-family-v1"

#: Binding scope for package (APKG/COLPKG) sources whose GUID identities are
#: globally scoped and therefore carry no per-source CSV scope token.
GLOBAL_SOURCE_SCOPE = "global"

CheckpointReviewKind = Literal["withheld", "absent", "checkpoint_confirmation"]


class CheckpointError(ValueError):
    """Raised when checkpoint state is corrupt or incompatible."""


@dataclass(frozen=True, slots=True)
class CardEligibilityReview:
    """A structured card-eligibility condition that needs an explicit decision."""

    kind: CheckpointReviewKind
    latinitas_id: str | None
    source_identity: str | None
    card_keys: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class PriorExportCheckpoint:
    """The retained prior-export evidence of one scoped CSV source."""

    source_scope: str
    note_schema: str
    template_registry_digest: str
    note_family: str = NOTE_FAMILY
    objects: tuple[tuple[str, tuple[str, ...]], ...] = ()
    export_fingerprint: str = ""
    checkpoint_version: int = CHECKPOINT_SCHEMA_VERSION
    template_registry_version: int = TEMPLATE_REGISTRY_VERSION

    def __post_init__(self) -> None:
        if self.checkpoint_version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(
                f"unsupported checkpoint schema version {self.checkpoint_version}; "
                f"supported version is {CHECKPOINT_SCHEMA_VERSION}"
            )
        if not self.source_scope.strip():
            raise CheckpointError("checkpoint requires a non-empty source scope")
        if not self.note_schema.strip() or not self.template_registry_digest.strip():
            raise CheckpointError("checkpoint requires its note-schema and template-registry bindings")
        if self.note_family != NOTE_FAMILY:
            raise CheckpointError(f"checkpoint binds note family {self.note_family!r}, expected {NOTE_FAMILY!r}")
        object_ids = [object_id for object_id, _keys in self.objects]
        if len(set(object_ids)) != len(object_ids):
            raise CheckpointError("checkpoint object IDs must be distinct")
        for object_id, keys in self.objects:
            if not object_id.strip():
                raise CheckpointError("checkpoint object IDs must be non-empty")
            if len(set(keys)) != len(keys) or any(not key.strip() for key in keys):
                raise CheckpointError("checkpoint card keys must be distinct and non-empty")

    @property
    def object_keys(self) -> dict[str, tuple[str, ...]]:
        return dict(self.objects)

    @classmethod
    def scoped(
        cls,
        *,
        source_scope: str,
        note_schema: str = NOTE_SCHEMA_VERSION,
        template_registry_digest: str = TEMPLATE_REGISTRY_DIGEST,
        objects: Sequence[tuple[str, tuple[str, ...]]] = (),
        export_fingerprint: str = "",
    ) -> PriorExportCheckpoint:
        return cls(
            source_scope=source_scope,
            note_schema=note_schema,
            template_registry_digest=template_registry_digest,
            objects=tuple(objects),
            export_fingerprint=export_fingerprint,
        )

    def incompatibility(self, *, source_scope: str) -> str | None:
        """Return why this checkpoint cannot authorize the current run, if any."""

        if self.source_scope != source_scope:
            return "the persisted source scope differs from the current committed scope"
        if self.note_family != NOTE_FAMILY:
            return "the checkpoint records a different note family"
        if self.note_schema != NOTE_SCHEMA_VERSION:
            return (
                f"the checkpoint binds note schema {self.note_schema}, "
                f"while this release writes schema {NOTE_SCHEMA_VERSION}"
            )
        if self.template_registry_digest != TEMPLATE_REGISTRY_DIGEST:
            return "the checkpoint binds a different template-slot registry"
        return None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PriorExportCheckpoint:
        scope = value.get("source_scope")
        note_schema = value.get("note_schema")
        registry = value.get("template_registry_digest")
        fingerprint = value.get("export_fingerprint", "")
        family = value.get("note_family", NOTE_FAMILY)
        raw_objects = value.get("objects")
        version = value.get("checkpoint_version", CHECKPOINT_SCHEMA_VERSION)
        if not isinstance(scope, str) or not isinstance(note_schema, str) or not isinstance(registry, str):
            raise CheckpointError("checkpoint requires source_scope, note_schema and template_registry_digest strings")
        if not isinstance(fingerprint, str):
            raise CheckpointError("checkpoint export_fingerprint must be a string")
        if not isinstance(family, str):
            raise CheckpointError("checkpoint note_family must be a string")
        if not isinstance(version, int) or isinstance(version, bool):
            raise CheckpointError("checkpoint_version must be an integer")
        if not isinstance(raw_objects, list):
            raise CheckpointError("checkpoint objects must be a list")
        objects: list[tuple[str, tuple[str, ...]]] = []
        for raw_object in raw_objects:
            if not isinstance(raw_object, Mapping):
                raise CheckpointError("checkpoint objects must be objects")
            object_id = raw_object.get("latinitas_id")
            raw_keys = raw_object.get("card_keys")
            if not isinstance(object_id, str) or not isinstance(raw_keys, list):
                raise CheckpointError("checkpoint objects require latinitas_id and card_keys")
            if not all(isinstance(key, str) for key in raw_keys):
                raise CheckpointError("checkpoint card keys must be strings")
            objects.append((object_id, tuple(raw_keys)))
        return cls(
            source_scope=scope,
            note_schema=note_schema,
            template_registry_digest=registry,
            note_family=family,
            objects=tuple(objects),
            export_fingerprint=fingerprint,
            checkpoint_version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> PriorExportCheckpoint:
        try:
            decoded: object = json.loads(value)
        except json.JSONDecodeError as error:
            raise CheckpointError("checkpoint is not valid JSON") from error
        if not isinstance(decoded, Mapping):
            raise CheckpointError("checkpoint JSON must contain an object")
        return cls.from_mapping(decoded)

    @classmethod
    def load(cls, path: str | Path) -> PriorExportCheckpoint:
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise CheckpointError(f"checkpoint '{Path(path).name}' is not valid UTF-8 text") from error
        return cls.from_json(raw)

    def to_mapping(self) -> dict[str, object]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "note_family": NOTE_FAMILY,
            "source_scope": self.source_scope,
            "note_schema": self.note_schema,
            "template_registry_digest": self.template_registry_digest,
            "template_registry_version": self.template_registry_version,
            "export_fingerprint": self.export_fingerprint,
            "objects": [{"latinitas_id": object_id, "card_keys": list(keys)} for object_id, keys in self.objects],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def compare_export_with_checkpoint(
    prior: PriorExportCheckpoint | None,
    notes: Sequence[GeneratedNote],
) -> tuple[frozenset[str], tuple[CardEligibilityReview, ...]]:
    """Compare current notes with retained prior evidence.

    A row is withheld whole whenever any previously exported card key is not
    currently eligible; the detectable causes are data-driven eligibility
    loss, recipe/profile disablement, and shared-field degradation that
    removes a card's required data.  Prior objects absent from the current
    notes are reported without asserting that their cards disappeared.
    """

    if prior is None:
        return frozenset(), ()
    prior_keys = prior.object_keys
    current_ids = {note.latinitas_id for note in notes}
    withheld_ids: set[str] = set()
    reviews: list[CardEligibilityReview] = []
    for note in notes:
        exported_keys = prior_keys.get(note.latinitas_id)
        if not exported_keys:
            continue
        current_keys = set(note.card_keys)
        lost = tuple(key for key in exported_keys if key not in current_keys)
        if lost:
            withheld_ids.add(note.latinitas_id)
            reviews.append(
                CardEligibilityReview(
                    kind="withheld",
                    latinitas_id=note.latinitas_id,
                    source_identity=note.provenance.source_identity,
                    card_keys=lost,
                    message=(
                        "Previously exported card keys are no longer eligible; the whole note row is "
                        "withheld pending review instead of clearing fronts or deleting cards."
                    ),
                )
            )
    for object_id, keys in prior.objects:
        if object_id in current_ids:
            continue
        reviews.append(
            CardEligibilityReview(
                kind="absent",
                latinitas_id=object_id,
                source_identity=None,
                card_keys=keys,
                message=(
                    "The object is absent from this export (removed or not parsed); its last safe "
                    "export evidence is retained."
                ),
            )
        )
    return frozenset(withheld_ids), tuple(reviews)


def advance_checkpoint(
    prior: PriorExportCheckpoint | None,
    *,
    source_scope: str,
    exported_notes: Sequence[GeneratedNote],
    export_fingerprint: str,
) -> PriorExportCheckpoint:
    """Build the next checkpoint, retaining last-safe evidence for untouched rows.

    Exported notes record their current eligible keys; withheld, absent, and
    zero-eligible objects keep their prior evidence untouched.
    """

    objects = dict(prior.objects) if prior is not None else {}
    for note in exported_notes:
        objects[note.latinitas_id] = tuple(note.card_keys)
    ordered = tuple((object_id, objects[object_id]) for object_id in sorted(objects))
    return PriorExportCheckpoint.scoped(
        source_scope=source_scope,
        objects=ordered,
        export_fingerprint=export_fingerprint,
    )


def export_fingerprint(payload: bytes) -> str:
    """Return the deterministic fingerprint of one committed CSV output."""

    return f"export-sha256:{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CardEligibilityReview",
    "CheckpointError",
    "GLOBAL_SOURCE_SCOPE",
    "NOTE_FAMILY",
    "PriorExportCheckpoint",
    "advance_checkpoint",
    "compare_export_with_checkpoint",
    "export_fingerprint",
]
