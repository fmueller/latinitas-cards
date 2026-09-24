"""Stable logical identities for generated Latinitas notes.

Identity is deliberately separate from rendered content and transport-local Anki
identifiers.  A generated exercise is identified by its immutable source
identity, recipe identity, and semantic exercise key only.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence

from .profile import DeckProfile
from .sources import CanonicalSourceRecord

LATINITAS_ID_VERSION = "v1"
_LATINITAS_ID_PREFIX = f"latinitas-{LATINITAS_ID_VERSION}-"
_ANKI_GUID_PREFIX = "anki-guid-v1-"


class IdentityError(ValueError):
    """Raised when a source or logical exercise identity is incomplete."""


def derive_latinitas_id(source_identity: str, recipe_identity: str, exercise_key: str) -> str:
    """Derive a stable ID from the three immutable parts of an exercise.

    The canonical JSON payload and domain-separated SHA-256 digest avoid
    delimiter collisions while excluding prompt, answer, gloss, HTML, tags,
    software versions, and local Anki IDs from the result.
    """

    payload = _logical_identity_payload(source_identity, recipe_identity, exercise_key)
    digest = hashlib.sha256(b"latinitas-logical-id-v1\0" + payload).hexdigest()
    return f"{_LATINITAS_ID_PREFIX}{digest}"


def derive_anki_guid(source_identity: str, recipe_identity: str, exercise_key: str) -> str:
    """Derive the future Anki note GUID from the same logical exercise identity."""

    return derive_anki_guid_from_latinitas_id(derive_latinitas_id(source_identity, recipe_identity, exercise_key))


def derive_anki_guid_from_latinitas_id(latinitas_id: str) -> str:
    """Derive a deterministic transport GUID from an already-derived Latinitas ID."""

    value = _required_part(latinitas_id, "latinitas_id")
    digest = hashlib.sha256(b"latinitas-anki-guid-v1\0" + value.encode("utf-8")).hexdigest()
    return f"{_ANKI_GUID_PREFIX}{digest}"


def resolve_source_identity(
    record: CanonicalSourceRecord,
    profile: DeckProfile,
    *,
    manifest_identity: str | None = None,
) -> str:
    """Resolve a profile-approved immutable identity for one canonical source record.

    CSV records may use a configured source-ID field or a manifest assignment.
    Native APKG/COLPKG note GUIDs are accepted, while their local numeric note
    IDs are never used as relationships or identity material.
    """

    strategy = profile.source_identity.strategy
    if strategy == "manifest":
        return _required_part(manifest_identity, "manifest source identity")

    if strategy == "source_id_field":
        field = profile.source_identity.field
        if field is None:
            raise IdentityError("source identity field is required for a source-ID strategy")
        value = record.source_identity or record.fields.get(field)
        if not isinstance(value, str) or not value.strip():
            raise IdentityError(f"record is missing a stable source identity in field '{field}'")
        return value

    if record.source_kind not in {"apkg", "colpkg"}:
        raise IdentityError("CSV records require a stable source identity field or manifest assignment")
    return _required_part(record.note_guid or record.source_identity, "stable source identity")


def resolve_source_identities(
    records: Sequence[CanonicalSourceRecord],
    profile: DeckProfile,
    *,
    manifest_identities: Sequence[str | None] | None = None,
) -> tuple[str, ...]:
    """Resolve and validate a unique source identity for every selected record."""

    if manifest_identities is not None and len(manifest_identities) != len(records):
        raise IdentityError("manifest source identities must match the selected record count")
    identities = tuple(
        resolve_source_identity(
            record,
            profile,
            manifest_identity=None if manifest_identities is None else manifest_identities[index],
        )
        for index, record in enumerate(records)
    )
    duplicate_count = sum(count - 1 for count in Counter(identities).values() if count > 1)
    if duplicate_count:
        raise IdentityError(f"source identities must be unique; {duplicate_count} duplicate assignment(s) found")
    return identities


def _logical_identity_payload(source_identity: str, recipe_identity: str, exercise_key: str) -> bytes:
    values = {
        "exercise_key": _required_part(exercise_key, "exercise_key"),
        "recipe_identity": _required_part(recipe_identity, "recipe_identity"),
        "source_identity": _required_part(source_identity, "source_identity"),
    }
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _required_part(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IdentityError(f"{name} must be a non-empty string")
    return value


make_latinitas_id = derive_latinitas_id
make_anki_guid = derive_anki_guid

__all__ = [
    "IdentityError",
    "LATINITAS_ID_VERSION",
    "derive_anki_guid",
    "derive_anki_guid_from_latinitas_id",
    "derive_latinitas_id",
    "make_anki_guid",
    "make_latinitas_id",
    "resolve_source_identity",
    "resolve_source_identities",
]
