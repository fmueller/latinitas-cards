"""Stable logical identities for generated Latinitas notes.

Identity is deliberately separate from rendered content and transport-local Anki
identifiers.  One learning-object note is identified by its immutable source
identity, an optional persisted CSV source scope, and the reviewed learning
object key only.  Recipe selection, semantic card roles, wording, profiles, and
software versions never participate in note identity; card semantic keys are
derived separately and never feed the note identifier.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .profile import DeckProfile
from .sources import CanonicalSourceRecord

LATINITAS_ID_VERSION = "v2"
_LATINITAS_ID_PREFIX = f"latinitas-{LATINITAS_ID_VERSION}-"
_NOTE_FAMILY_IDENTITY_SEED = b"latinitas-note-family-v1\0"
_ANKI_GUID_PREFIX = "anki-guid-v1-"
_CARD_KEY_DELIMITER = ":"


class IdentityError(ValueError):
    """Raised when a source or logical learning-object identity is incomplete."""


@dataclass(frozen=True, slots=True)
class ResolvedSourceIdentity:
    """One resolved source identity plus its persisted CSV scope, if any."""

    value: str
    scope: str | None = None


def derive_latinitas_id(
    source_identity: str,
    object_key: str,
    *,
    source_scope: str | None = None,
) -> str:
    """Derive a stable note ID for one learning object.

    The canonical JSON payload and the domain-separated note-family digest keep
    the identifier tied to the immutable source identity, its persisted CSV
    scope, and the reviewed object key, while excluding recipes, card roles,
    wording, HTML, tags, profiles, software versions, and local Anki IDs.
    """

    payload = _note_identity_payload(source_identity, object_key, source_scope)
    digest = hashlib.sha256(_NOTE_FAMILY_IDENTITY_SEED + payload).hexdigest()
    return f"{_LATINITAS_ID_PREFIX}{digest}"


def derive_anki_guid_from_latinitas_id(latinitas_id: str) -> str:
    """Derive a deterministic transport GUID from an already-derived Latinitas ID."""

    value = _required_part(latinitas_id, "latinitas_id")
    digest = hashlib.sha256(b"latinitas-anki-guid-v1\0" + value.encode("utf-8")).hexdigest()
    return f"{_ANKI_GUID_PREFIX}{digest}"


def derive_anki_guid(source_identity: str, object_key: str, *, source_scope: str | None = None) -> str:
    """Derive the future Anki note GUID from the same logical note identity."""

    latinitas_id = derive_latinitas_id(source_identity, object_key, source_scope=source_scope)
    return derive_anki_guid_from_latinitas_id(latinitas_id)


def derive_card_semantic_key(recipe_identity: str, semantic_role: str) -> str:
    """Derive the stable semantic key of one card on a learning-object note.

    Card keys are deliberately separate from note identity: they name the
    recipe plus confirmed semantic role that a template slot renders, and they
    never feed ``derive_latinitas_id``.
    """

    recipe = _required_part(recipe_identity, "recipe_identity")
    role = _required_part(semantic_role, "semantic_role")
    for name, part in (("recipe_identity", recipe), ("semantic_role", role)):
        if _CARD_KEY_DELIMITER in part:
            raise IdentityError(f"{name} must not contain the card-key delimiter '{_CARD_KEY_DELIMITER}'")
    return f"{recipe}{_CARD_KEY_DELIMITER}{role}"


def resolve_source_identity(
    record: CanonicalSourceRecord,
    profile: DeckProfile,
    *,
    manifest_identity: str | None = None,
    source_scope: str | None = None,
) -> ResolvedSourceIdentity:
    """Resolve a profile-approved immutable identity for one canonical source record.

    CSV records use a configured source-ID field or a manifest assignment and
    require a persisted source scope so independent sources stay distinct.
    Native APKG/COLPKG note GUIDs are accepted as globally scoped identities,
    while their local numeric note IDs are never identity material.
    """

    strategy = profile.source_identity.strategy
    if strategy == "manifest":
        identity = _required_part(manifest_identity, "manifest source identity")
        return ResolvedSourceIdentity(identity, _required_scope(source_scope))

    if strategy == "source_id_field":
        field = profile.source_identity.field
        if field is None:
            raise IdentityError("source identity field is required for a source-ID strategy")
        value = record.source_identity or record.fields.get(field)
        if not isinstance(value, str) or not value.strip():
            raise IdentityError(f"record is missing a stable source identity in field '{field}'")
        if record.source_kind == "csv":
            return ResolvedSourceIdentity(value, _required_scope(source_scope))
        return ResolvedSourceIdentity(value, None)

    if record.source_kind not in {"apkg", "colpkg"}:
        raise IdentityError("CSV records require a stable source identity field or manifest assignment")
    return ResolvedSourceIdentity(
        _required_part(record.note_guid or record.source_identity, "stable source identity"), None
    )


def resolve_source_identities(
    records: Sequence[CanonicalSourceRecord],
    profile: DeckProfile,
    *,
    manifest_identities: Sequence[str | None] | None = None,
    source_scope: str | None = None,
) -> tuple[ResolvedSourceIdentity, ...]:
    """Resolve and validate a unique source identity for every selected record."""

    if manifest_identities is not None and len(manifest_identities) != len(records):
        raise IdentityError("manifest source identities must match the selected record count")
    identities = tuple(
        resolve_source_identity(
            record,
            profile,
            manifest_identity=None if manifest_identities is None else manifest_identities[index],
            source_scope=source_scope,
        )
        for index, record in enumerate(records)
    )
    duplicate_count = sum(count - 1 for count in Counter(identities).values() if count > 1)
    if duplicate_count:
        raise IdentityError(f"source identities must be unique; {duplicate_count} duplicate assignment(s) found")
    return identities


def _note_identity_payload(source_identity: str, object_key: str, source_scope: str | None) -> bytes:
    values: dict[str, str] = {
        "object_key": _required_part(object_key, "object_key"),
        "source_identity": _required_part(source_identity, "source_identity"),
    }
    if source_scope is not None:
        values["source_scope"] = _required_part(source_scope, "source_scope")
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _required_part(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IdentityError(f"{name} must be a non-empty string")
    return value


def _required_scope(source_scope: str | None) -> str:
    return _required_part(source_scope, "source scope of a CSV source")


__all__ = [
    "IdentityError",
    "LATINITAS_ID_VERSION",
    "ResolvedSourceIdentity",
    "derive_anki_guid",
    "derive_anki_guid_from_latinitas_id",
    "derive_card_semantic_key",
    "derive_latinitas_id",
    "resolve_source_identity",
    "resolve_source_identities",
]
