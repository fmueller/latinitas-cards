"""Sidecar identity manifests for CSV sources without stable ID columns.

Exact, unique row fingerprints may be reused automatically.  Edits, duplicate
rows, insertions, removals, and stale manifest state remain explicit review
items until the caller approves a reuse or allocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from .sources import CanonicalSourceRecord

MANIFEST_SCHEMA_VERSION = 1
ManifestReviewKind = Literal["stale_manifest", "edited", "duplicate", "unmatched", "removed"]
AssignmentAction = Literal["reused", "approved_reuse", "allocated"]


class ManifestError(ValueError):
    """Raised when a sidecar manifest cannot safely be reconciled."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One persisted source identity and its last known row fingerprint."""

    source_identity: str
    fingerprint: str
    last_row_index: int | None = None
    active: bool = True

    def __post_init__(self) -> None:
        if not self.source_identity.strip():
            raise ManifestError("manifest source identities must be non-empty")
        if not self.fingerprint.strip():
            raise ManifestError("manifest row fingerprints must be non-empty")
        if self.last_row_index is not None and self.last_row_index < 0:
            raise ManifestError("manifest row indexes must not be negative")


@dataclass(frozen=True, slots=True)
class CsvIdentityManifest:
    """Human-readable, persistent assignments for an ID-less CSV."""

    source_columns: tuple[str, ...]
    entries: tuple[ManifestEntry, ...] = ()
    source_digest: str = ""
    next_source_number: int = 1
    schema_version: int = MANIFEST_SCHEMA_VERSION
    mapping_digest: str = ""

    def __post_init__(self) -> None:
        columns = tuple(self.source_columns)
        if not columns or any(not column.strip() for column in columns):
            raise ManifestError("manifest source columns must be non-empty")
        if len(set(columns)) != len(columns):
            raise ManifestError("manifest source columns must be distinct")
        entries = tuple(self.entries)
        identities = [entry.source_identity for entry in entries]
        if len(set(identities)) != len(identities):
            raise ManifestError("manifest source identities must be distinct")
        if self.next_source_number < 1:
            raise ManifestError("manifest allocation counter must be positive")
        object.__setattr__(self, "source_columns", columns)
        object.__setattr__(self, "entries", entries)

    @classmethod
    def empty(cls, source_columns: Sequence[str]) -> CsvIdentityManifest:
        """Create an empty manifest for a validated CSV header."""

        return cls(source_columns=tuple(source_columns))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CsvIdentityManifest:
        """Load a manifest mapping without trusting mutable row positions."""

        raw_columns = value.get("source_columns")
        raw_entries = value.get("entries")
        if not isinstance(raw_columns, list) or not all(isinstance(item, str) for item in raw_columns):
            raise ManifestError("manifest source_columns must be a list of strings")
        if not isinstance(raw_entries, list):
            raise ManifestError("manifest entries must be a list")

        entries: list[ManifestEntry] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise ManifestError("manifest entries must be objects")
            source_identity = raw_entry.get("source_identity")
            fingerprint = raw_entry.get("fingerprint")
            last_row_index = raw_entry.get("last_row_index")
            active = raw_entry.get("active", True)
            if not isinstance(source_identity, str) or not isinstance(fingerprint, str):
                raise ManifestError("manifest entries require source_identity and fingerprint strings")
            if not isinstance(active, bool):
                raise ManifestError("manifest entry active must be a boolean")
            if last_row_index is not None and (not isinstance(last_row_index, int) or isinstance(last_row_index, bool)):
                raise ManifestError("manifest entry last_row_index must be an integer or null")
            entries.append(
                ManifestEntry(
                    source_identity=source_identity,
                    fingerprint=fingerprint,
                    last_row_index=last_row_index,
                    active=active,
                )
            )

        schema_version = value.get("schema_version", MANIFEST_SCHEMA_VERSION)
        next_source_number = value.get("next_source_number", 1)
        source_digest = value.get("source_digest", "")
        mapping_digest = value.get("mapping_digest", "")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ManifestError("manifest schema_version must be an integer")
        if not isinstance(next_source_number, int) or isinstance(next_source_number, bool):
            raise ManifestError("manifest next_source_number must be an integer")
        if not isinstance(source_digest, str):
            raise ManifestError("manifest source_digest must be a string")
        if not isinstance(mapping_digest, str):
            raise ManifestError("manifest mapping_digest must be a string")
        return cls(
            source_columns=tuple(cast(list[str], raw_columns)),
            entries=tuple(entries),
            source_digest=source_digest,
            next_source_number=next_source_number,
            schema_version=schema_version,
            mapping_digest=mapping_digest,
        )

    @classmethod
    def from_json(cls, value: str) -> CsvIdentityManifest:
        try:
            decoded: object = json.loads(value)
        except json.JSONDecodeError as error:
            raise ManifestError("manifest is not valid JSON") from error
        if not isinstance(decoded, Mapping):
            raise ManifestError("manifest JSON must contain an object")
        return cls.from_mapping(decoded)

    @classmethod
    def load(cls, path: str | Path) -> CsvIdentityManifest:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_columns": list(self.source_columns),
            "source_digest": self.source_digest,
            "mapping_digest": self.mapping_digest,
            "next_source_number": self.next_source_number,
            "entries": [
                {
                    "source_identity": entry.source_identity,
                    "fingerprint": entry.fingerprint,
                    "last_row_index": entry.last_row_index,
                    "active": entry.active,
                }
                for entry in self.entries
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
                temporary_file.write(self.to_json())
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)
            raise


@dataclass(frozen=True, slots=True)
class ManifestReviewItem:
    """A row or manifest condition that needs an explicit decision."""

    kind: ManifestReviewKind
    row_index: int | None
    source_identity: str | None
    candidate_identities: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class ManifestAssignment:
    """A safe or explicitly approved row-to-source identity assignment."""

    row_index: int
    source_identity: str
    action: AssignmentAction


@dataclass(frozen=True, slots=True)
class ManifestReconciliation:
    """Pure reconciliation output, including a candidate persisted manifest."""

    assignments: tuple[ManifestAssignment, ...]
    reviews: tuple[ManifestReviewItem, ...]
    manifest: CsvIdentityManifest

    @property
    def identities_by_row(self) -> dict[int, str]:
        return {assignment.row_index: assignment.source_identity for assignment in self.assignments}

    @property
    def complete(self) -> bool:
        return not self.reviews


def reconcile_csv_manifest(
    records: Sequence[CanonicalSourceRecord],
    manifest: CsvIdentityManifest | None,
    *,
    approved_reuse: Mapping[int, str] | None = None,
    approved_allocations: Mapping[int, str | None] | Iterable[int] | None = None,
    approved_removals: Iterable[str] | None = None,
) -> ManifestReconciliation:
    """Reconcile CSV records without silently transferring an identity.

    Unique exact fingerprints are safe to reuse.  ``approved_reuse`` explicitly
    authorizes a reuse for an edited or ambiguous row.  ``approved_allocations``
    explicitly authorizes a new ID; its value may provide the ID or be ``None``
    to allocate the next manifest ID. Removed identities remain reviewable until
    their source identity is listed in ``approved_removals``.
    """

    rows = tuple(records)
    columns = _source_columns(rows, manifest)
    current_fingerprints = tuple(_record_fingerprint(record) for record in rows)
    current_digest = _fingerprint_digest(current_fingerprints)
    base_manifest = manifest or CsvIdentityManifest.empty(columns)
    if manifest is not None and not columns:
        columns = manifest.source_columns

    approvals_reuse = dict(approved_reuse or {})
    approvals_allocate = _normalise_allocations(approved_allocations)
    approvals_remove = set(approved_removals or ())
    _validate_approval_rows(approvals_reuse, approvals_allocate, len(rows))

    active_entries = tuple(entry for entry in base_manifest.entries if entry.active)
    entries_by_identity = {entry.source_identity: entry for entry in base_manifest.entries}
    unknown_removals = approvals_remove - entries_by_identity.keys()
    if unknown_removals:
        raise ManifestError("approved removal references an unknown source identity")
    entries_by_fingerprint: dict[str, list[ManifestEntry]] = {}
    for entry in active_entries:
        entries_by_fingerprint.setdefault(entry.fingerprint, []).append(entry)
    current_counts = Counter(current_fingerprints)
    columns_match = columns == base_manifest.source_columns
    manifest_snapshot_digest = _fingerprint_digest(entry.fingerprint for entry in active_entries)
    manifest_mapping_digest = _mapping_digest(base_manifest.entries)
    structurally_stale = (
        base_manifest.schema_version != MANIFEST_SCHEMA_VERSION
        or not columns_match
        or (bool(base_manifest.entries) and base_manifest.source_digest != manifest_snapshot_digest)
        or (bool(base_manifest.entries) and base_manifest.mapping_digest != manifest_mapping_digest)
    )

    assignments: dict[int, ManifestAssignment] = {}
    used_identities: set[str] = set()
    review_candidates: dict[int, tuple[ManifestReviewKind, tuple[str, ...], str]] = {}
    reviewed_entry_ids: set[str] = set()

    if columns_match and not structurally_stale:
        for row_index, fingerprint in enumerate(current_fingerprints):
            candidates = entries_by_fingerprint.get(fingerprint, [])
            candidate_ids = tuple(entry.source_identity for entry in candidates)
            if current_counts[fingerprint] > 1 or len(candidates) != 1:
                if candidates:
                    reviewed_entry_ids.update(candidate_ids)
                if current_counts[fingerprint] > 1:
                    review_candidates[row_index] = (
                        "duplicate",
                        candidate_ids,
                        "row fingerprint occurs more than once; choose reuse or allocation explicitly",
                    )
                elif candidates:
                    review_candidates[row_index] = (
                        "duplicate",
                        candidate_ids,
                        "manifest has multiple identities for this row fingerprint",
                    )
                continue
            entry = candidates[0]
            if entry.source_identity not in used_identities:
                assignments[row_index] = ManifestAssignment(row_index, entry.source_identity, "reused")
                used_identities.add(entry.source_identity)

    for row_index, source_identity in approvals_reuse.items():
        approved_entry = entries_by_identity.get(source_identity)
        if approved_entry is None:
            raise ManifestError(f"approved reuse references unknown source identity for row {row_index}")
        existing = assignments.get(row_index)
        if existing is not None and existing.source_identity != source_identity:
            raise ManifestError(f"row {row_index} has conflicting reuse and automatic assignments")
        if source_identity in used_identities and (existing is None or existing.source_identity != source_identity):
            raise ManifestError(f"source identity '{source_identity}' is approved for more than one row")
        assignments[row_index] = ManifestAssignment(row_index, source_identity, "approved_reuse")
        used_identities.add(source_identity)
        review_candidates.pop(row_index, None)
        reviewed_entry_ids.add(source_identity)

    next_source_number = base_manifest.next_source_number
    for row_index in approvals_allocate:
        if row_index in assignments:
            raise ManifestError(f"row {row_index} has conflicting reuse and allocation approvals")
        requested_identity = approvals_allocate[row_index]
        if requested_identity is None:
            requested_identity, next_source_number = _allocate_identity(
                entries_by_identity.keys(), next_source_number, used_identities
            )
        elif not requested_identity.strip():
            raise ManifestError(f"approved allocation for row {row_index} must be non-empty")
        if requested_identity in entries_by_identity or requested_identity in used_identities:
            raise ManifestError(f"allocated source identity '{requested_identity}' is already present")
        assignments[row_index] = ManifestAssignment(row_index, requested_identity, "allocated")
        used_identities.add(requested_identity)
        review_candidates.pop(row_index, None)

    conflicting_removals = approvals_remove & used_identities
    if conflicting_removals:
        raise ManifestError("approved removal conflicts with a current row assignment")

    for row_index, _fingerprint in enumerate(current_fingerprints):
        if row_index in assignments or row_index in review_candidates:
            continue
        previous = next(
            (
                entry
                for entry in active_entries
                if entry.last_row_index == row_index and entry.source_identity not in used_identities
            ),
            None,
        )
        if previous is not None:
            review_candidates[row_index] = (
                "edited",
                (previous.source_identity,),
                "row changed without a stable ID; approve reuse or allocate a new identity",
            )
            reviewed_entry_ids.add(previous.source_identity)
        else:
            review_candidates[row_index] = (
                "unmatched",
                (),
                "row has no exact manifest match; approve allocation before assigning an identity",
            )

    removed_reviews: list[ManifestReviewItem] = []
    for entry in active_entries:
        if entry.source_identity in used_identities or entry.source_identity in reviewed_entry_ids:
            continue
        if entry.source_identity in approvals_remove:
            continue
        removed_reviews.append(
            ManifestReviewItem(
                kind="removed",
                row_index=None,
                source_identity=entry.source_identity,
                candidate_identities=(),
                message="manifest identity is absent from the current CSV; retain as a tombstone until reviewed",
            )
        )

    reviews: list[ManifestReviewItem] = [
        ManifestReviewItem(
            kind=kind,
            row_index=row_index,
            source_identity=candidates[0] if kind == "edited" and candidates else None,
            candidate_identities=candidates,
            message=message,
        )
        for row_index, (kind, candidates, message) in sorted(review_candidates.items())
    ]
    reviews.extend(removed_reviews)
    has_unresolved_changes = bool(review_candidates or removed_reviews)
    source_snapshot_is_unresolved = base_manifest.source_digest != current_digest and has_unresolved_changes
    stale_requires_review = structurally_stale and not (
        bool(approvals_reuse or approvals_allocate or approvals_remove)
        and not review_candidates
        and not removed_reviews
    )
    if manifest is not None and (stale_requires_review or source_snapshot_is_unresolved):
        reviews.insert(
            0,
            ManifestReviewItem(
                kind="stale_manifest",
                row_index=None,
                source_identity=None,
                candidate_identities=(),
                message="manifest snapshot or schema is stale; inspect row decisions before persisting it",
            ),
        )

    updated_manifest = _updated_manifest(
        base_manifest,
        columns,
        current_fingerprints,
        assignments,
        approvals_remove,
        next_source_number,
        current_digest,
        has_unresolved_changes,
        structurally_stale,
    )
    ordered_assignments = tuple(assignments[row_index] for row_index in sorted(assignments))
    return ManifestReconciliation(ordered_assignments, tuple(reviews), updated_manifest)


def _source_columns(records: Sequence[CanonicalSourceRecord], manifest: CsvIdentityManifest | None) -> tuple[str, ...]:
    if records:
        columns = tuple(records[0].fields)
        if any(tuple(record.fields) != columns for record in records[1:]):
            raise ManifestError("CSV records must use the same source columns")
        return columns
    if manifest is None:
        raise ManifestError("source columns are required to reconcile an empty CSV")
    return manifest.source_columns


def _record_fingerprint(record: CanonicalSourceRecord) -> str:
    if record.provenance.content_hash:
        return record.provenance.content_hash
    canonical = json.dumps(list(record.fields.items()), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fingerprint_digest(fingerprints: Iterable[str]) -> str:
    encoded = json.dumps(sorted(fingerprints), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping_digest(entries: Iterable[ManifestEntry]) -> str:
    mapping = sorted(
        (
            entry.source_identity,
            entry.fingerprint,
            entry.last_row_index,
            entry.active,
        )
        for entry in entries
    )
    encoded = json.dumps(mapping, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_allocations(
    approvals: Mapping[int, str | None] | Iterable[int] | None,
) -> dict[int, str | None]:
    if approvals is None:
        return {}
    if isinstance(approvals, Mapping):
        return dict(approvals)
    return {row_index: None for row_index in approvals}


def _validate_approval_rows(reuse: Mapping[int, str], allocations: Mapping[int, str | None], row_count: int) -> None:
    for row_index in (*reuse.keys(), *allocations.keys()):
        if not isinstance(row_index, int) or isinstance(row_index, bool) or not 0 <= row_index < row_count:
            raise ManifestError(f"approval references invalid CSV row index {row_index!r}")


def _allocate_identity(existing: Iterable[str], next_number: int, used: set[str]) -> tuple[str, int]:
    existing_ids = set(existing) | used
    candidate_number = next_number
    while True:
        candidate = f"csv-source-{candidate_number:06d}"
        candidate_number += 1
        if candidate not in existing_ids:
            return candidate, candidate_number


def _updated_manifest(
    base: CsvIdentityManifest,
    columns: tuple[str, ...],
    fingerprints: tuple[str, ...],
    assignments: Mapping[int, ManifestAssignment],
    approved_removals: set[str],
    next_source_number: int,
    current_digest: str,
    has_unresolved_changes: bool,
    structurally_stale: bool,
) -> CsvIdentityManifest:
    entries = {entry.source_identity: entry for entry in base.entries}
    for assignment in assignments.values():
        existing = entries.get(assignment.source_identity)
        if existing is None:
            entries[assignment.source_identity] = ManifestEntry(
                source_identity=assignment.source_identity,
                fingerprint=fingerprints[assignment.row_index],
                last_row_index=assignment.row_index,
                active=True,
            )
        else:
            entries[assignment.source_identity] = replace(
                existing,
                fingerprint=fingerprints[assignment.row_index],
                last_row_index=assignment.row_index,
                active=True,
            )
    for source_identity in approved_removals:
        if source_identity in entries:
            entries[source_identity] = replace(entries[source_identity], active=False, last_row_index=None)

    updated_entries = tuple(entries.values())
    if not has_unresolved_changes:
        digest = current_digest
        mapping_digest = _mapping_digest(updated_entries)
        output_columns = columns
        output_schema_version = MANIFEST_SCHEMA_VERSION
    else:
        digest = base.source_digest
        mapping_digest = base.mapping_digest
        output_columns = base.source_columns if structurally_stale else columns
        output_schema_version = base.schema_version
    return CsvIdentityManifest(
        source_columns=output_columns,
        entries=updated_entries,
        source_digest=digest,
        next_source_number=next_source_number,
        schema_version=output_schema_version,
        mapping_digest=mapping_digest,
    )


reconcile_manifest = reconcile_csv_manifest

__all__ = [
    "CsvIdentityManifest",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestAssignment",
    "ManifestEntry",
    "ManifestError",
    "ManifestReconciliation",
    "ManifestReviewItem",
    "reconcile_csv_manifest",
    "reconcile_manifest",
]
