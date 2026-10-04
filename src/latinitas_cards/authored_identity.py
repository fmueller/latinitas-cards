"""Stable identities and whole-file reconciliation for authored notes."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass

from .authored_import import AuthoredImportError, AuthoredImportResult, AuthoredItem, ImportIssue
from .identity import IdentityError, derive_latinitas_id


def normalize_authored_key(key: str) -> str:
    """NFC, strip outer whitespace, collapse Unicode whitespace; preserve case."""
    normalized = " ".join(unicodedata.normalize("NFC", key).split())
    if not normalized:
        raise IdentityError("authored key must be a non-empty string")
    return normalized


def derive_authored_id(namespace: str, item: AuthoredItem) -> str:
    """Use the v0.1.0 note contract with an authored-family source identity."""
    if not namespace.strip():
        raise IdentityError("collection namespace must be a non-empty string")
    source = json.dumps(["authored", namespace, item.kind], ensure_ascii=False, separators=(",", ":"))
    return derive_latinitas_id(source, normalize_authored_key(item.key))


@dataclass(frozen=True)
class IdentifiedAuthoredItem:
    latinitas_id: str
    item: AuthoredItem
    lines: tuple[int, ...]


@dataclass(frozen=True)
class AuthoredIdentityResult:
    """Diagnostic notes must not be selected/exported until require_valid passes."""

    diagnostic_notes: tuple[IdentifiedAuthoredItem, ...]
    errors: tuple[ImportIssue, ...]

    @property
    def merged_duplicates(self) -> tuple[tuple[int, ...], ...]:
        return tuple(note.lines for note in self.diagnostic_notes if len(note.lines) > 1)

    def require_valid(self) -> tuple[IdentifiedAuthoredItem, ...]:
        if self.errors:
            raise AuthoredImportError(self.errors)
        return self.diagnostic_notes


def reconcile_authored_import(namespace: str, loaded: AuthoredImportResult) -> AuthoredIdentityResult:
    """Reconcile the whole input before filtering, retaining loader diagnostics."""
    if not namespace.strip():
        raise IdentityError("collection namespace must be a non-empty string")
    groups: dict[tuple[str, str], list[AuthoredItem]] = {}
    for item in loaded.diagnostic_items:
        groups.setdefault((item.kind, normalize_authored_key(item.key)), []).append(item)
    notes: list[IdentifiedAuthoredItem] = []
    errors = list(loaded.errors)
    for (_, key), items in sorted(groups.items()):
        items.sort(key=lambda item: item.line_number)
        first = items[0]
        # Keep the actual contributor's line for optional-field conflict diagnostics.
        values: dict[str, tuple[str | None, int]] = {}
        conflicts: list[ImportIssue] = []
        for item in items:
            fields: dict[str, str | None] = {
                name: getattr(item, name)
                for name in type(item).model_fields
                if name not in {"key", "kind", "schema_version", "line_number", "tags", "status", "provenance"}
            }
            fields.update(
                {f"provenance.{name}": getattr(item.provenance, name) for name in type(item.provenance).model_fields}
            )
            for name, value in fields.items():
                # Only optional fields can be empty in a validated typed item.
                if value is None or value == "":
                    continue
                previous = values.get(name)
                if previous is not None and previous[0] != value:
                    conflicts.append(ImportIssue(item.line_number, name, f"conflicts with line {previous[1]}"))
                else:
                    values.setdefault(name, (value, item.line_number))
        if conflicts:
            errors.extend(conflicts)
            continue
        updates: dict[str, object] = {
            name: values[name][0] if name in values else None
            for name in type(first).model_fields
            if name not in {"key", "kind", "schema_version", "line_number", "tags", "status", "provenance"}
        }
        updates.update(
            key=key,
            tags=tuple(sorted({tag for item in items for tag in item.tags})),
            status="skip" if any(item.status == "skip" for item in items) else "include",
            provenance=first.provenance.model_copy(
                update={"reference": values.get("provenance.reference", (None, 0))[0]}
            ),
        )
        merged = first.model_copy(update=updates)
        notes.append(
            IdentifiedAuthoredItem(
                derive_authored_id(namespace, merged), merged, tuple(item.line_number for item in items)
            )
        )
    return AuthoredIdentityResult(tuple(notes), tuple(errors))
