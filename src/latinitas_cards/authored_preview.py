"""Read-only authored selection and card previews, independent of terminal output."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .authored_identity import IdentifiedAuthoredItem, reconcile_authored_import
from .authored_import import AuthoredImportError, AuthoredItem, ImportIssue, load_authored_import
from .authored_notes import render_authored_note
from .notes import GenerationMetadata


@dataclass(frozen=True)
class AuthoredFilters:
    """OR within each dimension, AND across dimensions; labels match exactly."""

    kinds: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def matches(self, item: AuthoredItem) -> bool:
        return (
            item.status == "include"
            and (not self.kinds or item.kind in self.kinds)
            and (not self.sections or item.provenance.section in self.sections)
            and (not self.references or item.provenance.reference in self.references)
            and (not self.tags or bool(set(self.tags).intersection(item.tags)))
        )


@dataclass(frozen=True)
class AuthoredCardPreview:
    latinitas_id: str
    kind: str
    key: str
    front: str
    back: str


@dataclass(frozen=True)
class AuthoredPreviewResult:
    filters: AuthoredFilters
    counts_by_kind: tuple[tuple[str, int], ...]
    counts_by_section: tuple[tuple[str, int], ...]
    counts_by_reference: tuple[tuple[str | None, int], ...]
    counts_by_status: tuple[tuple[str, int], ...]
    merged_duplicates: tuple[tuple[int, ...], ...]
    errors: tuple[ImportIssue, ...]
    diagnostic_notes: tuple[IdentifiedAuthoredItem, ...]
    selected_notes: tuple[IdentifiedAuthoredItem, ...]
    cards: tuple[AuthoredCardPreview, ...]

    def require_valid_selection(self) -> tuple[IdentifiedAuthoredItem, ...]:
        if self.errors:
            raise AuthoredImportError(self.errors)
        return self.selected_notes


def _render_template(template: str, fields: dict[str, str]) -> str:
    # Only the fixed authored templates are interpreted, never imported content.
    template = re.sub(
        r"\{\{#([^}]+)\}\}(.*?)\{\{/\1\}\}",
        lambda match: match[2] if fields.get(match[1]) else "",
        template,
        flags=re.DOTALL,
    )
    return re.sub(r"\{\{([^}]+)\}\}", lambda match: fields.get(match[1], ""), template)


def preview_authored_import(
    path: Path, namespace: str, filters: AuthoredFilters | None = None
) -> AuthoredPreviewResult:
    filters = filters if filters is not None else AuthoredFilters()
    reconciled = reconcile_authored_import(namespace, load_authored_import(path))
    notes = reconciled.diagnostic_notes
    matching = tuple(note for note in notes if filters.matches(note.item))
    cards = []
    # Authored previews need no corpus profile or principal-part field mapping.
    metadata = GenerationMetadata(profile_digest="authored-preview")
    # One representative card per selected kind, in reconciliation's stable order.
    seen: set[str] = set()
    for note in matching:
        if note.item.kind in seen:
            continue
        seen.add(note.item.kind)
        rendered = render_authored_note(note, metadata)
        fields = dict(rendered.managed_fields)
        front = _render_template(rendered.note_type.front_template, fields)
        back = _render_template(rendered.note_type.back_template, {**fields, "FrontSide": front})
        cards.append(AuthoredCardPreview(note.latinitas_id, note.item.kind, note.item.key, front, back))
    references = Counter(note.item.provenance.reference for note in notes)
    return AuthoredPreviewResult(
        filters=filters,
        counts_by_kind=tuple(sorted(Counter(note.item.kind for note in notes).items())),
        counts_by_section=tuple(sorted(Counter(note.item.provenance.section for note in notes).items())),
        counts_by_reference=tuple(sorted(references.items(), key=lambda pair: (pair[0] is not None, pair[0] or ""))),
        counts_by_status=tuple(sorted(Counter(note.item.status for note in notes).items())),
        merged_duplicates=reconciled.merged_duplicates,
        errors=reconciled.errors,
        diagnostic_notes=matching,
        selected_notes=() if reconciled.errors else matching,
        cards=tuple(cards),
    )
