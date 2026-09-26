"""Generate principal-part exercises from confirmed source records.

Generation consumes the parser's successful semantic output and the recipes in a
confirmed profile.  It does not infer eligibility, choose recipes from setup
suggestions, or render source text as trusted HTML.
"""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .html_text import source_html_to_text
from .identity import IdentityError, resolve_source_identity
from .notes import GeneratedNote, GeneratedNoteProvenance, ManagedNoteContent, RecipeMetadata
from .principal_parts import (
    ParsedPrincipalParts,
    PrincipalPartParseFailure,
    PrincipalPartValue,
    parse_principal_parts,
)
from .profile import DeckProfile, tag_character_violation
from .profile_setup import encode_unsafe_controls
from .sources import CanonicalSourceRecord

GenerationSkipStatus = Literal["incomplete", "unsupported", "ambiguous", "identity_error", "collision"]

_ROLE_LABELS = {
    "present_infinitive": "Infinitiv",
    "present_1s": "Präsens, 1. Person Singular",
    "perfect_1s": "Perfekt, 1. Person Singular",
    "perfect_passive_participle": "Partizip Perfekt Passiv (PPP)",
    "supine": "Supinum",
}


@dataclass(frozen=True, slots=True)
class GenerationSkip:
    """A source or exercise that was not safe to generate."""

    status: GenerationSkipStatus
    code: str
    message: str
    source_identity: str | None = None
    source_location: str | None = None
    recipe_identity: str | None = None


@dataclass(frozen=True, slots=True)
class PrincipalPartGenerationResult:
    """Typed generated notes and structured skips for one profile run."""

    notes: tuple[GeneratedNote, ...]
    skips: tuple[GenerationSkip, ...]

    @property
    def generated_count(self) -> int:
        return len(self.notes)

    @property
    def skipped_count(self) -> int:
        return len(self.skips)


def generate_principal_part_study_cards(
    records: Sequence[CanonicalSourceRecord],
    profile: DeckProfile,
    *,
    manifest_identities: Sequence[str | None] | None = None,
) -> PrincipalPartGenerationResult:
    """Generate one independent note for each selected recipe and parsed role.

    The profile's ``selected_recipes`` are the only recipe selection input.  A
    manifest identity sequence is required only for profiles using the manifest
    source-identity strategy.
    """

    if manifest_identities is not None and len(manifest_identities) != len(records):
        raise ValueError("manifest source identities must match the selected record count")

    selected_records = tuple(
        (index, record)
        for index, record in enumerate(records)
        if record.note_type is None or record.note_type == profile.note_type
    )
    skipped: list[GenerationSkip] = [
        GenerationSkip(
            status="unsupported",
            code="note_type_mismatch",
            message="The source record does not match the confirmed profile note type.",
            source_identity=record.source_identity,
            source_location=record.provenance.location,
        )
        for record in records
        if record.note_type is not None and record.note_type != profile.note_type
    ]

    resolved_records: list[tuple[CanonicalSourceRecord, str | None]] = []
    for original_index, record in selected_records:
        manifest_identity = None if manifest_identities is None else manifest_identities[original_index]
        try:
            source_identity = resolve_source_identity(record, profile, manifest_identity=manifest_identity)
        except IdentityError as error:
            source_identity = None
            skipped.append(
                GenerationSkip(
                    status="identity_error",
                    code="source_identity_unresolved",
                    message=str(error),
                    source_identity=record.source_identity,
                    source_location=record.provenance.location,
                )
            )
        resolved_records.append((record, source_identity))

    duplicate_identities = {
        identity
        for identity, count in Counter(identity for _, identity in resolved_records).items()
        if identity is not None and count > 1
    }
    notes: list[GeneratedNote] = []
    note_ids: set[str] = set()
    for record, source_identity in resolved_records:
        if source_identity is None:
            continue
        if source_identity in duplicate_identities:
            skipped.append(
                GenerationSkip(
                    status="collision",
                    code="duplicate_source_identity",
                    message="The source identity is assigned to more than one source record.",
                    source_identity=source_identity,
                    source_location=record.provenance.location,
                )
            )
            continue

        invalid_tag = _first_invalid_tag(record.source_tags)
        if invalid_tag is not None:
            position, reason = invalid_tag
            skipped.append(
                GenerationSkip(
                    status="unsupported",
                    code="invalid_source_tags",
                    message=(
                        f"The source note has an invalid tag at position {position} containing {reason}; "
                        "fix the tag in the source deck."
                    ),
                    source_identity=source_identity,
                    source_location=record.provenance.location,
                )
            )
            continue

        parsed = parse_principal_parts(record, profile)
        if isinstance(parsed, PrincipalPartParseFailure):
            skipped.append(_parse_failure_skip(parsed, record, source_identity))
            continue

        for recipe_identity in profile.selected_recipes:
            for part in parsed.value.parts:
                if part.is_omitted:
                    skipped.append(
                        GenerationSkip(
                            status="incomplete",
                            code="omitted_principal_part",
                            message="The confirmed source explicitly omits this principal-part role.",
                            source_identity=source_identity,
                            source_location=record.provenance.location,
                            recipe_identity=recipe_identity,
                        )
                    )
                    continue

                note = _render_note(
                    record,
                    parsed.value,
                    part,
                    source_identity=source_identity,
                    recipe_identity=recipe_identity,
                    profile=profile,
                )
                if note.latinitas_id in note_ids:
                    skipped.append(
                        GenerationSkip(
                            status="collision",
                            code="duplicate_logical_identity",
                            message="Two generated exercises resolved to the same logical identity.",
                            source_identity=source_identity,
                            source_location=record.provenance.location,
                            recipe_identity=recipe_identity,
                        )
                    )
                    continue
                note_ids.add(note.latinitas_id)
                notes.append(note)

    return PrincipalPartGenerationResult(notes=tuple(notes), skips=tuple(skipped))


def _parse_failure_skip(
    failure: PrincipalPartParseFailure,
    record: CanonicalSourceRecord,
    source_identity: str,
) -> GenerationSkip:
    return GenerationSkip(
        status=failure.status,
        code=failure.code,
        message=failure.message,
        source_identity=source_identity,
        source_location=record.provenance.location,
    )


def _first_invalid_tag(source_tags: tuple[str, ...]) -> tuple[int, str] | None:
    for position, tag in enumerate(source_tags, start=1):
        if tag_character_violation(tag) is not None:
            return position, "whitespace or control characters"
        if any(character in tag for character in "&<>"):
            return position, "the characters '&', '<', or '>' which cannot be exported safely"
    return None


def _combined_tags(record: CanonicalSourceRecord, profile: DeckProfile) -> tuple[str, ...]:
    """Combine inherited parent tags first with configured tags, without duplicates."""

    return tuple(dict.fromkeys((*record.source_tags, *profile.tags)))


def _render_note(
    record: CanonicalSourceRecord,
    parsed: ParsedPrincipalParts,
    part: PrincipalPartValue,
    *,
    source_identity: str,
    recipe_identity: str,
    profile: DeckProfile,
) -> GeneratedNote:
    meaning = _meaning(record, profile)
    tags = _combined_tags(record, profile)
    if recipe_identity == "principal_part_completion":
        content = _completion_content(parsed, part, meaning, tags=tags)
    else:
        content = _recognition_content(parsed, part, meaning, tags=tags)
    return GeneratedNote.create(
        source_identity=source_identity,
        provenance=GeneratedNoteProvenance(
            source_kind=record.source_kind,
            location=record.provenance.location,
            source_path=None,
            source_identity=source_identity,
        ),
        recipe=RecipeMetadata(recipe_identity=recipe_identity, exercise_key=part.identity_role),
        content=content,
    )


def _completion_content(
    parsed: ParsedPrincipalParts,
    missing: PrincipalPartValue,
    meaning: str,
    *,
    tags: tuple[str, ...],
) -> ManagedNoteContent:
    prompt = (
        "<div>Ergänze die fehlende Stammform.</div>"
        "<div><strong>Stammformen</strong></div>"
        f"<div>{_render_parts(parsed.parts, omitted_role=missing.role)}</div>"
        f"<div><strong>Bedeutung:</strong> {_escape_multiline(meaning)}</div>"
    )
    answer = (
        f"<div><strong>Fehlende Stammform:</strong> {_escape_multiline(missing.display or '')}</div>"
        f"<div><strong>Rolle:</strong> {_escape(_role_label(missing.role))}</div>"
    )
    return ManagedNoteContent(prompt=prompt, answer=answer, tags=tags)


def _recognition_content(
    parsed: ParsedPrincipalParts,
    supplied: PrincipalPartValue,
    meaning: str,
    *,
    tags: tuple[str, ...],
) -> ManagedNoteContent:
    prompt = f"Welche Stammform ist „{_escape_multiline(supplied.display or '')}“?"
    answer = (
        f"<div><strong>Lemma:</strong> {_escape_multiline(parsed.lexical_entry)}</div>"
        "<div><strong>Stammformen</strong></div>"
        f"<div>{_render_parts(parsed.parts)}</div>"
        f"<div><strong>Rolle:</strong> {_escape(_role_label(supplied.role))}</div>"
        f"<div><strong>Bedeutung:</strong> {_escape_multiline(meaning)}</div>"
    )
    return ManagedNoteContent(prompt=prompt, answer=answer, tags=tags)


def _render_parts(parts: Sequence[PrincipalPartValue], *, omitted_role: str | None = None) -> str:
    lines = []
    for part in parts:
        value = "_____" if part.role == omitted_role else (part.display if part.display is not None else "—")
        lines.append(f"<strong>{_escape(_role_label(part.role))}:</strong> {_escape_multiline(value)}")
    return "<br>".join(lines)


def _meaning(record: CanonicalSourceRecord, profile: DeckProfile) -> str:
    field = profile.fields.meaning_field
    return "" if field is None else record.fields.get(field, "").strip()


def _role_label(role: str) -> str:
    return _ROLE_LABELS.get(role, role)


def _escape(value: str) -> str:
    return _escape_text(source_html_to_text(value))


def _escape_multiline(value: str) -> str:
    return "<br>".join(_escape_text(line) for line in source_html_to_text(value).split("\n"))


def _escape_text(value: str) -> str:
    return html.escape(encode_unsafe_controls(value), quote=True)


generate_principal_part_notes = generate_principal_part_study_cards


__all__ = [
    "GenerationSkip",
    "GenerationSkipStatus",
    "PrincipalPartGenerationResult",
    "generate_principal_part_notes",
    "generate_principal_part_study_cards",
]
