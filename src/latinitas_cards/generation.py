"""Generate coherent learning-object notes from confirmed source records.

Generation consumes the parser's successful semantic output and the confirmed
profile.  Each eligible source note becomes exactly one learning-object note
for its single confirmed lexeme; multi-object or ambiguous rows are reported
for review instead of being merged or split silently.  Card semantic keys are
derived per eligible recipe and role but never participate in note identity,
and no recipe or wording is part of the identifier.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from .cards import render_cards, role_display_label
from .html_text import source_html_to_text
from .identity import (
    IdentityError,
    ResolvedSourceIdentity,
    resolve_source_identity,
)
from .notes import (
    GeneratedNote,
    GeneratedNoteProvenance,
    GenerationMetadata,
    ManagedNoteContent,
)
from .principal_parts import (
    ParsedPrincipalParts,
    PrincipalPartParseFailure,
    PrincipalPartValue,
    normalize_principal_part_for_comparison,
    parse_principal_parts,
)
from .profile import DeckProfile, tag_character_violation
from .profile_setup import encode_unsafe_controls
from .source_extraction import SourceExtraction
from .sources import CanonicalSourceRecord

GenerationSkipStatus = Literal["incomplete", "unsupported", "ambiguous", "identity_error", "collision"]

SINGLE_LEXEME_OBJECT_KEY = "lexeme-1"
_MULTI_OBJECT_SPLIT = re.compile(r"[,;\n\t|]| / | — | - ")


@dataclass(frozen=True, slots=True)
class GenerationSkip:
    """A source or object that was not safe to generate."""

    status: GenerationSkipStatus
    code: str
    message: str
    source_identity: str | None = None
    source_location: str | None = None
    evidence: SourceExtraction | None = None


@dataclass(frozen=True, slots=True)
class LearningObjectGenerationResult:
    """Typed generated notes and structured skips for one profile run."""

    notes: tuple[GeneratedNote, ...]
    skips: tuple[GenerationSkip, ...]
    source_entry_count: int | None = None

    @property
    def generated_count(self) -> int:
        return len(self.notes)

    @property
    def skipped_count(self) -> int:
        if self.source_entry_count is not None:
            return self.source_entry_count - self.generated_count
        return sum(skip.code not in {"omitted_principal_part", "unresolved_source_evidence"} for skip in self.skips)

    @property
    def generated_warning_count(self) -> int:
        generated = {note.provenance.source_identity for note in self.notes}
        return len({skip.source_identity for skip in self.skips if skip.source_identity in generated})


def profile_digest(profile: DeckProfile) -> str:
    """Return the deterministic digest of one effective profile for provenance."""

    canonical = json.dumps(
        profile.to_machine_readable(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"profile-sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def generate_learning_object_notes(
    records: Sequence[CanonicalSourceRecord],
    profile: DeckProfile,
    *,
    manifest_identities: Sequence[str | None] | None = None,
    source_scope: str | None = None,
) -> LearningObjectGenerationResult:
    """Generate one coherent learning-object note per eligible source record.

    The profile's ``selected_recipes`` enable card semantic keys on each note;
    they never change which note exists.  A manifest identity sequence is
    required only for profiles using the manifest source-identity strategy.
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

    resolved_records: list[tuple[CanonicalSourceRecord, ResolvedSourceIdentity | None]] = []
    for original_index, record in selected_records:
        manifest_identity = None if manifest_identities is None else manifest_identities[original_index]
        try:
            source_identity = resolve_source_identity(
                record,
                profile,
                manifest_identity=manifest_identity,
                source_scope=source_scope,
            )
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
    metadata = GenerationMetadata(profile_digest=profile_digest(profile))
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
                    source_identity=source_identity.value,
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
                    source_identity=source_identity.value,
                    source_location=record.provenance.location,
                )
            )
            continue

        parsed = parse_principal_parts(record, profile)
        if isinstance(parsed, PrincipalPartParseFailure):
            skipped.append(_parse_failure_skip(parsed, record, source_identity.value))
            continue

        if _names_multiple_lexemes(parsed.value.lexical_entry):
            skipped.append(
                GenerationSkip(
                    status="ambiguous",
                    code="multi_object_source",
                    message=(
                        "The lexical entry names more than one distinct lexeme; confirm one coherent "
                        "learning object or review explicit object boundaries. Spelling variants that "
                        "normalize to one comparison form remain one object."
                    ),
                    source_identity=source_identity.value,
                    source_location=record.provenance.location,
                )
            )
            continue

        omitted_roles = tuple(part.role for part in parsed.value.parts if part.is_omitted)
        if omitted_roles:
            skipped.append(
                GenerationSkip(
                    status="incomplete",
                    code="omitted_principal_part",
                    message=(
                        "The confirmed source explicitly omits these principal-part roles: "
                        + ", ".join(omitted_roles)
                        + "."
                    ),
                    source_identity=source_identity.value,
                    source_location=record.provenance.location,
                    evidence=parsed.value.evidence,
                )
            )

        unresolved_roles = tuple(part.role for part in parsed.value.parts if part.unresolved)
        if unresolved_roles:
            skipped.append(
                GenerationSkip(
                    status="ambiguous",
                    code="unresolved_source_evidence",
                    message="Extraction matched, but linguistic/selection uncertainty withholds targets for: "
                    + ", ".join(unresolved_roles)
                    + ". Review alternatives/hints; no answer was selected.",
                    source_identity=source_identity.value,
                    source_location=record.provenance.location,
                    evidence=parsed.value.evidence,
                )
            )

        note = _render_note(
            record,
            parsed.value,
            source_identity,
            profile=profile,
            metadata=metadata,
        )
        if note.latinitas_id in note_ids:
            skipped.append(
                GenerationSkip(
                    status="collision",
                    code="duplicate_logical_identity",
                    message="Two learning objects resolved to the same logical identity.",
                    source_identity=source_identity.value,
                    source_location=record.provenance.location,
                )
            )
            continue
        note_ids.add(note.latinitas_id)
        notes.append(note)

    return LearningObjectGenerationResult(notes=tuple(notes), skips=tuple(skipped), source_entry_count=len(records))


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
        evidence=failure.evidence,
    )


def _names_multiple_lexemes(lemma_text: str) -> bool:
    """Report whether one lexical entry names more than one distinct lexeme.

    Spelling variants of the same lexeme (for example macron and plain forms)
    normalize to one comparison form and stay one coherent object; genuinely
    distinct lemmas in one entry remain a review item instead of being merged.
    """

    segments = [segment.strip() for segment in _MULTI_OBJECT_SPLIT.split(lemma_text) if segment.strip()]
    comparisons = {normalize_principal_part_for_comparison(segment) for segment in segments}
    return len(comparisons) > 1


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
    source_identity: ResolvedSourceIdentity,
    *,
    profile: DeckProfile,
    metadata: GenerationMetadata,
) -> GeneratedNote:
    meaning = _meaning(record, profile)
    meaning_text = source_html_to_text(meaning)
    tags = _combined_tags(record, profile)
    content = ManagedNoteContent(
        lemma=_escape_normalized_lines(parsed.lexical_entry),
        principal_parts=(
            '<span hidden class="source-extraction">'
            + _escape_text(json.dumps(asdict(parsed.evidence), ensure_ascii=False))
            + "</span>"
            if parsed.evidence is not None
            else ""
        )
        + _render_parts(parsed.parts),
        meaning=_escape_normalized_lines(meaning_text),
        tags=tags,
    )
    cards = render_cards(parsed, selected_recipes=profile.selected_recipes, meaning=meaning_text)
    return GeneratedNote.create(
        source_identity=source_identity.value,
        source_scope=source_identity.scope,
        provenance=GeneratedNoteProvenance(
            source_kind=record.source_kind,
            location=record.provenance.location,
            source_scope=source_identity.scope,
        ),
        object_key=SINGLE_LEXEME_OBJECT_KEY,
        metadata=metadata,
        content=content,
        cards=cards,
    )


def _render_parts(parts: Sequence[PrincipalPartValue]) -> str:
    lines = []
    for part in parts:
        value = "—" if part.is_omitted else _escape_normalized_lines(part.display or "")
        if part.unresolved:
            value += " <em>(unresolved source evidence; target withheld)</em>"
        lines.append(f"<strong>{_escape_text(role_display_label(part.role))}:</strong> {value}")
    return "<br>".join(lines)


def _meaning(record: CanonicalSourceRecord, profile: DeckProfile) -> str:
    field = profile.fields.meaning_field
    return "" if field is None else record.fields.get(field, "").strip()


def _escape_normalized_lines(value: str) -> str:
    return "<br>".join(_escape_text(line) for line in value.split("\n"))


def _escape_text(value: str) -> str:
    return html.escape(encode_unsafe_controls(value), quote=True)


__all__ = [
    "GenerationSkip",
    "GenerationSkipStatus",
    "LearningObjectGenerationResult",
    "SINGLE_LEXEME_OBJECT_KEY",
    "generate_learning_object_notes",
    "profile_digest",
]
