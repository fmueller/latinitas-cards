"""Evidence-based assisted setup for reusable deck profiles.

This module deliberately stops at profile setup.  It inspects immutable canonical
source records, proposes a profile, and reports structural evidence without
claiming that a syntactic principal-part match is a confirmed verb.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .principal_parts import PrincipalPartParseSuccess, parse_principal_parts
from .profile import (
    DEFAULT_GENERATED_NOTE_TYPE,
    DEFAULT_LANGUAGE_TAG,
    DEFAULT_PRINCIPAL_PART_ROLES,
    DEFAULT_SELECTED_RECIPES,
    DEFAULT_TAGS,
    DEFAULT_TARGET_DECK,
    DeckProfile,
    ProfileOverrides,
    SourceIdentityConfig,
    resolve_profile,
)
from .sources import CanonicalSourceRecord, SourceInspection, inspect_source

ASSISTED_PRINCIPAL_PART_ROLES = (
    "present_infinitive",
    "present_1s",
    "perfect_1s",
    "perfect_passive_participle",
)
ASSISTED_THREE_ROLE_ORDER = ("present_1s", "present_infinitive", "perfect_1s")
_CANDIDATE_SEPARATORS = (",", " — ", ";", " / ", "|", "\n")
_ROLE_NAME_HINTS = {
    3: ASSISTED_THREE_ROLE_ORDER,
    4: ASSISTED_PRINCIPAL_PART_ROLES,
}
_LEXICAL_NAME_HINTS = ("entry", "lemma", "lexical", "latin", "latein", "word", "front")
_PRINCIPAL_NAME_HINTS = ("principal", "part", "form", "construction", "hint", "stamm")
_MEANING_NAME_HINTS = ("meaning", "gloss", "german", "deutsch", "translation", "definition", "bedeutung")
_IDENTITY_NAME_HINTS = ("id", "guid", "identity", "sourceid", "stableid", "noteid")

SetupStatus = Literal["success", "incomplete", "unsupported", "ambiguous"]


class ProfileSetupError(ValueError):
    """Raised when source inspection cannot produce a safe setup proposal."""


@dataclass(frozen=True, slots=True)
class FieldCandidate:
    """One named source field and the evidence supporting its proposed role."""

    field: str
    role: str
    score: int
    evidence: tuple[str, ...]

    def to_machine_readable(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "role": self.role,
            "score": self.score,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class RepresentativeExample:
    """A sanitized-by-boundary representative source value and structural result."""

    source_location: str
    lexical_entry: str
    principal_parts: str
    meaning: str | None
    structural_status: SetupStatus
    structural_message: str
    uncertainty: str

    def to_machine_readable(self) -> dict[str, Any]:
        return {
            "source_location": self.source_location,
            "lexical_entry": self.lexical_entry,
            "principal_parts": self.principal_parts,
            "meaning": self.meaning,
            "structural_status": self.structural_status,
            "structural_message": self.structural_message,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class ProfileSetupProposal:
    """A deterministic profile proposal plus evidence shown before confirmation."""

    source_kind: str
    record_count: int
    note_types: tuple[str, ...]
    fields: tuple[str, ...]
    profile: DeckProfile
    field_candidates: tuple[FieldCandidate, ...]
    examples: tuple[RepresentativeExample, ...]
    uncertainties: tuple[str, ...]
    recipe_suggestions: tuple[str, ...]

    def with_profile(self, profile: DeckProfile) -> ProfileSetupProposal:
        records = _records_for_profile(self._records_for_examples, profile)
        fields = tuple(sorted({field for record in records for field in record.fields}))
        return replace(
            self,
            record_count=len(records),
            fields=fields,
            profile=profile,
            field_candidates=_field_candidates(
                fields,
                profile.fields.lexical_entry_field,
                profile.fields.principal_parts_field,
                profile.fields.meaning_field,
            ),
            examples=build_representative_examples(self._records_for_examples, profile),
        )

    # The proposal keeps records private to avoid making source rows part of the
    # serialized profile contract.  This attribute is attached by the factory.
    _records_for_examples: tuple[CanonicalSourceRecord, ...] = ()

    def to_machine_readable(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "record_count": self.record_count,
            "note_types": list(self.note_types),
            "fields": list(self.fields),
            "profile": self.profile.to_machine_readable(),
            "field_candidates": [candidate.to_machine_readable() for candidate in self.field_candidates],
            "examples": [example.to_machine_readable() for example in self.examples],
            "uncertainties": list(self.uncertainties),
            "recipe_suggestions": list(self.recipe_suggestions),
        }


def propose_profile(source_path: str | Path, *, note_type: str | None = None) -> ProfileSetupProposal:
    """Inspect a source and return a deterministic, reviewable profile proposal."""

    path = Path(source_path)
    inspection = inspect_source(path)
    return propose_profile_from_inspection(inspection, path, note_type=note_type)


def propose_profile_from_inspection(
    inspection: SourceInspection,
    source_path: str | Path,
    *,
    note_type: str | None = None,
) -> ProfileSetupProposal:
    """Build a proposal from already-inspected canonical records."""

    if not inspection.records:
        raise ProfileSetupError("The source contains no records to inspect.")

    source_kind = inspection.records[0].source_kind
    selected_note_type = _select_note_type(inspection.note_types, note_type, source_kind)
    records = tuple(
        record
        for record in inspection.records
        if selected_note_type == "CSV source" or record.note_type == selected_note_type
    )
    if not records:
        raise ProfileSetupError("The selected note type contains no source records.")

    fields = tuple(sorted({field for record in records for field in record.fields}))
    if len(fields) < 2:
        raise ProfileSetupError("The source must expose at least two named fields for profile setup.")

    principal_field, separator, role_count, principal_score = _choose_principal_field(records, fields)
    lexical_field = _choose_lexical_field(records, fields, principal_field)
    meaning_field = _choose_meaning_field(records, fields, lexical_field, principal_field)
    source_identity = _propose_source_identity(source_kind, records, fields)

    role_order = _ROLE_NAME_HINTS.get(role_count, DEFAULT_PRINCIPAL_PART_ROLES)
    if len(role_order) != role_count:
        role_order = tuple(f"role_{index}" for index in range(1, role_count + 1))
    profile = DeckProfile.default(
        note_type=selected_note_type,
        lexical_entry_field=lexical_field,
        principal_parts_field=principal_field,
        meaning_field=meaning_field,
        source_identity=source_identity,
        principal_part_roles=role_order,
        separators=(separator,),
        language_tag=DEFAULT_LANGUAGE_TAG,
        generated_note_type=DEFAULT_GENERATED_NOTE_TYPE,
        target_deck=DEFAULT_TARGET_DECK,
        tags=DEFAULT_TAGS,
        selected_recipes=DEFAULT_SELECTED_RECIPES,
    )

    field_candidates = _field_candidates(fields, lexical_field, principal_field, meaning_field)
    examples = build_representative_examples(records, profile)
    uncertainties = _uncertainties(
        inspection,
        source_identity=source_identity,
        principal_score=principal_score,
        examples=examples,
        meaning_field=meaning_field,
    )
    return ProfileSetupProposal(
        source_kind=source_kind,
        record_count=len(records),
        note_types=inspection.note_types,
        fields=fields,
        profile=profile,
        field_candidates=field_candidates,
        examples=examples,
        uncertainties=uncertainties,
        recipe_suggestions=DEFAULT_SELECTED_RECIPES,
        _records_for_examples=inspection.records,
    )


def apply_profile_overrides(
    proposal: ProfileSetupProposal,
    overrides: ProfileOverrides | Mapping[str, Any] | None = None,
) -> ProfileSetupProposal:
    """Return a corrected proposal without mutating source records or the base proposal."""

    if overrides is None:
        return proposal
    return proposal.with_profile(resolve_profile(proposal.profile, overrides))


def build_representative_examples(
    records: Sequence[CanonicalSourceRecord],
    profile: DeckProfile,
    *,
    limit: int = 5,
) -> tuple[RepresentativeExample, ...]:
    """Parse a bounded sample and label structural uncertainty conservatively."""

    selected = _records_for_profile(records, profile)
    examples: list[RepresentativeExample] = []
    for record in selected[:limit]:
        lexical = safe_source_value(
            record.fields.get(profile.fields.lexical_entry_field, ""),
            profile.fields.lexical_entry_field,
        )
        principal_parts = safe_source_value(
            record.fields.get(profile.fields.principal_parts_field, ""),
            profile.fields.principal_parts_field,
        )
        meaning = None
        if profile.fields.meaning_field is not None:
            meaning = safe_source_value(
                record.fields.get(profile.fields.meaning_field, ""),
                profile.fields.meaning_field,
            )
        result = parse_principal_parts(record, profile)
        if isinstance(result, PrincipalPartParseSuccess):
            status: SetupStatus = "success"
            message = "Structural layout matched the proposed role order."
        else:
            status = result.status
            message = result.message
        examples.append(
            RepresentativeExample(
                source_location=record.provenance.location,
                lexical_entry=lexical,
                principal_parts=principal_parts,
                meaning=meaning,
                structural_status=status,
                structural_message=message,
                uncertainty=(
                    "A structural parse does not confirm verb eligibility or the semantic identity of the PPP."
                ),
            )
        )
    return tuple(examples)


def profile_source_issues(
    profile: DeckProfile,
    inspection: SourceInspection,
) -> tuple[str, ...]:
    """Return safe compatibility diagnostics for reusing a saved profile."""

    records = tuple(
        record for record in inspection.records if record.note_type is None or record.note_type == profile.note_type
    )
    issues: list[str] = []
    if not records:
        issues.append(f"note type '{profile.note_type}' is not present in the inspected source")
        return tuple(issues)

    available_fields = {field for record in records for field in record.fields}
    for field_name, label in (
        (profile.fields.lexical_entry_field, "lexical-entry"),
        (profile.fields.principal_parts_field, "principal-parts"),
    ):
        if field_name not in available_fields:
            issues.append(f"configured {label} field is missing from the source")
    if profile.fields.meaning_field is not None and profile.fields.meaning_field not in available_fields:
        issues.append("configured optional meaning field is missing from the source")

    if profile.source_identity.strategy == "note_guid":
        if any(record.source_kind not in {"apkg", "colpkg"} for record in records):
            issues.append("note_guid source identity is only valid for Anki package input")
    elif profile.source_identity.strategy == "source_id_field":
        field = profile.source_identity.field
        if field is None or field not in available_fields:
            issues.append("configured source-ID field is missing from the source")
        elif any(record.source_kind == "csv" for record in records):
            source_ids = tuple(record.fields.get(field, "") for record in records)
            if any(not source_id.strip() for source_id in source_ids):
                issues.append("configured CSV source-ID field contains blank values")
            if len(set(source_ids)) != len(source_ids):
                issues.append("configured CSV source-ID field must contain unique values")
    return tuple(issues)


def _records_for_profile(
    records: Sequence[CanonicalSourceRecord],
    profile: DeckProfile,
) -> tuple[CanonicalSourceRecord, ...]:
    return tuple(record for record in records if record.note_type is None or record.note_type == profile.note_type)


_SENSITIVE_FIELD_PARTS = (
    "accesskey",
    "apikey",
    "credential",
    "password",
    "privatekey",
    "secret",
    "token",
)
_BIDI_CONTROL_CODES = {
    0x061C,
    0x200E,
    0x200F,
    0x202A,
    0x202B,
    0x202C,
    0x202D,
    0x202E,
    0x2066,
    0x2067,
    0x2068,
    0x2069,
}


def safe_source_value(value: str, field_name: str, *, limit: int = 160) -> str:
    """Bound and escape source values before displaying representative examples."""

    normalised_field = re.sub(r"[^a-z0-9]+", "", field_name.lower())
    if any(part in normalised_field for part in _SENSITIVE_FIELD_PARTS):
        return "[redacted sensitive source field]"
    escaped = "".join(
        "\\n"
        if character == "\n"
        else "\\r"
        if character == "\r"
        else "\\t"
        if character == "\t"
        else f"\\u{ord(character):04x}"
        if ord(character) in _BIDI_CONTROL_CODES
        else f"\\x{ord(character):02x}"
        if ord(character) < 32 or ord(character) == 127
        else character
        for character in value
    )
    return escaped if len(escaped) <= limit else escaped[: limit - 1] + "…"


def _select_note_type(note_types: Sequence[str], requested: str | None, source_kind: str) -> str:
    if requested is not None:
        if note_types and requested not in note_types:
            raise ProfileSetupError("The requested note type is not present in the source.")
        return requested
    if note_types:
        return note_types[0]
    if source_kind == "csv":
        return "CSV source"
    return "Source note type"


def _choose_principal_field(
    records: Sequence[CanonicalSourceRecord],
    fields: Sequence[str],
) -> tuple[str, str, int, int]:
    best: tuple[int, int, int, str, str] | None = None
    for field in fields:
        values = tuple(record.fields.get(field, "") for record in records)
        name_score = _name_score(field, _PRINCIPAL_NAME_HINTS)
        for separator_index, separator in enumerate(_CANDIDATE_SEPARATORS):
            exact_count = sum(1 for value in values if value.strip() and value.count(separator) == 3)
            three_count = sum(1 for value in values if value.strip() and value.count(separator) == 2)
            score = exact_count * 100 + three_count * 15 + name_score * 4
            candidate = (score, exact_count, -separator_index, field, separator)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise ProfileSetupError("The source has no candidate principal-parts field.")

    _score, exact_count, _separator_order, field, separator = best
    role_count = 4 if exact_count else 3
    if not exact_count:
        role_count = 3 if any(record.fields.get(field, "").count(separator) == 2 for record in records) else 4
    return field, separator, role_count, best[0]


def _choose_lexical_field(
    records: Sequence[CanonicalSourceRecord],
    fields: Sequence[str],
    principal_field: str,
) -> str:
    candidates: list[tuple[int, str]] = []
    for field in fields:
        if field == principal_field:
            continue
        values = tuple(record.fields.get(field, "") for record in records)
        non_empty = sum(bool(value.strip()) for value in values)
        no_separators = sum(
            bool(value.strip()) and not any(separator in value for separator in _CANDIDATE_SEPARATORS[:-1])
            for value in values
        )
        score = _name_score(field, _LEXICAL_NAME_HINTS) * 20 + non_empty * 2 + no_separators
        candidates.append((score, field))
    if not candidates:
        raise ProfileSetupError("The source has no candidate lexical-entry field.")
    return max(candidates)[1]


def _choose_meaning_field(
    records: Sequence[CanonicalSourceRecord],
    fields: Sequence[str],
    lexical_field: str,
    principal_field: str,
) -> str | None:
    candidates: list[tuple[int, str]] = []
    for field in fields:
        if field in {lexical_field, principal_field} or _looks_like_identity_field(field):
            continue
        values = tuple(record.fields.get(field, "") for record in records)
        non_empty = sum(bool(value.strip()) for value in values)
        score = _name_score(field, _MEANING_NAME_HINTS) * 20 + non_empty
        if score:
            candidates.append((score, field))
    return max(candidates)[1] if candidates else None


def _propose_source_identity(
    source_kind: str,
    records: Sequence[CanonicalSourceRecord],
    fields: Sequence[str],
) -> SourceIdentityConfig:
    if source_kind in {"apkg", "colpkg"}:
        return SourceIdentityConfig(strategy="note_guid")
    candidates = [
        field
        for field in fields
        if _looks_like_identity_field(field)
        and all(record.fields.get(field, "").strip() for record in records)
        and len({record.fields.get(field, "") for record in records}) == len(records)
    ]
    if candidates:
        return SourceIdentityConfig(strategy="source_id_field", field=sorted(candidates)[0])
    return SourceIdentityConfig(strategy="manifest")


def _field_candidates(
    fields: Sequence[str],
    lexical_field: str,
    principal_field: str,
    meaning_field: str | None,
) -> tuple[FieldCandidate, ...]:
    candidates: list[FieldCandidate] = []
    for field in fields:
        if field == lexical_field:
            role = "lexical_entry"
            score = _name_score(field, _LEXICAL_NAME_HINTS)
            evidence = ("name and observed values fit a lexical-entry field",)
        elif field == principal_field:
            role = "principal_parts"
            score = _name_score(field, _PRINCIPAL_NAME_HINTS)
            evidence = ("observed values contain repeated candidate role separators",)
        elif field == meaning_field:
            role = "meaning"
            score = _name_score(field, _MEANING_NAME_HINTS)
            evidence = ("field name and non-empty text fit an optional meaning/gloss field",)
        else:
            role = "unmapped"
            score = 0
            evidence = ("retained as an unmapped source field",)
        candidates.append(FieldCandidate(field=field, role=role, score=score, evidence=evidence))
    return tuple(candidates)


def _uncertainties(
    inspection: SourceInspection,
    *,
    source_identity: SourceIdentityConfig,
    principal_score: int,
    examples: Sequence[RepresentativeExample],
    meaning_field: str | None,
) -> tuple[str, ...]:
    uncertainties = [
        "Field mappings and role names are proposals based on source structure; confirm them before saving.",
        "Structural principal-part matches do not confirm verb eligibility or the semantic identity of the PPP.",
        "Recipe suggestions are compatible choices only; no exercise is generated during setup.",
    ]
    if len(inspection.note_types) > 1:
        uncertainties.append("Multiple note types are present; the selected note type is only a proposal.")
    if source_identity.strategy == "manifest":
        uncertainties.append(
            "CSV records have no confirmed stable ID field; manifest identity review remains required."
        )
    if meaning_field is None:
        uncertainties.append("No optional meaning/gloss field was identified.")
    if principal_score <= 0:
        uncertainties.append(
            "No repeated four-slot structural pattern was observed in the proposed principal-parts field."
        )
    if any(example.structural_status != "success" for example in examples):
        uncertainties.append("Some representative values do not match the proposal and remain incomplete or ambiguous.")
    return tuple(uncertainties)


def _name_score(field: str, hints: Sequence[str]) -> int:
    normalised = re.sub(r"[^a-z0-9]+", "", field.lower())
    return sum(1 for hint in hints if hint in normalised)


def _looks_like_identity_field(field: str) -> bool:
    normalised = re.sub(r"[^a-z0-9]+", "", field.lower())
    return normalised in _IDENTITY_NAME_HINTS or any(normalised.endswith(suffix) for suffix in ("id", "guid"))


__all__ = [
    "ASSISTED_PRINCIPAL_PART_ROLES",
    "FieldCandidate",
    "ProfileSetupError",
    "ProfileSetupProposal",
    "RepresentativeExample",
    "apply_profile_overrides",
    "build_representative_examples",
    "profile_source_issues",
    "propose_profile",
    "propose_profile_from_inspection",
    "safe_source_value",
]
