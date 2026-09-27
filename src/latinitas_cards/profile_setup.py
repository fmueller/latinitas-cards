"""Evidence-based assisted setup for reusable deck profiles.

This module deliberately stops at profile setup.  It inspects immutable canonical
source records, proposes a profile, and reports structural evidence without
claiming that a syntactic principal-part match is a confirmed verb.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
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
    FieldOverrides,
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
_LEXICAL_OVERLAP_WEIGHT = 50
_LEXICAL_WORD_WEIGHT = 5
_LEXICAL_NAME_WEIGHT = 5
_LEXICAL_SAMPLE_LIMIT = 2
_SINGLE_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_NUMERIC_CODE_RE = re.compile(r"\d+(?:[.,/\-–—]\d+)*")

SetupStatus = Literal["success", "incomplete", "unsupported", "ambiguous"]
FieldChoiceRole = Literal["lexical_entry", "principal_parts"]


class ProfileSetupError(ValueError):
    """Raised when source inspection cannot produce a safe setup proposal."""


@dataclass(frozen=True, slots=True)
class FieldCandidate:
    """One named source field, sanitized sample values, and the evidence for its role."""

    field: str
    role: str
    score: int
    evidence: tuple[str, ...]
    sample_values: tuple[str, ...] = ()

    def to_machine_readable(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "role": self.role,
            "score": self.score,
            "evidence": list(self.evidence),
            "sample_values": list(self.sample_values),
        }


@dataclass(frozen=True, slots=True)
class RequiredFieldChoice:
    """One field role whose evidence is too weak or tied to save without an explicit choice."""

    role: FieldChoiceRole
    reason: str
    candidates: tuple[str, ...]

    def to_machine_readable(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "reason": self.reason,
            "candidates": list(self.candidates),
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
    field_choices_required: tuple[RequiredFieldChoice, ...] = ()

    # The proposal keeps records private to avoid making source rows part of the
    # serialized profile contract.  This attribute is attached by the factory.
    _records_for_examples: tuple[CanonicalSourceRecord, ...] = ()

    def with_profile(
        self,
        profile: DeckProfile,
        *,
        explicit_fields: frozenset[str] = frozenset(),
    ) -> ProfileSetupProposal:
        records = _records_for_profile(self._records_for_examples, profile)
        fields = tuple(sorted({field for record in records for field in record.fields}))
        separator = profile.principal_parts.separators[0]
        principal_evidences = _principal_field_evidences(records, fields)
        lexical_evidences = _lexical_field_evidences(records, fields, profile.fields.principal_parts_field, separator)
        return replace(
            self,
            record_count=len(records),
            fields=fields,
            profile=profile,
            field_candidates=_field_candidates(
                records,
                fields,
                lexical_evidences=lexical_evidences,
                principal_evidences=principal_evidences,
                lexical_field=profile.fields.lexical_entry_field,
                principal_field=profile.fields.principal_parts_field,
                meaning_field=profile.fields.meaning_field,
                separator=separator,
            ),
            field_choices_required=_field_choice_requirements(
                lexical_evidences,
                principal_evidences,
                lexical_field=profile.fields.lexical_entry_field,
                principal_field=profile.fields.principal_parts_field,
                explicit_fields=explicit_fields,
            ),
            examples=build_representative_examples(self._records_for_examples, profile),
        )

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
            "field_choices_required": [choice.to_machine_readable() for choice in self.field_choices_required],
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

    principal_evidences = _principal_field_evidences(records, fields)
    principal_field, separator, role_count, principal_score = _choose_principal_field(principal_evidences)
    lexical_evidences = _lexical_field_evidences(records, fields, principal_field, separator)
    lexical_field = _choose_lexical_field(lexical_evidences)
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

    field_candidates = _field_candidates(
        records,
        fields,
        lexical_evidences=lexical_evidences,
        principal_evidences=principal_evidences,
        lexical_field=lexical_field,
        principal_field=principal_field,
        meaning_field=meaning_field,
        separator=separator,
    )
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
        field_choices_required=_field_choice_requirements(
            lexical_evidences,
            principal_evidences,
            lexical_field=lexical_field,
            principal_field=principal_field,
            explicit_fields=frozenset(),
        ),
        _records_for_examples=inspection.records,
    )


def apply_profile_overrides(
    proposal: ProfileSetupProposal,
    overrides: ProfileOverrides | Mapping[str, Any] | None = None,
) -> ProfileSetupProposal:
    """Return a corrected proposal without mutating source records or the base proposal."""

    if overrides is None:
        return proposal
    parsed = overrides if isinstance(overrides, ProfileOverrides) else ProfileOverrides.from_mapping(overrides)
    explicit_fields = _explicit_field_choices(parsed)
    meaning_field = proposal.profile.fields.meaning_field
    if meaning_field is not None and meaning_field in explicit_fields and not _meaning_field_explicitly_set(parsed):
        parsed = _with_meaning_field_cleared(parsed)
    return proposal.with_profile(resolve_profile(proposal.profile, parsed), explicit_fields=explicit_fields)


def _explicit_field_choices(overrides: ProfileOverrides) -> frozenset[str]:
    fields = overrides.fields
    if fields is None:
        return frozenset()
    chosen = (fields.lexical_entry_field, fields.principal_parts_field)
    return frozenset(name for name in chosen if name is not None)


def _meaning_field_explicitly_set(overrides: ProfileOverrides) -> bool:
    return overrides.fields is not None and "meaning_field" in overrides.fields.model_fields_set


def _with_meaning_field_cleared(overrides: ProfileOverrides) -> ProfileOverrides:
    values: dict[str, str | None] = {}
    if overrides.fields is not None:
        if overrides.fields.lexical_entry_field is not None:
            values["lexical_entry_field"] = overrides.fields.lexical_entry_field
        if overrides.fields.principal_parts_field is not None:
            values["principal_parts_field"] = overrides.fields.principal_parts_field
    values["meaning_field"] = None
    return overrides.model_copy(update={"fields": FieldOverrides.model_validate(values)})


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
        else (f"\\u{ord(character):04x}" if ord(character) > 0xFF else f"\\x{ord(character):02x}")
        if (
            ord(character) < 32
            or 0x80 <= ord(character) <= 0x9F
            or ord(character) == 127
            or ord(character) in {0x2028, 0x2029}
        )
        else character
        for character in value
    )
    return escaped if len(escaped) <= limit else escaped[: limit - 1] + "…"


def encode_unsafe_controls(value: str, *, preserve_line_breaks: bool = True) -> str:
    """Encode terminal and line-oriented control characters as visible text."""

    encoded: list[str] = []
    for character in value:
        codepoint = ord(character)
        if preserve_line_breaks and character in "\n\r\t":
            encoded.append(character)
        elif codepoint < 32 or 0x7F <= codepoint <= 0x9F or codepoint in {0x2028, 0x2029}:
            encoded.append(f"\\x{codepoint:02x}" if codepoint <= 0xFF else f"\\u{codepoint:04x}")
        elif codepoint in _BIDI_CONTROL_CODES:
            encoded.append(f"\\u{codepoint:04x}")
        else:
            encoded.append(character)
    return "".join(encoded)


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


@dataclass(frozen=True, slots=True)
class _LexicalFieldEvidence:
    """Content-derived ranking evidence for one lexical-entry candidate field."""

    field: str
    sampled: int
    non_empty: int
    word_like: int
    overlap: int
    numeric_codes: int
    content_score: int
    total_score: int
    samples: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PrincipalFieldEvidence:
    """Structural ranking evidence for one (field, separator) principal-parts candidate."""

    field: str
    separator: str
    separator_index: int
    sampled: int
    exact_count: int
    three_count: int
    name_score: int
    score: int


def _lexical_field_evidences(
    records: Sequence[CanonicalSourceRecord],
    fields: Sequence[str],
    principal_field: str,
    separator: str,
) -> tuple[_LexicalFieldEvidence, ...]:
    return tuple(
        _lexical_field_evidence(field, records, principal_field, separator)
        for field in fields
        if field != principal_field
    )


def _lexical_field_evidence(
    field: str,
    records: Sequence[CanonicalSourceRecord],
    principal_field: str,
    separator: str,
) -> _LexicalFieldEvidence:
    values = tuple(unicodedata.normalize("NFC", record.fields.get(field, "")) for record in records)
    non_empty_values = [value for value in values if value.strip()]
    word_like = sum(1 for value in non_empty_values if _SINGLE_WORD_RE.fullmatch(value.strip()))
    overlap = sum(
        1
        for record in records
        if _matches_principal_part(
            record.fields.get(field, ""),
            record.fields.get(principal_field, ""),
            separator,
        )
    )
    numeric_codes = sum(1 for value in non_empty_values if _NUMERIC_CODE_RE.fullmatch(value.strip()))
    sampled = len(values)
    non_empty = len(non_empty_values)
    content_score = overlap * _LEXICAL_OVERLAP_WEIGHT + word_like * _LEXICAL_WORD_WEIGHT + non_empty
    name_bonus = _name_score(field, _LEXICAL_NAME_HINTS) * _LEXICAL_NAME_WEIGHT

    reasons = [
        f"{non_empty}/{sampled} sampled values are non-empty",
        f"{word_like}/{sampled} sampled values are single alphabetic words",
        f"{overlap}/{sampled} sampled values match a principal-part form of the same record",
    ]
    if numeric_codes:
        reasons.append(f"{numeric_codes}/{sampled} sampled values are numeric codes only")
    if name_bonus:
        reasons.append("field name resembles a lexical-entry field name")

    return _LexicalFieldEvidence(
        field=field,
        sampled=sampled,
        non_empty=non_empty,
        word_like=word_like,
        overlap=overlap,
        numeric_codes=numeric_codes,
        content_score=content_score,
        total_score=content_score + name_bonus,
        samples=_sample_values(values, field),
        reasons=tuple(reasons),
    )


def _matches_principal_part(value: str, principal_value: str, separator: str) -> bool:
    folded = unicodedata.normalize("NFC", value).strip().casefold()
    if not folded:
        return False
    parts = {
        unicodedata.normalize("NFC", part).strip().casefold()
        for part in principal_value.split(separator)
        if part.strip()
    }
    return folded in parts


def _principal_field_evidences(
    records: Sequence[CanonicalSourceRecord],
    fields: Sequence[str],
) -> tuple[_PrincipalFieldEvidence, ...]:
    evidences: list[_PrincipalFieldEvidence] = []
    for field in fields:
        values = tuple(record.fields.get(field, "") for record in records)
        name_score = _name_score(field, _PRINCIPAL_NAME_HINTS)
        for separator_index, separator in enumerate(_CANDIDATE_SEPARATORS):
            exact_count = sum(1 for value in values if value.strip() and value.count(separator) == 3)
            three_count = sum(1 for value in values if value.strip() and value.count(separator) == 2)
            evidences.append(
                _PrincipalFieldEvidence(
                    field=field,
                    separator=separator,
                    separator_index=separator_index,
                    sampled=len(values),
                    exact_count=exact_count,
                    three_count=three_count,
                    name_score=name_score,
                    score=exact_count * 100 + three_count * 15 + name_score * 4,
                )
            )
    return tuple(evidences)


def _choose_principal_field(
    evidences: Sequence[_PrincipalFieldEvidence],
) -> tuple[str, str, int, int]:
    if not evidences:
        raise ProfileSetupError("The source has no candidate principal-parts field.")
    best = max(
        evidences,
        key=lambda evidence: (
            evidence.score,
            evidence.exact_count,
            -evidence.separator_index,
            evidence.field,
            evidence.separator,
        ),
    )
    role_count = 4 if best.exact_count else (3 if best.three_count else 4)
    return best.field, best.separator, role_count, best.score


def _choose_lexical_field(evidences: Sequence[_LexicalFieldEvidence]) -> str:
    if not evidences:
        raise ProfileSetupError("The source has no candidate lexical-entry field.")
    return max(evidences, key=lambda evidence: (evidence.total_score, evidence.field)).field


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
        non_empty_values = [value.strip() for value in values if value.strip()]
        name_score = _name_score(field, _MEANING_NAME_HINTS)
        numeric_only = all(_NUMERIC_CODE_RE.fullmatch(value) for value in non_empty_values)
        if non_empty_values and name_score == 0 and numeric_only:
            continue
        score = name_score * 20 + len(non_empty_values)
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
    records: Sequence[CanonicalSourceRecord],
    fields: Sequence[str],
    *,
    lexical_evidences: Sequence[_LexicalFieldEvidence],
    principal_evidences: Sequence[_PrincipalFieldEvidence],
    lexical_field: str,
    principal_field: str,
    meaning_field: str | None,
    separator: str,
) -> tuple[FieldCandidate, ...]:
    lexical_by_field = {evidence.field: evidence for evidence in lexical_evidences}
    principal_evidence = next(
        (
            evidence
            for evidence in principal_evidences
            if evidence.field == principal_field and evidence.separator == separator
        ),
        None,
    )
    candidates: list[FieldCandidate] = []
    for field in fields:
        if field == principal_field:
            role = "principal_parts"
            samples = _sample_values((record.fields.get(field, "") for record in records), field)
            if principal_evidence is None:
                score = 0
                reasons = ["proposed separator is not among the separators observed in the sampled values"]
            else:
                score = principal_evidence.score
                reasons = [
                    f"{principal_evidence.exact_count}/{principal_evidence.sampled} sampled values split into "
                    "four parts by the proposed separator",
                    f"{principal_evidence.three_count}/{principal_evidence.sampled} sampled values split into "
                    "three parts by the proposed separator",
                ]
                if principal_evidence.name_score:
                    reasons.append("field name resembles a principal-parts field name")
        else:
            lexical = lexical_by_field[field]
            if field == lexical_field:
                role = "lexical_entry"
                score = lexical.total_score
                reasons = [*lexical.reasons, "strongest observed content evidence for the lexical-entry role"]
            elif field == meaning_field:
                role = "meaning"
                score = _name_score(field, _MEANING_NAME_HINTS) * 20 + lexical.non_empty
                reasons = [
                    f"{lexical.non_empty}/{lexical.sampled} sampled values are non-empty",
                    "field name and non-empty text fit an optional meaning/gloss field",
                ]
            else:
                role = "unmapped"
                score = 0
                reasons = [*lexical.reasons, "retained as an unmapped source field"]
            samples = lexical.samples
        candidates.append(
            FieldCandidate(field=field, role=role, score=score, evidence=tuple(reasons), sample_values=samples)
        )
    return tuple(candidates)


def _sample_values(values: Iterable[str], field: str) -> tuple[str, ...]:
    samples: list[str] = []
    for value in values:
        if not value.strip():
            continue
        sanitized = safe_source_value(value, field)
        if sanitized not in samples:
            samples.append(sanitized)
        if len(samples) == _LEXICAL_SAMPLE_LIMIT:
            break
    return tuple(samples)


def _field_choice_requirements(
    lexical_evidences: Sequence[_LexicalFieldEvidence],
    principal_evidences: Sequence[_PrincipalFieldEvidence],
    *,
    lexical_field: str,
    principal_field: str,
    explicit_fields: frozenset[str],
) -> tuple[RequiredFieldChoice, ...]:
    requirements: list[RequiredFieldChoice] = []

    if lexical_field not in explicit_fields:
        lexical = _lexical_choice_requirement(lexical_evidences, lexical_field)
        if lexical is not None:
            requirements.append(lexical)

    if principal_field not in explicit_fields:
        principal = _principal_choice_requirement(principal_evidences)
        if principal is not None:
            requirements.append(principal)

    return tuple(requirements)


def _lexical_choice_requirement(
    evidences: Sequence[_LexicalFieldEvidence],
    chosen_field: str,
) -> RequiredFieldChoice | None:
    if not evidences:
        return None
    by_field = {evidence.field: evidence for evidence in evidences}
    chosen = by_field.get(chosen_field)
    if chosen is None:
        return None
    candidate_fields = tuple(sorted(by_field))

    top_content = max(evidence.content_score for evidence in evidences)
    tied = sorted(evidence.field for evidence in evidences if evidence.content_score == top_content)
    if len(tied) > 1:
        return RequiredFieldChoice(
            role="lexical_entry",
            reason=(
                f"content evidence ties between the fields {', '.join(repr(name) for name in tied)}; "
                "an explicit lexical-entry field choice is required"
            ),
            candidates=tuple(tied),
        )

    if chosen.overlap + chosen.word_like == 0:
        return RequiredFieldChoice(
            role="lexical_entry",
            reason=(
                f"field {chosen.field!r} shows no sampled Latin-word or principal-part-match evidence "
                f"({chosen.overlap}/{chosen.sampled} sampled values match a principal-part form); "
                "an explicit lexical-entry field choice is required"
            ),
            candidates=candidate_fields,
        )

    if chosen.non_empty * 2 < chosen.sampled:
        return RequiredFieldChoice(
            role="lexical_entry",
            reason=(
                f"field {chosen.field!r} has sparse evidence "
                f"({chosen.non_empty}/{chosen.sampled} sampled values are non-empty); "
                "an explicit lexical-entry field choice is required"
            ),
            candidates=candidate_fields,
        )
    return None


def _principal_choice_requirement(
    evidences: Sequence[_PrincipalFieldEvidence],
) -> RequiredFieldChoice | None:
    structural: dict[str, tuple[int, int]] = {}
    for evidence in evidences:
        counts = (evidence.exact_count, evidence.three_count)
        structural[evidence.field] = max(structural.get(evidence.field, (0, 0)), counts)

    if not structural:
        return None
    top = max(structural.values())
    candidate_fields = tuple(sorted(structural))
    if top == (0, 0):
        return RequiredFieldChoice(
            role="principal_parts",
            reason=(
                "no sampled value in any field shows a repeated principal-part separator pattern; "
                "an explicit principal-parts field choice is required"
            ),
            candidates=candidate_fields,
        )
    tied = sorted(name for name, counts in structural.items() if counts == top)
    if len(tied) > 1:
        return RequiredFieldChoice(
            role="principal_parts",
            reason=(
                f"equally strong principal-part structure in the fields {', '.join(repr(name) for name in tied)}; "
                "an explicit principal-parts field choice is required"
            ),
            candidates=tuple(tied),
        )
    return None


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
    "RequiredFieldChoice",
    "apply_profile_overrides",
    "build_representative_examples",
    "profile_source_issues",
    "propose_profile",
    "propose_profile_from_inspection",
    "safe_source_value",
    "encode_unsafe_controls",
]
