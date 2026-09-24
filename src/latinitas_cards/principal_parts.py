"""Parse confirmed principal-part fields into semantic roles.

The parser is deliberately limited to the layout confirmed in a
:class:`~latinitas_cards.profile.DeckProfile`.  It preserves display text while
exposing a comparison-only normalization and never chooses a semantic role for
an unmarked omission.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

from .profile import DeckProfile, PrincipalPartLayout
from .sources import CanonicalSourceRecord

ParseFailureStatus = Literal["incomplete", "unsupported", "ambiguous"]

_SEPARATOR_HINTS = (",", ";", " / ", "|", " - ", " — ", "\n", "\t")


@dataclass(frozen=True, slots=True)
class PrincipalPartValue:
    """One named principal-part role and its preserved/normalized values."""

    role: str
    display: str | None
    comparison: str | None

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("principal-part roles must be non-empty")
        if (self.display is None) != (self.comparison is None):
            raise ValueError("omitted principal parts must have no display or comparison value")

    @property
    def identity_role(self) -> str:
        """Return the semantic role used by downstream exercise identity."""

        return self.role

    @property
    def is_omitted(self) -> bool:
        return self.display is None


@dataclass(frozen=True, slots=True)
class ParsedPrincipalParts:
    """A successfully parsed lexical entry with values keyed by semantic role."""

    lexical_entry: str
    lexical_entry_comparison: str
    parts: tuple[PrincipalPartValue, ...]
    source_identity: str | None = None
    source_location: str | None = None
    _by_role: Mapping[str, PrincipalPartValue] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        roles = tuple(part.role for part in parts)
        if len(set(roles)) != len(roles):
            raise ValueError("parsed principal-part roles must be distinct")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "_by_role", MappingProxyType({part.role: part for part in parts}))

    @property
    def by_role(self) -> Mapping[str, PrincipalPartValue]:
        """Return the parsed values keyed by the profile's semantic role names."""

        return self._by_role

    @property
    def semantic_roles(self) -> tuple[str, ...]:
        """Return roles in confirmed profile order, including omitted roles."""

        return tuple(part.role for part in self.parts)

    @property
    def identity_roles(self) -> tuple[str, ...]:
        """Return the role keys that downstream recipes may use for identity."""

        return self.semantic_roles

    @property
    def omitted_roles(self) -> tuple[str, ...]:
        return tuple(part.role for part in self.parts if part.is_omitted)


@dataclass(frozen=True, slots=True)
class PrincipalPartParseSuccess:
    """Typed successful parser output."""

    value: ParsedPrincipalParts
    status: Literal["success"] = "success"


@dataclass(frozen=True, slots=True)
class PrincipalPartParseFailure:
    """Structured parser output for an assumption that was not satisfied."""

    status: ParseFailureStatus
    code: str
    assumption: str
    message: str
    source_identity: str | None = None
    source_location: str | None = None
    observed_count: int | None = None
    expected_count: int | None = None


type PrincipalPartParseResult = PrincipalPartParseSuccess | PrincipalPartParseFailure


def normalize_principal_part_for_comparison(value: str) -> str:
    """Normalize only for comparison; never use this value for display output."""

    decomposed = unicodedata.normalize("NFD", value).casefold()
    without_marks = "".join(character for character in decomposed if unicodedata.category(character) != "Mn")
    return " ".join(without_marks.split())


def parse_principal_parts(record: CanonicalSourceRecord, profile: DeckProfile) -> PrincipalPartParseResult:
    """Parse the profile-selected fields from one canonical source record."""

    lexical_field = profile.fields.lexical_entry_field
    principal_parts_field = profile.fields.principal_parts_field
    source_identity = record.source_identity
    source_location = record.provenance.location

    if lexical_field not in record.fields:
        return _failure(
            "incomplete",
            "missing_lexical_entry",
            "the confirmed lexical-entry field is present in every canonical source record",
            "The confirmed lexical-entry field is missing from this source record.",
            source_identity=source_identity,
            source_location=source_location,
        )
    lexical_entry = _display_value(record.fields[lexical_field])
    if lexical_entry is None:
        return _failure(
            "incomplete",
            "missing_lexical_entry",
            "the confirmed lexical-entry field contains a non-empty value",
            "The confirmed lexical-entry field is empty.",
            source_identity=source_identity,
            source_location=source_location,
        )
    if principal_parts_field not in record.fields:
        return _failure(
            "incomplete",
            "missing_principal_parts_field",
            "the confirmed principal-parts field is present in every canonical source record",
            "The confirmed principal-parts field is missing from this source record.",
            source_identity=source_identity,
            source_location=source_location,
        )

    return parse_principal_part_value(
        record.fields[principal_parts_field],
        profile.principal_parts,
        lexical_entry=lexical_entry,
        source_identity=source_identity,
        source_location=source_location,
    )


def parse_principal_part_value(
    value: str,
    layout: PrincipalPartLayout,
    *,
    lexical_entry: str,
    source_identity: str | None = None,
    source_location: str | None = None,
) -> PrincipalPartParseResult:
    """Parse one principal-parts value using one already-confirmed layout."""

    display_lexical_entry = _display_value(lexical_entry)
    if display_lexical_entry is None:
        return _failure(
            "incomplete",
            "missing_lexical_entry",
            "the confirmed lexical-entry value is non-empty",
            "The lexical-entry value is empty.",
            source_identity=source_identity,
            source_location=source_location,
        )

    if not value.strip():
        return _failure(
            "incomplete",
            "missing_principal_parts",
            "the confirmed principal-parts value is non-empty",
            "The principal-parts value is empty.",
            source_identity=source_identity,
            source_location=source_location,
            expected_count=len(layout.roles),
        )

    segments_or_failure = _split_value(value, layout)
    if isinstance(segments_or_failure, PrincipalPartParseFailure):
        return _with_source(segments_or_failure, source_identity, source_location)

    segments = segments_or_failure
    if len(segments) != len(layout.roles):
        status: ParseFailureStatus = "incomplete" if len(segments) < len(layout.roles) - 1 else "ambiguous"
        code = "too_few_parts" if status == "incomplete" else "unmarked_omission"
        assumption = (
            "the confirmed layout contains enough segments for every required role"
            if status == "incomplete"
            else "each omitted role is explicitly represented by a blank slot or a confirmed shorter layout"
        )
        message = (
            "The principal-parts value does not contain enough segments for the confirmed layout."
            if status == "incomplete"
            else "The principal-parts value omits a role without an explicit blank slot."
        )
        return _failure(
            status,
            code,
            assumption,
            message,
            source_identity=source_identity,
            source_location=source_location,
            observed_count=len(segments),
            expected_count=len(layout.roles),
        )

    values = tuple(
        PrincipalPartValue(
            role=role,
            display=display,
            comparison=None if display is None else normalize_principal_part_for_comparison(display),
        )
        for role, display in zip(layout.roles, (_display_value(segment) for segment in segments), strict=True)
    )
    if any(value.is_omitted for value in values[:2]):
        return _failure(
            "incomplete",
            "required_role_omitted",
            "the first two confirmed semantic roles are present",
            "A required leading principal-part role is explicitly omitted.",
            source_identity=source_identity,
            source_location=source_location,
            observed_count=len(segments),
            expected_count=len(layout.roles),
        )

    return PrincipalPartParseSuccess(
        ParsedPrincipalParts(
            lexical_entry=display_lexical_entry,
            lexical_entry_comparison=normalize_principal_part_for_comparison(display_lexical_entry),
            parts=values,
            source_identity=source_identity,
            source_location=source_location,
        )
    )


def _split_value(value: str, layout: PrincipalPartLayout) -> tuple[str, ...] | PrincipalPartParseFailure:
    expected_separator_count = len(layout.roles) - 1
    separators = layout.separators
    if len(separators) == 1:
        separator = separators[0]
        occurrences = value.count(separator)
        if occurrences > expected_separator_count:
            return _failure(
                "ambiguous",
                "extra_separator",
                "the confirmed separator occurs only between the confirmed number of roles",
                "The principal-parts value contains more configured separators than the confirmed layout.",
                observed_count=occurrences + 1,
                expected_count=len(layout.roles),
            )
        if occurrences < expected_separator_count:
            if occurrences == expected_separator_count - 1:
                return tuple(value.split(separator))
            if _has_unconfigured_separator(value, separators):
                return _failure(
                    "unsupported",
                    "separator_mismatch",
                    "the configured separators match the source principal-parts field",
                    "The source principal-parts separators do not match the confirmed layout.",
                    observed_count=occurrences + 1,
                    expected_count=len(layout.roles),
                )
            return _failure(
                "incomplete",
                "too_few_parts",
                "the confirmed layout contains enough segments for every required role",
                "The principal-parts value does not contain enough configured separators.",
                observed_count=occurrences + 1,
                expected_count=len(layout.roles),
            )
        return tuple(value.split(separator))

    segments = _segment_value(value, separators)
    if segments is None:
        if _has_unconfigured_separator(value, separators):
            return _failure(
                "unsupported",
                "separator_mismatch",
                "the configured separators match the source principal-parts field",
                "The source principal-parts separators do not match the confirmed layout.",
                expected_count=len(layout.roles),
            )
        return _failure(
            "incomplete",
            "too_few_parts",
            "the confirmed layout contains every position-specific separator",
            "The principal-parts value is missing a configured separator.",
            expected_count=len(layout.roles),
        )

    if any(separator in segment for segment in segments for separator in separators):
        return _failure(
            "ambiguous",
            "extra_separator",
            "the configured separators occur only at the confirmed role boundaries",
            "The principal-parts value contains an additional configured separator.",
            observed_count=len(segments) + 1,
            expected_count=len(layout.roles),
        )
    return segments


def _segment_value(value: str, separators: tuple[str, ...]) -> tuple[str, ...] | None:
    segments: list[str] = []
    start = 0
    for separator in separators:
        position = value.find(separator, start)
        if position < 0:
            return None
        segments.append(value[start:position])
        start = position + len(separator)
    return tuple((*segments, value[start:]))


def _has_unconfigured_separator(value: str, configured: tuple[str, ...]) -> bool:
    configured_set = set(configured)
    return any(hint not in configured_set and hint in value for hint in _SEPARATOR_HINTS)


def _display_value(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _failure(
    status: ParseFailureStatus,
    code: str,
    assumption: str,
    message: str,
    *,
    source_identity: str | None = None,
    source_location: str | None = None,
    observed_count: int | None = None,
    expected_count: int | None = None,
) -> PrincipalPartParseFailure:
    return PrincipalPartParseFailure(
        status=status,
        code=code,
        assumption=assumption,
        message=message,
        source_identity=source_identity,
        source_location=source_location,
        observed_count=observed_count,
        expected_count=expected_count,
    )


def _with_source(
    failure: PrincipalPartParseFailure,
    source_identity: str | None,
    source_location: str | None,
) -> PrincipalPartParseFailure:
    return replace(
        failure,
        source_identity=source_identity,
        source_location=source_location,
    )


__all__ = [
    "ParseFailureStatus",
    "ParsedPrincipalParts",
    "PrincipalPartParseFailure",
    "PrincipalPartParseResult",
    "PrincipalPartParseSuccess",
    "PrincipalPartValue",
    "normalize_principal_part_for_comparison",
    "parse_principal_parts",
    "parse_principal_part_value",
]
