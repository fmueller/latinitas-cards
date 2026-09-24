from dataclasses import replace
from pathlib import Path

from latinitas_cards.principal_parts import (
    PrincipalPartParseFailure,
    PrincipalPartParseSuccess,
    parse_principal_part_value,
    parse_principal_parts,
)
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig
from latinitas_cards.sources import CanonicalSourceRecord, SourceProvenance


def _profile(
    *,
    roles: tuple[str, ...] = ("present_1s", "present_infinitive", "perfect_1s", "supine"),
    separators: tuple[str, ...] = (" — ",),
) -> DeckProfile:
    return DeckProfile.default(
        note_type="Latin vocabulary",
        lexical_entry_field="Lemma",
        principal_parts_field="Principal parts",
        meaning_field="German",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="Source ID"),
        principal_part_roles=roles,
        separators=separators,
    )


def _record(
    principal_parts: str,
    *,
    lexical_entry: str = "dīcō",
    source_identity: str = "entry-17",
    profile: DeckProfile | None = None,
) -> CanonicalSourceRecord:
    resolved_profile = profile or _profile()
    return CanonicalSourceRecord(
        source_kind="csv",
        note_type=resolved_profile.note_type,
        fields={
            "Source ID": source_identity,
            resolved_profile.fields.lexical_entry_field: lexical_entry,
            resolved_profile.fields.principal_parts_field: principal_parts,
            "German": "sagen",
        },
        provenance=SourceProvenance(
            source_path=Path("deck.csv"),
            location="row 2",
            row_number=2,
        ),
        source_identity=source_identity,
    )


def test_regular_entry_maps_display_values_to_named_roles_without_losing_macrons() -> None:
    result = parse_principal_parts(
        _record("dīcō — dīcere — dīxī — dictum"),
        _profile(),
    )

    assert isinstance(result, PrincipalPartParseSuccess)
    parsed = result.value
    assert parsed.lexical_entry == "dīcō"
    assert parsed.lexical_entry_comparison == "dico"
    assert parsed.semantic_roles == ("present_1s", "present_infinitive", "perfect_1s", "supine")
    assert parsed.by_role["perfect_1s"].display == "dīxī"
    assert parsed.by_role["perfect_1s"].comparison == "dixi"
    assert parsed.by_role["perfect_1s"].identity_role == "perfect_1s"
    assert parsed.source_identity == "entry-17"


def test_configured_position_specific_separators_are_preserved_as_one_layout() -> None:
    profile = _profile(separators=(", ", "; ", " / "))

    result = parse_principal_parts(
        _record("amo, amare; amavi / amatum", profile=profile),
        profile,
    )

    assert isinstance(result, PrincipalPartParseSuccess)
    assert [part.display for part in result.value.parts] == ["amo", "amare", "amavi", "amatum"]


def test_confirmed_three_role_layout_supports_deponent_entries_without_inference() -> None:
    profile = _profile(roles=("present_1s", "present_infinitive", "perfect_1s"))

    result = parse_principal_parts(
        _record("sequor — sequi — secutus sum", lexical_entry="sequor", profile=profile),
        profile,
    )

    assert isinstance(result, PrincipalPartParseSuccess)
    assert result.value.semantic_roles == ("present_1s", "present_infinitive", "perfect_1s")
    assert result.value.by_role["perfect_1s"].display == "secutus sum"


def test_explicit_blank_slot_represents_an_omitted_form_without_shifting_roles() -> None:
    profile = _profile()

    result = parse_principal_parts(
        _record("sum — esse — fui — ", lexical_entry="sum", profile=profile),
        profile,
    )

    assert isinstance(result, PrincipalPartParseSuccess)
    omitted = result.value.by_role["supine"]
    assert omitted.display is None
    assert omitted.comparison is None
    assert omitted.is_omitted


def test_unmarked_short_layout_is_ambiguous_instead_of_guessing_the_omitted_role() -> None:
    profile = _profile()

    result = parse_principal_parts(
        _record("amo — amare — amavi", lexical_entry="amō", profile=profile),
        profile,
    )

    assert isinstance(result, PrincipalPartParseFailure)
    assert result.status == "ambiguous"
    assert result.code == "unmarked_omission"
    assert "role" in result.assumption
    assert result.source_identity == "entry-17"


def test_wrong_separator_is_unsupported_and_does_not_expose_field_contents() -> None:
    profile = _profile()

    result = parse_principal_parts(
        _record("amo, amare, amavi, amatum", lexical_entry="amō", profile=profile),
        profile,
    )

    assert isinstance(result, PrincipalPartParseFailure)
    assert result.status == "unsupported"
    assert result.code == "separator_mismatch"
    assert "configured separators" in result.assumption
    assert "amatum" not in result.message


def test_empty_and_too_short_values_are_incomplete() -> None:
    profile = _profile()

    empty = parse_principal_parts(_record("", profile=profile), profile)
    short = parse_principal_parts(_record("amo — amare", profile=profile), profile)

    assert isinstance(empty, PrincipalPartParseFailure)
    assert empty.status == "incomplete"
    assert empty.code == "missing_principal_parts"
    assert isinstance(short, PrincipalPartParseFailure)
    assert short.status == "incomplete"
    assert short.code == "too_few_parts"


def test_missing_leading_part_is_incomplete_even_when_the_slot_is_explicit() -> None:
    profile = _profile()

    result = parse_principal_parts(
        _record(" — amare — amavi — amatum", profile=profile),
        profile,
    )

    assert isinstance(result, PrincipalPartParseFailure)
    assert result.status == "incomplete"
    assert result.code == "required_role_omitted"


def test_extra_separator_is_ambiguous_and_not_silently_discarded() -> None:
    profile = _profile()

    result = parse_principal_parts(
        _record("amo — amare — amavi — amatum — extra", profile=profile),
        profile,
    )

    assert isinstance(result, PrincipalPartParseFailure)
    assert result.status == "ambiguous"
    assert result.code == "extra_separator"


def test_position_specific_extra_separator_inside_a_non_final_segment_is_ambiguous() -> None:
    profile = _profile(separators=(",", "; ", " / "))

    result = parse_principal_parts(
        _record("a, b; c, d / e", profile=profile),
        profile,
    )

    assert isinstance(result, PrincipalPartParseFailure)
    assert result.status == "ambiguous"
    assert result.code == "extra_separator"


def test_extreme_position_specific_layout_does_not_escape_with_recursion_error() -> None:
    roles = tuple(f"role_{index}" for index in range(1_051))
    separators = tuple(f" |{index}| " for index in range(1_050))
    profile = _profile(roles=roles, separators=separators)
    value = "".join(f"value_{index}{separator}" for index, separator in enumerate(separators)) + "value_1050"

    result = parse_principal_part_value(
        value,
        profile.principal_parts,
        lexical_entry="entry",
    )

    assert isinstance(result, PrincipalPartParseSuccess)
    assert len(result.value.parts) == len(roles)


def test_value_parser_can_be_used_without_a_source_record() -> None:
    profile = _profile()

    result = parse_principal_part_value(
        "ferō — ferre — tulī — lātum",
        profile.principal_parts,
        lexical_entry="ferō",
    )

    assert isinstance(result, PrincipalPartParseSuccess)
    assert result.value.source_identity is None
    assert result.value.by_role["supine"].comparison == "latum"


def test_record_parser_uses_profile_field_names_instead_of_positional_fields() -> None:
    profile = _profile()
    record = replace(
        _record("dīcō — dīcere — dīxī — dictum", profile=profile),
        fields={
            "German": "sagen",
            "Source ID": "entry-17",
            "Principal parts": "dīcō — dīcere — dīxī — dictum",
            "Lemma": "dīcō",
            "Unrelated": "not a principal part",
        },
    )

    result = parse_principal_parts(record, profile)

    assert isinstance(result, PrincipalPartParseSuccess)
    assert result.value.lexical_entry == "dīcō"
