from pathlib import Path

from latinitas_cards.identity import resolve_source_identities
from latinitas_cards.principal_parts import PrincipalPartParseFailure, PrincipalPartParseSuccess, parse_principal_parts
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig
from latinitas_cards.sources import inspect_source, read_source_records

FIXTURE = Path(__file__).parents[1] / "fixtures" / "representative-university-latin.apkg"
VALIDATION_NOTE = Path(__file__).parents[1].parent / "docs" / "representative-deck-validation.md"
PARSER_NOTE = Path(__file__).parents[1].parent / "docs" / "principal-part-parsing.md"


def _profile() -> DeckProfile:
    return DeckProfile.default(
        note_type="Representative Latin Vocabulary",
        lexical_entry_field="Entry",
        principal_parts_field="Construction hints",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="note_guid"),
        principal_part_roles=(
            "present_infinitive",
            "present_1s",
            "perfect_1s",
            "perfect_passive_participle",
        ),
        separators=(",",),
    )


def test_sanitized_representative_fixture_preserves_mixed_named_fields_and_native_identity() -> None:
    inspection = inspect_source(FIXTURE)

    assert inspection.note_types == ("Representative Latin Vocabulary",)
    assert inspection.fields_by_note_type == {
        "Representative Latin Vocabulary": (
            "Construction hints",
            "Entry",
            "Flag",
            "German gloss",
            "Reference A",
            "Reference B",
            "Reference C",
        )
    }
    assert len(inspection.records) == 5
    assert all(record.source_kind == "apkg" for record in inspection.records)
    assert all(record.note_guid == record.source_identity for record in inspection.records)
    assert len({record.source_identity for record in inspection.records}) == len(inspection.records)
    assert all((record.provenance.note_id or 0) >= 1001 for record in inspection.records)


def test_fixture_keeps_lexical_variants_and_principal_part_layout_variants() -> None:
    records = read_source_records(FIXTURE)
    entries = {record.fields["Entry"] for record in records}
    construction_hints = {record.fields["Construction hints"] for record in records}

    assert {"dīcere", "amāre|amare", "<b>sequor</b>", "vidēre / videre", "ferre"} <= entries
    assert "" in construction_hints
    assert any(", " in value for value in construction_hints)
    assert any("; " in value for value in construction_hints)
    assert any("|" in value for value in construction_hints)
    assert all(record.fields["German gloss"] or record.fields["Entry"] == "ferre" for record in records)


def test_fixture_maps_observed_primary_order_and_reports_variant_layouts() -> None:
    records = read_source_records(FIXTURE)
    results = tuple(parse_principal_parts(record, _profile()) for record in records)

    assert isinstance(results[0], PrincipalPartParseSuccess)
    assert results[0].value.semantic_roles == (
        "present_infinitive",
        "present_1s",
        "perfect_1s",
        "perfect_passive_participle",
    )
    assert [part.display for part in results[0].value.parts] == ["dīcere", "dīcō", "dīxī", "dictum"]
    assert isinstance(results[1], PrincipalPartParseSuccess)
    assert isinstance(results[2], PrincipalPartParseFailure)
    assert results[2].code == "missing_principal_parts"
    assert isinstance(results[3], PrincipalPartParseFailure)
    assert results[3].code == "separator_mismatch"
    assert isinstance(results[4], PrincipalPartParseSuccess)
    assert [part.display for part in results[4].value.parts] == ["ferre", "ferō", "tulī", "lātum"]
    assert records[4].fields["German gloss"] == ""


def test_fixture_identity_resolution_uses_guid_with_explicit_principal_part_mapping() -> None:
    records = read_source_records(FIXTURE)

    identities = resolve_source_identities(records, _profile())

    assert identities == tuple(record.note_guid for record in records)
    assert _profile().fields.principal_parts_field in records[0].fields


def test_validation_note_keeps_source_metadata_private_and_bounds_candidate_counts() -> None:
    note = VALIDATION_NOTE.read_text(encoding="utf-8")
    parser_note = PARSER_NOTE.read_text(encoding="utf-8")
    parser_text = " ".join(parser_note.split())

    assert "source lexical-entry field" in note
    assert "source German gloss/meaning field" in note
    assert "source construction/form field" in note
    assert "a subset of these 37" in note
    assert "not an additional group" in note
    assert "structural/lexical-shape candidates" in note
    assert "perfect_passive_participle" in note
    assert "Partizip Perfekt Passiv (PPP)" in note
    assert "None of these counts alone is a count of eligible verbs." in note
    assert "role assignment comes from the approved profile, not from the isolated form" in note
    assert "unambiguous public PPP example" not in note
    assert "supine" not in note.lower()
    assert "T-011 validated the representative-deck mapping and initial German terminology" in parser_text
