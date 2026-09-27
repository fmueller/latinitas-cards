"""Stratified extraction-content review over committed synthetic and sanitized populations.

The recorded tables below are the hand-derived adjudications of the T-032 review:
every expected status, code, display, and comparison value was derived from the
supported parser matrix in ``docs/principal-part-parsing.md`` and the safe-HTML
contract, not by re-running the pipeline.  The tests verify that the actual
pipeline reproduces the adjudicated review, that the stratified sample selection
is deterministic, and that the committed review note states the same counts and
the synthetic-only coverage boundary.
"""

from pathlib import Path
from typing import Any

from latinitas_cards.generation import GenerationSkip, generate_learning_object_notes
from latinitas_cards.notes import GeneratedNote
from latinitas_cards.principal_parts import (
    PrincipalPartParseFailure,
    PrincipalPartParseSuccess,
    parse_principal_parts,
)
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig
from latinitas_cards.sources import CanonicalSourceRecord, read_source_records

CORPUS_FIXTURE = Path(__file__).parents[1] / "fixtures" / "extraction-review-corpus.csv"
SANITIZED_FIXTURE = Path(__file__).parents[1] / "fixtures" / "representative-university-latin.apkg"
REVIEW_NOTE = Path(__file__).parents[2] / "docs" / "extraction-content-review.md"

CORPUS_ROLES = (
    "present_infinitive",
    "present_1s",
    "perfect_1s",
    "perfect_passive_participle",
)

RECORDED_OUTCOMES: dict[str, tuple[str, str | None]] = {
    "rev-001": ("generated", None),
    "rev-002": ("generated", None),
    "rev-003": ("generated", None),
    "rev-004": ("generated", None),
    "rev-005": ("generated", None),
    "rev-006": ("generated", None),
    "rev-007": ("generated", None),
    "rev-008": ("generated", None),
    "rev-009": ("generated", None),
    "rev-010": ("generated", None),
    "rev-011": ("incomplete", "missing_principal_parts"),
    "rev-012": ("incomplete", "too_few_parts"),
    "rev-013": ("incomplete", "required_role_omitted"),
    "rev-014": ("incomplete", "required_role_omitted"),
    "rev-015": ("incomplete", "omitted_principal_part"),
    "rev-016": ("incomplete", "omitted_principal_part"),
    "rev-017": ("incomplete", "omitted_principal_part"),
    "rev-018": ("incomplete", "missing_lexical_entry"),
    "rev-019": ("unsupported", "separator_mismatch"),
    "rev-020": ("unsupported", "separator_mismatch"),
    "rev-021": ("unsupported", "separator_mismatch"),
    "rev-022": ("ambiguous", "extra_separator"),
    "rev-023": ("ambiguous", "unmarked_omission"),
    "rev-024": ("ambiguous", "multi_object_source"),
    "rev-025": ("ambiguous", "unmarked_omission"),
}

RECORDED_POPULATION_COUNTS = {"generated": 10, "incomplete": 8, "unsupported": 3, "ambiguous": 4}
SAMPLES_PER_STRATUM = 2
RECORDED_DEEP_SAMPLE = (
    "rev-001",
    "rev-002",
    "rev-011",
    "rev-012",
    "rev-013",
    "rev-014",
    "rev-015",
    "rev-016",
    "rev-018",
    "rev-019",
    "rev-020",
    "rev-022",
    "rev-023",
    "rev-024",
    "rev-025",
)
RECORDED_REMAINDER = (
    "rev-003",
    "rev-004",
    "rev-005",
    "rev-006",
    "rev-007",
    "rev-008",
    "rev-009",
    "rev-010",
    "rev-017",
    "rev-021",
)

RECORDED_SANITIZED_OUTCOMES: dict[str, tuple[str, str | None]] = {
    "fixture-guid-001": ("generated", None),
    "fixture-guid-002": ("generated", None),
    "fixture-guid-003": ("incomplete", "missing_principal_parts"),
    "fixture-guid-004": ("unsupported", "separator_mismatch"),
    "fixture-guid-005": ("generated", None),
}


def _corpus_profile() -> DeckProfile:
    return DeckProfile.default(
        note_type="CSV source",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="Stable ID"),
        principal_part_roles=CORPUS_ROLES,
        separators=(",",),
    )


def _sanitized_profile() -> DeckProfile:
    return DeckProfile.default(
        note_type="Representative Latin Vocabulary",
        lexical_entry_field="Entry",
        principal_parts_field="Construction hints",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="note_guid"),
        principal_part_roles=CORPUS_ROLES,
        separators=(",",),
    )


def _corpus_records() -> tuple[CanonicalSourceRecord, ...]:
    return read_source_records(CORPUS_FIXTURE, source_id_field="Stable ID")


def _review(
    records: tuple[CanonicalSourceRecord, ...],
    profile: DeckProfile,
    **kwargs: Any,
) -> tuple[dict[str, GeneratedNote], dict[str, tuple[GenerationSkip, ...]]]:
    result = generate_learning_object_notes(records, profile, **kwargs)
    notes_by_id = {note.provenance.source_identity or "": note for note in result.notes}
    skips_by_id: dict[str, tuple[GenerationSkip, ...]] = {record_id: () for record_id in notes_by_id}
    for skip in result.skips:
        identity = skip.source_identity or ""
        skips_by_id.setdefault(identity, ())
        skips_by_id[identity] = (*skips_by_id[identity], skip)
    return notes_by_id, skips_by_id


def _pipeline_category(note: GeneratedNote | None, skips: tuple[GenerationSkip, ...]) -> str:
    if note is not None and not skips:
        return "generated"
    return skips[0].status


def _parse_displays(record: CanonicalSourceRecord, profile: DeckProfile) -> tuple[str | None, ...]:
    result = parse_principal_parts(record, profile)
    assert isinstance(result, PrincipalPartParseSuccess)
    return tuple(part.display for part in result.value.parts)


def _assert_failure(
    record: CanonicalSourceRecord,
    profile: DeckProfile,
    *,
    status: str,
    code: str,
) -> None:
    result = parse_principal_parts(record, profile)
    assert isinstance(result, PrincipalPartParseFailure)
    assert result.status == status
    assert result.code == code


def _corpus_record(stable_id: str) -> CanonicalSourceRecord:
    return next(record for record in _corpus_records() if record.source_identity == stable_id)


def test_corpus_population_matches_the_recorded_review_counts() -> None:
    profile = _corpus_profile()
    corpus_records = _corpus_records()
    assert {record.source_identity for record in corpus_records} == set(RECORDED_OUTCOMES)
    assert len(corpus_records) == 25
    notes_by_id, skips_by_id = _review(corpus_records, profile, source_scope="scope-extraction-review")

    pipeline_outcomes = {
        record_id: (
            _pipeline_category(notes_by_id.get(record_id), skips_by_id.get(record_id, ())),
            skips_by_id[record_id][0].code if skips_by_id.get(record_id) else None,
        )
        for record_id in RECORDED_OUTCOMES
    }
    assert pipeline_outcomes == RECORDED_OUTCOMES

    counts = {"generated": 0, "incomplete": 0, "unsupported": 0, "ambiguous": 0}
    for category, _ in RECORDED_OUTCOMES.values():
        counts[category] += 1
    assert counts == RECORDED_POPULATION_COUNTS


def test_stratified_sample_selection_is_deterministic_and_recorded() -> None:
    strata: dict[tuple[str, str | None], list[str]] = {}
    for record_id, outcome in RECORDED_OUTCOMES.items():
        strata.setdefault(outcome, []).append(record_id)
    for stratum in strata.values():
        stratum.sort()

    sample = tuple(
        record_id
        for stratum_key in sorted(strata, key=lambda item: (item[0], item[1] or ""))
        for record_id in strata[stratum_key][:SAMPLES_PER_STRATUM]
    )

    assert tuple(sorted(sample)) == tuple(sorted(RECORDED_DEEP_SAMPLE))
    assert len(sample) == len(RECORDED_DEEP_SAMPLE) == 15
    remainder = tuple(sorted(set(RECORDED_OUTCOMES) - set(RECORDED_DEEP_SAMPLE)))
    assert remainder == RECORDED_REMAINDER
    assert len(remainder) == 10


def test_generated_sample_adjudications_hold_value_contracts() -> None:
    profile = _corpus_profile()

    assert _parse_displays(_corpus_record("rev-001"), profile) == ("dīcere", "dīcō", "dīxī", "dictum")
    assert _parse_displays(_corpus_record("rev-002"), profile) == ("ferre", "ferō", "tulī", "lātum")

    notes_by_id, skips_by_id = _review(_corpus_records(), profile, source_scope="scope-extraction-review")
    assert not skips_by_id["rev-001"]
    assert not skips_by_id["rev-002"]
    rev_001 = notes_by_id["rev-001"]
    rev_002 = notes_by_id["rev-002"]
    assert rev_001.content.lemma == "dīcō"
    assert rev_001.content.meaning == "sagen"
    assert "<strong>Partizip Perfekt Passiv (PPP):</strong> dictum" in rev_001.content.principal_parts
    assert rev_002.content.meaning == ""
    for note in (rev_001, rev_002):
        assert len(note.card_keys) == 8
        eligible_answers = [
            line.split("</strong>")[-1].strip()
            for line in note.content.principal_parts.split("<br>")
            if line.split("</strong>")[-1].strip() != "—"
        ]
        assert len(eligible_answers) == 4
        assert all(eligible_answers)


def test_incomplete_sample_adjudications_distinguish_missing_data_from_the_markup_defect() -> None:
    profile = _corpus_profile()
    notes_by_id, skips_by_id = _review(_corpus_records(), profile, source_scope="scope-extraction-review")

    _assert_failure(_corpus_record("rev-011"), profile, status="incomplete", code="missing_principal_parts")
    _assert_failure(_corpus_record("rev-012"), profile, status="incomplete", code="too_few_parts")
    _assert_failure(_corpus_record("rev-013"), profile, status="incomplete", code="required_role_omitted")
    _assert_failure(_corpus_record("rev-014"), profile, status="incomplete", code="required_role_omitted")
    _assert_failure(_corpus_record("rev-018"), profile, status="incomplete", code="missing_lexical_entry")

    for record_id, code in (
        ("rev-011", "missing_principal_parts"),
        ("rev-012", "too_few_parts"),
        ("rev-013", "required_role_omitted"),
        ("rev-014", "required_role_omitted"),
        ("rev-018", "missing_lexical_entry"),
    ):
        assert record_id not in notes_by_id
        skips = skips_by_id[record_id]
        assert len(skips) == 1
        assert skips[0].status == "incomplete"
        assert skips[0].code == code
        assert "amātum" not in skips[0].message


def test_markup_only_sample_adjudications_keep_notes_and_omit_blank_answers() -> None:
    profile = _corpus_profile()
    records = {record_id: _corpus_record(record_id) for record_id in ("rev-015", "rev-016")}

    assert _parse_displays(records["rev-015"], profile) == ("amāre", "amō", None, "amātum")
    assert _parse_displays(records["rev-016"], profile) == ("amāre", "amō", None, "amātum")
    parsed_015 = parse_principal_parts(records["rev-015"], profile)
    assert isinstance(parsed_015, PrincipalPartParseSuccess)
    assert parsed_015.value.by_role["perfect_1s"].raw == " <b></b>"
    parsed_016 = parse_principal_parts(records["rev-016"], profile)
    assert isinstance(parsed_016, PrincipalPartParseSuccess)
    assert parsed_016.value.by_role["perfect_1s"].raw == " <!-- Form unklar -->"

    notes_by_id, skips_by_id = _review(_corpus_records(), profile, source_scope="scope-extraction-review")
    for record_id in ("rev-015", "rev-016"):
        note = notes_by_id[record_id]
        skips = skips_by_id[record_id]
        assert len(skips) == 1
        assert skips[0].status == "incomplete"
        assert skips[0].code == "omitted_principal_part"
        assert "perfect_1s" in skips[0].message
        assert len(note.card_keys) == 6
        assert "principal_part_completion:perfect_1s" not in note.card_keys
        assert "principal_part_recognition:perfect_1s" not in note.card_keys
        assert "<strong>Perfekt, 1. Person Singular:</strong> —" in note.content.principal_parts
        eligible_answers = [
            line.split("</strong>")[-1].strip()
            for line in note.content.principal_parts.split("<br>")
            if line.split("</strong>")[-1].strip() != "—"
        ]
        assert eligible_answers == ["amāre", "amō", "amātum"]


def test_unsupported_and_ambiguous_sample_adjudications_match_the_matrix() -> None:
    profile = _corpus_profile()
    notes_by_id, skips_by_id = _review(_corpus_records(), profile, source_scope="scope-extraction-review")

    for record_id, code in (
        ("rev-019", "separator_mismatch"),
        ("rev-020", "separator_mismatch"),
        ("rev-022", "extra_separator"),
        ("rev-023", "unmarked_omission"),
        ("rev-024", "multi_object_source"),
        ("rev-025", "unmarked_omission"),
    ):
        assert record_id not in notes_by_id
        skips = skips_by_id[record_id]
        assert len(skips) == 1
        assert skips[0].code == code
        assert skips[0].status == RECORDED_OUTCOMES[record_id][0]
        assert "dictum" not in skips[0].message


def test_sanitized_fixture_population_is_fully_reviewed() -> None:
    profile = _sanitized_profile()
    records = read_source_records(SANITIZED_FIXTURE)

    assert _parse_displays(records[0], profile) == ("dīcere", "dīcō", "dīxī", "dictum")
    assert _parse_displays(records[1], profile) == (
        "amāre|amare",
        "amō|amo",
        "amāvī|amavi",
        "amātum|amatum",
    )
    _assert_failure(records[2], profile, status="incomplete", code="missing_principal_parts")
    _assert_failure(records[3], profile, status="unsupported", code="separator_mismatch")
    assert _parse_displays(records[4], profile) == ("ferre", "ferō", "tulī", "lātum")

    notes_by_id, skips_by_id = _review(records, profile)
    pipeline_outcomes = {}
    for record in records:
        identity = record.source_identity or ""
        skips = skips_by_id.get(identity, ())
        first_skip = skips[0] if skips else None
        pipeline_outcomes[identity] = (
            _pipeline_category(notes_by_id.get(identity), skips),
            first_skip.code if first_skip is not None else None,
        )
    assert pipeline_outcomes == RECORDED_SANITIZED_OUTCOMES

    alternatives_note = notes_by_id["fixture-guid-002"]
    assert alternatives_note.content.lemma == "amāre|amare"
    assert len(alternatives_note.card_keys) == 8
    assert notes_by_id["fixture-guid-005"].content.meaning == ""
    assert notes_by_id["fixture-guid-001"].content.meaning == "sagen"


def test_review_note_records_counts_selection_and_synthetic_boundary() -> None:
    note = REVIEW_NOTE.read_text(encoding="utf-8")
    text = " ".join(note.split())

    assert "extraction-review-corpus.csv" in text
    assert "25 synthetic entries" in text
    assert "5 sanitized entries" in text
    assert "generated: 10" in text
    assert "incomplete: 8" in text
    assert "unsupported: 3" in text
    assert "ambiguous: 4" in text
    assert "deep-reviewed 15" in text
    assert "unreviewed remainder: 10" in text
    assert "not a claim of universal or private-deck correctness" in text
    assert "No private deck, collection, or export data was used" in text
    assert "separator success alone does not establish semantic eligibility" in text
    assert "Partizip Perfekt Passiv (PPP)" in text
