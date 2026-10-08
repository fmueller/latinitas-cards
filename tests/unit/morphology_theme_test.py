import csv
import io
from dataclasses import replace
from pathlib import Path

import pytest

from latinitas_cards.cards import render_cards
from latinitas_cards.claim_review import Claim, Evidence, assess_claim, review_claim
from latinitas_cards.commands.principal_parts import render_principal_part_preview
from latinitas_cards.generation import generate_learning_object_notes
from latinitas_cards.preview_export import PrincipalPartExportResult, deterministic_csv_bytes
from latinitas_cards.principal_parts import ParsedPrincipalParts, PrincipalPartValue
from latinitas_cards.principal_relationships import PrincipalPartComparison, RoleComparison, compare_principal_parts
from latinitas_cards.profile import DeckProfile, ProfileValidationError
from latinitas_cards.source_extraction import SourceExtraction
from latinitas_cards.sources import CanonicalSourceRecord, SourceProvenance


def profile() -> DeckProfile:
    return DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms")


def test_versioned_theme_roundtrip_overrides_and_legacy() -> None:
    original = profile()
    assert original.schema_version == 2
    effective = original.apply_overrides({"morphology": {"theme": "monochrome", "appearance": "dark"}})
    assert effective.to_machine_readable()["morphology"] == {
        "version": 1,
        "theme": "monochrome",
        "appearance": "dark",
        "comparison": "static",
    }
    assert DeckProfile.from_json(effective.to_human_readable()) == effective
    assert original.morphology.theme == "muted"
    legacy = original.to_machine_readable()
    legacy["schema_version"] = 1
    legacy.pop("morphology")
    assert DeckProfile.from_mapping(legacy).morphology.comparison == "static"
    legacy["morphology"] = {"theme": "monochrome"}
    with pytest.raises(ProfileValidationError, match="legacy"):
        DeckProfile.from_mapping(legacy)


@pytest.mark.parametrize("morphology", [{}, {"theme": "monochrome"}])
def test_explicit_profile_morphology_requires_version(morphology: dict[str, object]) -> None:
    data = profile().to_machine_readable()
    data["morphology"] = morphology
    with pytest.raises(ProfileValidationError, match="morphology.*version.*required"):
        DeckProfile.from_mapping(data)


def test_absent_morphology_defaults_and_versioned_partial_settings_roundtrip() -> None:
    data = profile().to_machine_readable()
    data.pop("morphology")
    assert DeckProfile.from_mapping(data).morphology == profile().morphology
    data["morphology"] = {"version": 1, "theme": "monochrome"}
    loaded = DeckProfile.from_mapping(data)
    assert loaded.to_machine_readable()["morphology"] == {
        "version": 1,
        "theme": "monochrome",
        "appearance": "light",
        "comparison": "static",
    }
    assert DeckProfile.from_json(loaded.to_json()) == loaded


@pytest.mark.parametrize("override", [{"version": 99}, {"theme": "neon"}, {"appearance": "auto"}, {"comparison": "js"}])
def test_unknown_presentation_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ProfileValidationError):
        profile().apply_overrides({"morphology": override})


def test_hostile_role_labels_and_segmentation_are_literal_not_markup() -> None:
    hostile = '<svg onload="evil()">&lt;i&gt;\nalt'
    parsed = ParsedPrincipalParts("&lt;lemma&gt;", "lemma", (PrincipalPartValue("perfect_1s", "amāvī", "amavi"),))
    comparison = PrincipalPartComparison((RoleComparison(hostile, hostile, "present", hostile, hostile, hostile),))
    answer = next(
        card.answer
        for card in render_cards(parsed, selected_recipes=profile().selected_recipes, comparison=comparison)
        if card.eligible
    )
    assert "&lt;svg onload=&quot;evil()&quot;&gt;&amp;lt;i&amp;gt;<br>alt" in answer
    assert "<strong>&lt;svg onload=&quot;evil()&quot;&gt;&amp;lt;i&amp;gt;<br>alt:</strong>" in answer
    assert '<span class="morphology-lemma">&amp;lt;lemma&amp;gt;</span>' in answer
    assert "<svg" not in answer and "<i>" not in answer
    assert 'class="morphology-marker"' not in answer


@pytest.mark.parametrize("theme", ["muted", "monochrome"])
@pytest.mark.parametrize("appearance", ["light", "dark"])
@pytest.mark.parametrize("mode", ["static", "disclosure"])
def test_safe_comparison_and_stable_slots(theme: str, appearance: str, mode: str) -> None:
    hostile = '<img src=x onerror="alert(1)"> &lt;b&gt;\n"quoted"'
    parsed = ParsedPrincipalParts("amō", "amo", (PrincipalPartValue("perfect_1s", "amāvī", "amavi"),))
    comparison = PrincipalPartComparison(
        (
            RoleComparison("present_1s", hostile, "present", hostile, "amā- | -v- | -ī", hostile),
            RoleComparison("present_infinitive", None, "absent", hostile, hostile, hostile),
            RoleComparison("perfect_1s", "amāvī", "present", "", "amā- | -v- | -ī", hostile),
            RoleComparison("supine", hostile, "withheld", hostile, hostile, hostile),
        )
    )
    settings = profile().apply_overrides({"morphology": {"theme": theme, "appearance": appearance, "comparison": mode}})
    cards = render_cards(
        parsed, selected_recipes=settings.selected_recipes, comparison=comparison, morphology=settings.morphology
    )
    base = render_cards(parsed, selected_recipes=settings.selected_recipes, comparison=comparison)
    assert [(c.slot, c.eligible, c.guard, c.prompt) for c in cards] == [
        (c.slot, c.eligible, c.guard, c.prompt) for c in base
    ]
    answer = next(c.answer for c in cards if c.eligible)
    assert f"morphology-{theme} morphology-{appearance}" in answer
    assert ("<details>" in answer) == (mode == "disclosure")
    assert "Stammformen vergleichen" in answer
    assert "Nicht vorhanden" in answer and "Zurückgehalten" in answer
    assert "<img" not in answer and "<b>" not in answer and "<script" not in answer
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp;lt;b&amp;gt;<br>&quot;quoted&quot;" in answer
    assert "&amp;lt;b&amp;gt;" in answer
    for semantic in ("stem", "marker", "ending"):
        assert f'class="morphology-{semantic}"' in answer
    # Unavailable-role prose must never become an asserted analysis.
    withheld_only = replace(comparison, roles=(comparison.roles[3],))
    withheld = next(
        c.answer
        for c in render_cards(parsed, selected_recipes=settings.selected_recipes, comparison=withheld_only)
        if c.eligible
    )
    assert 'class="morphology-stem"' not in withheld


@pytest.mark.parametrize("theme", ["muted", "monochrome"])
@pytest.mark.parametrize("appearance", ["light", "dark"])
@pytest.mark.parametrize("mode", ["static", "disclosure"])
def test_presentation_never_changes_reviewed_claims_or_generation_identity(
    theme: str, appearance: str, mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    original = profile().apply_overrides(
        {
            "source_identity": {"strategy": "source_id_field", "field": "ID"},
            "principal_parts": {"separators": [","]},
        }
    )
    effective = original.apply_overrides({"morphology": {"theme": theme, "appearance": appearance, "comparison": mode}})
    evidence = SourceExtraction("amāvī", ("perfect_1s",), "supported", (("amāvī",),))
    parsed = ParsedPrincipalParts(
        "amō", "amo", (PrincipalPartValue("perfect_1s", "amāvī", "amavi"),), source_identity="x", evidence=evidence
    )
    claim = Claim(
        "x:perfect_1s",
        "amāvī",
        "segmentation",
        "amā- | -v- | -ī",
        "manual/v1",
        (Evidence("grammar", "1", "v1", "Reviewed"),),
        original,
        "v2",
        extraction=evidence,
    )
    decision = review_claim(claim, status="accepted", reviewer="fixture", reason="Reviewed independently")
    changed_claim = replace(claim, profile=effective)
    assert changed_claim.fingerprint == claim.fingerprint
    reviewed = assess_claim(changed_claim, decision)
    assert reviewed.status == "accepted"
    result = compare_principal_parts(parsed, effective, (assess_claim(claim, decision),))
    assert result == compare_principal_parts(parsed, original, (assess_claim(claim, decision),))
    assert [(r.role, r.status, r.segmentation) for r in result.roles] == [
        ("present_1s", "absent", None),
        ("present_infinitive", "absent", None),
        ("perfect_1s", "present", "amā- | -v- | -ī"),
        ("supine", "absent", None),
    ]
    record = CanonicalSourceRecord(
        source_kind="csv",
        note_type="Latin",
        fields={"ID": "x", "Lemma": "amō", "Forms": "amō,amāre,amāvī,amātum"},
        provenance=SourceProvenance(source_path=Path("fixture.csv"), location="row 2"),
        source_identity="x",
    )
    base = generate_learning_object_notes((record,), original, source_scope="fixture").notes[0]
    generation = generate_learning_object_notes((record,), effective, source_scope="fixture")
    generated = generation.notes[0]
    assert generated.latinitas_id == base.latinitas_id
    assert generated.card_keys == base.card_keys
    assert len(generated.card_keys) == 6
    assert tuple(c.slot.semantic_key for c in generated.cards if c.eligible) == (
        "principal_part_completion:present_1s",
        "principal_part_completion:present_infinitive",
        "principal_part_completion:perfect_1s",
        "principal_part_recognition:present_1s",
        "principal_part_recognition:present_infinitive",
        "principal_part_recognition:perfect_1s",
    )
    assert [(c.slot.semantic_key, c.slot.ordinal, c.eligible) for c in generated.cards] == [
        (c.slot.semantic_key, c.slot.ordinal, c.eligible) for c in base.cards
    ]
    assert generated.content == base.content
    assert generated.personal_notes == base.personal_notes == ""
    export = PrincipalPartExportResult(
        source_path=Path("fixture.csv"), profile=effective, generation=generation, source_entry_count=1
    )
    payload = deterministic_csv_bytes(export).decode("utf-8")
    assert "Personal Notes" not in payload
    render_principal_part_preview(export, limit=1)
    preview = capsys.readouterr().out
    assert f"Morphology v1: {theme}/{appearance}/{mode}" in preview
    rows = list(csv.reader(io.StringIO(payload)))
    header = next(row for row in rows if row and row[0] == "#columns:LatinitasID")
    columns = ["LatinitasID", *header[1:]]
    values = rows[-1]
    for card in generated.cards:
        assert values[columns.index(card.slot.answer_field)] == card.answer
        if card.eligible:
            assert card.answer in preview
