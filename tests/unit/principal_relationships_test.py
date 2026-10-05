from dataclasses import replace
from pathlib import Path

import pytest

from latinitas_cards.claim_review import Claim, Evidence, assess_claim, review_claim
from latinitas_cards.principal_parts import ParsedPrincipalParts, PrincipalPartValue
from latinitas_cards.principal_relationships import compare_principal_parts
from latinitas_cards.profile import DeckProfile
from latinitas_cards.source_extraction import SourceExtraction


def _source_evidence(form: str, role: str = "perfect_1s") -> SourceExtraction:
    return SourceExtraction(form, (role,), "supported", ((form,),))


def test_only_bound_individually_reviewed_claims_enrich_the_tested_form() -> None:
    profile = DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms")
    parsed = ParsedPrincipalParts(
        "amō",
        "amo",
        (PrincipalPartValue("perfect_1s", "amāvī", "amavi"),),
        source_identity="entry-1",
        evidence=_source_evidence("amāvī"),
    )
    claim = Claim(
        "entry-1:perfect_1s",
        "amāvī",
        "segmentation",
        "amā- | -v- | -ī",
        "manual/v1",
        (Evidence("grammar", "v-perfect", "v1", "Reviewed v-perfect"),),
        profile,
        "confirmed/v1",
        extraction=parsed.evidence,
    )
    decision = review_claim(claim, status="accepted", reviewer="fixture", reason="Authored judgment")
    comparison = compare_principal_parts(parsed, profile, (assess_claim(claim, decision),))
    assert comparison.roles[2].segmentation == "amā- | -v- | -ī"
    assert comparison.roles[2].explanation is None
    assert comparison.roles[3].status == "absent"
    stale = compare_principal_parts(parsed, profile, (assess_claim(replace(claim, candidate_text="tulī"), decision),))
    assert stale.roles[2].segmentation is None
    assert stale.roles[2].claims[0].status == "withheld"
    assert "Current candidate" in stale.roles[2].claims[0].reason


def test_fourth_role_is_withheld_without_explicit_role_review() -> None:
    profile = DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms")
    parsed = ParsedPrincipalParts("amō", "amo", (PrincipalPartValue("supine", "amātum", "amatum"),))
    comparison = compare_principal_parts(parsed, profile)
    assert comparison.roles[3].status == "withheld"
    assert "PPP" not in (comparison.roles[3].explanation or "")


def test_generation_withholds_unreviewed_fourth_role_but_preserves_siblings() -> None:
    from latinitas_cards.generation import generate_learning_object_notes
    from latinitas_cards.profile import SourceIdentityConfig
    from latinitas_cards.sources import CanonicalSourceRecord, SourceProvenance

    profile = DeckProfile.default(
        note_type="Latin",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="ID"),
        separators=(",",),
    )
    record = CanonicalSourceRecord(
        source_kind="csv",
        note_type="Latin",
        fields={"ID": "entry-1", "Lemma": "amō", "Forms": "amō, amāre, amāvī, amātum"},
        provenance=SourceProvenance(source_path=Path("fixture.csv"), location="row 2"),
        source_identity="entry-1",
    )
    result = generate_learning_object_notes((record,), profile, source_scope="fixture")
    assert len(result.notes[0].card_keys) == 6
    assert not any(key.endswith(":supine") for key in result.notes[0].card_keys)
    assert "Stammformen vergleichen" in result.notes[0].cards[0].answer


@pytest.mark.parametrize(
    ("form", "split", "prose"),
    [
        ("amāvī", "amā- | -v- | -ī", "Das -v- markiert hier das Perfekt."),
        ("amāvī", "amāv- | -ī", "Nur Perfektstamm und Endung sind belegt."),
        ("tulī", "", "ferre → tulī → lātum: verschiedene Stämme, keine Zeichenableitung."),
    ],
)
def test_reviewed_detailed_coarse_and_irregular_claims_stay_independent(form: str, split: str, prose: str) -> None:
    profile = DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms")
    parsed = ParsedPrincipalParts(
        "ferō",
        "fero",
        (PrincipalPartValue("perfect_1s", form, form),),
        source_identity="x",
        evidence=_source_evidence(form),
    )
    explanation = Claim(
        "x:perfect_1s",
        form,
        "explanation",
        prose,
        "reviewed-fixture/v1",
        (Evidence("authored fixture", "stipulated judgment", "v1", prose),),
        profile,
        "v1",
        extraction=parsed.evidence,
    )
    claims = [explanation]
    if split:
        claims.append(replace(explanation, kind="segmentation", value=split))
    assessments = tuple(
        assess_claim(c, review_claim(c, status="accepted", reviewer="fixture", reason="Independent expectation"))
        for c in claims
    )
    result = compare_principal_parts(parsed, profile, assessments).roles[2]
    assert result.segmentation == (split or None)
    assert result.explanation == prose
    assert compare_principal_parts(parsed, profile).roles[2].explanation is None
    from latinitas_cards.cards import render_cards

    cards = render_cards(
        parsed,
        selected_recipes=("principal_part_recognition",),
        comparison=compare_principal_parts(parsed, profile, assessments),
    )
    answer = next(card.answer for card in cards if card.eligible)
    assert prose in answer
    assert "Weitere Stämme: Analyse zurückgehalten" in answer
    assert all(prose not in card.prompt for card in cards)


@pytest.mark.parametrize("role", ["perfect_passive_participle", "supine"])
def test_fourth_label_acceptance_is_explicit_and_profile_changes_invalidate_it(role: str) -> None:
    profile = DeckProfile.default(
        note_type="Latin",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        principal_part_roles=("present_1s", "present_infinitive", "perfect_1s", role),
    )
    parsed = ParsedPrincipalParts(
        "amō",
        "amo",
        (PrincipalPartValue(role, "amātum", "amatum"),),
        source_identity="x",
        evidence=_source_evidence("amātum", role),
    )
    claim = Claim(
        "x:" + role,
        "amātum",
        "label",
        role,
        "manual/v1",
        (Evidence("fixture", "fourth-role context", "v1", "Explicit context, not suffix"),),
        profile,
        "v1",
        extraction=parsed.evidence,
    )
    assessment = assess_claim(
        claim, review_claim(claim, status="accepted", reviewer="fixture", reason="Context checked")
    )
    assert compare_principal_parts(parsed, profile, (assessment,)).roles[3].status == "present"
    changed = profile.apply_overrides({"principal_parts": {"separators": [";"]}})
    assert compare_principal_parts(parsed, changed, (assessment,)).roles[3].status == "withheld"


def test_deponent_and_ambiguous_forms_do_not_acquire_unreviewed_derivations() -> None:
    profile = DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms")
    parsed = ParsedPrincipalParts(
        "loquor",
        "loquor",
        (
            PrincipalPartValue("present_1s", "loquor", "loquor"),
            PrincipalPartValue("perfect_1s", "locūtus sum|locutus sum", "locutus sum", unresolved=True),
            PrincipalPartValue("supine", None, None),
        ),
    )
    result = compare_principal_parts(parsed, profile)
    assert result.roles[0].segmentation is None
    assert result.roles[2].status == "withheld"
    assert result.roles[3].status == "absent"
    assert all(role.explanation is None for role in result.roles)


@pytest.mark.parametrize("recipe", ["principal_part_completion", "principal_part_recognition"])
def test_preview_export_review_handoff_rechecks_scope_evidence_and_conflicts(tmp_path: Path, recipe: str) -> None:
    import json
    from html import unescape

    from latinitas_cards.generation import SINGLE_LEXEME_OBJECT_KEY
    from latinitas_cards.identity import derive_latinitas_id
    from latinitas_cards.preview_export import (
        deterministic_csv_bytes,
        prepare_principal_part_export,
        write_principal_part_csv,
    )
    from latinitas_cards.principal_parts import PrincipalPartParseSuccess, parse_principal_parts
    from latinitas_cards.profile import SourceIdentityConfig
    from latinitas_cards.sources import read_source_records

    source = tmp_path / "source.csv"
    source.write_text('ID,Lemma,Forms\nx,amō,"amō, amāre, amāvī, amātum"\n', encoding="utf-8")
    profile = DeckProfile.default(
        note_type="Latin",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        separators=(",",),
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="ID"),
        selected_recipes=(recipe,),
    )
    initial = prepare_principal_part_export(source, profile, approve_new_scope=True)
    write_principal_part_csv(initial, tmp_path / "initial.csv")
    parsed = parse_principal_parts(read_source_records(source, source_id_field="ID")[0], profile)
    assert isinstance(parsed, PrincipalPartParseSuccess)
    identity = derive_latinitas_id("x", SINGLE_LEXEME_OBJECT_KEY, source_scope=initial.source_scope)
    label = Claim(
        identity + ":supine",
        "amātum",
        "label",
        "supine",
        "authored-fixture/v1",
        (Evidence("fixture", "explicit supine context", "v1", "Stipulated supine; not inferred from -um"),),
        profile,
        "v1",
        extraction=parsed.evidence,
    )
    accepted = assess_claim(label, review_claim(label, status="accepted", reviewer="fixture", reason="Context checked"))
    reviewed = prepare_principal_part_export(source, profile, claim_assessments=(accepted,))
    from latinitas_cards.generation import generate_learning_object_notes

    generation = reviewed.generation
    assert len(generation.notes[0].card_keys) == 4
    payload = generation.notes[0].content.principal_parts.split('class="principal-part-review">')[1].split("</span>")[0]
    evidence = json.loads(unescape(payload))
    assert evidence["roles"][3]["claims"][0]["claim"]["evidence"][0]["reference"] == "explicit supine context"
    assert evidence["roles"][3]["status"] == "present"
    other_scope = generate_learning_object_notes(
        read_source_records(source, source_id_field="ID"), profile, source_scope="other", claim_assessments=(accepted,)
    )
    assert len(other_scope.notes[0].card_keys) == 3
    conflict = replace(label, value="perfect_passive_participle")
    conflict_review = assess_claim(
        conflict, review_claim(conflict, status="accepted", reviewer="fixture", reason="Conflict")
    )
    conflicting = generate_learning_object_notes(
        read_source_records(source, source_id_field="ID"),
        profile,
        source_scope=initial.source_scope,
        claim_assessments=(accepted, conflict_review),
    )
    assert len(conflicting.notes[0].card_keys) == 3
    serialized = deterministic_csv_bytes(replace(reviewed, generation=conflicting))
    assert b"principal-part-review" in serialized
    assert "Zurückgehalten" in serialized.decode()


def test_withholding_a_split_cannot_be_overridden_by_an_older_acceptance() -> None:
    profile = DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms")
    parsed = ParsedPrincipalParts(
        "amō",
        "amo",
        (PrincipalPartValue("perfect_1s", "amāvī", "amavi"),),
        source_identity="x",
        evidence=_source_evidence("amāvī"),
    )
    claim = Claim(
        "x:perfect_1s",
        "amāvī",
        "segmentation",
        "amā- | -v- | -ī",
        "manual/v1",
        (Evidence("fixture", "v-perfect", "v1", "Stipulated detailed split"),),
        profile,
        "v1",
        extraction=parsed.evidence,
    )
    accepted = assess_claim(claim, review_claim(claim, status="accepted", reviewer="fixture", reason="Initial review"))
    withheld = assess_claim(
        claim, review_claim(claim, status="withheld", reviewer="fixture", reason="Evidence disputed")
    )
    result = compare_principal_parts(parsed, profile, (accepted, withheld)).roles[2]
    assert result.segmentation is None
    assert result.status == "present"


def test_incompatible_fourth_role_profile_requires_explicit_resolution() -> None:
    profile = DeckProfile.default(
        note_type="Latin",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        principal_part_roles=("present_1s", "present_infinitive", "perfect_1s", "supine", "perfect_passive_participle"),
    )
    with pytest.raises(ValueError, match="resolve.*PPP.*supine"):
        compare_principal_parts(ParsedPrincipalParts("amō", "amo", ()), profile)


@pytest.mark.parametrize("conflicting_kind", ["segmentation", "explanation"])
def test_analysis_conflicts_do_not_suppress_the_independently_accepted_kind(conflicting_kind: str) -> None:
    profile = DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms")
    parsed = ParsedPrincipalParts(
        "amō",
        "amo",
        (PrincipalPartValue("perfect_1s", "amāvī", "amavi"),),
        source_identity="x",
        evidence=_source_evidence("amāvī"),
    )
    split = Claim(
        "x:perfect_1s",
        "amāvī",
        "segmentation",
        "amā- | -v- | -ī",
        "manual/v1",
        (Evidence("fixture", "v-perfect", "v1", "Stipulated detailed split"),),
        profile,
        "v1",
        extraction=parsed.evidence,
    )
    explanation = replace(split, kind="explanation", value="Das -v- markiert hier das Perfekt.")
    conflicting = replace(
        split if conflicting_kind == "segmentation" else explanation,
        value="amāv- | -ī" if conflicting_kind == "segmentation" else "Nur grobe Stammgrenzen belegt.",
    )
    assessments = tuple(
        assess_claim(c, review_claim(c, status="accepted", reviewer="fixture", reason="Independent review"))
        for c in (split, explanation, conflicting)
    )
    result = compare_principal_parts(parsed, profile, assessments).roles[2]
    assert result.status == "present"
    assert result.segmentation == (None if conflicting_kind == "segmentation" else "amā- | -v- | -ī")
    assert result.explanation == (None if conflicting_kind == "explanation" else "Das -v- markiert hier das Perfekt.")


def test_review_without_any_extraction_snapshot_cannot_assert_a_split() -> None:
    profile = DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms")
    parsed = ParsedPrincipalParts(
        "amō", "amo", (PrincipalPartValue("perfect_1s", "amāvī", "amavi"),), source_identity="x"
    )
    claim = Claim(
        "x:perfect_1s",
        "amāvī",
        "segmentation",
        "amā- | -v- | -ī",
        "manual/v1",
        (Evidence("fixture", "v-perfect", "v1", "Grammar citation without source snapshot"),),
        profile,
        "v1",
    )
    accepted = assess_claim(claim, review_claim(claim, status="accepted", reviewer="fixture", reason="Initial review"))
    result = compare_principal_parts(parsed, profile, (accepted,)).roles[2]
    assert result.segmentation is None
    assert result.claims[0].status == "withheld"
