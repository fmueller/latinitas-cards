"""Phrase recipe contract: explicit context, not inferred disambiguation."""

import pytest

from latinitas_cards.claim_review import review_claim
from latinitas_cards.phrase_recipes import (
    PHRASE_LAYOUT,
    PHRASE_LAYOUT_FIELDS,
    PHRASE_LAYOUT_VERSION,
    Exercise,
    RecipePlan,
    Reference,
    contextual_claim,
    render_exercise,
)
from latinitas_cards.profile import DeckProfile, TerminologySettings

PROFILE = DeckProfile.default(note_type="Vocabulary", lexical_entry_field="Word", principal_parts_field="Parts")


def candidate(text: str = "cum amico") -> dict[str, object]:
    return {
        "candidate_key": "phrase-object-v1-example",
        "text": text,
        "constituents": [{"role": "object", "form": {"text": "amico"}}],
        "input_claims": [{"fingerprint": "lexical-evidence"}],
    }


def exercise(recipe: str = "phrase_recognition", **updates: object) -> Exercise:
    data: dict[str, object] = {
        "candidate_key": "phrase-object-v1-example",
        "recipe": recipe,
        "target": "object" if recipe == "phrase_recognition" else None,
        "distinction": "case" if recipe == "phrase_recognition" else "meaning",
        "reference": {
            "readings": [
                {
                    "translations": ["mit einem Freund", "mit dem Freund"],
                    "features": {"object": {"case": "ablative", "number": "singular", "gender": "masculine"}},
                    "contributions": {"object": "Begleitung; cum regiert den Ablativ."},
                    "explanation": "Die Präposition cum bestimmt hier den Kasus.",
                }
            ],
            "evidence": [{"source": "fixture", "reference": "cum", "version": "1", "text": "reviewed context"}],
        },
    }
    return Exercise.model_validate(data | updates)


def test_proposed_and_reviewed_references_and_terminology() -> None:
    selected = exercise()
    profile = PROFILE
    phrase = candidate()
    proposed = render_exercise(phrase, selected, profile)
    assert proposed["review_state"] == "proposed"
    assert "Proposed" in proposed["answer"]
    assert "amico" in proposed["prompt"] and "Casus" in proposed["prompt"]
    claim = contextual_claim(phrase, selected, profile)
    decision = review_claim(claim, status="accepted", reviewer="Latin teacher", reason="Checked context")
    reviewed = render_exercise(phrase, selected, profile, decision)
    assert reviewed["review_state"] == "reviewed"
    assert "Casus: ablativus" in reviewed["answer"]
    assert "Numerus: singularis" in reviewed["answer"]
    assert "mit dem Freund" in reviewed["answer"]
    assert "cum regiert" in reviewed["answer"]
    german = profile.model_copy(
        update={
            "morphology": profile.morphology.model_copy(update={"terminology": TerminologySettings(language="user")})
        }
    )
    localized = render_exercise(phrase, selected, german, decision)
    assert "Kasus: Ablativ" in localized["answer"]
    assert localized["semantic_key"] == reviewed["semantic_key"]
    assert contextual_claim(phrase, selected, german).fingerprint == claim.fingerprint
    stale = render_exercise(candidate("ad amico"), selected, profile, decision)
    assert stale["review_state"] == "withheld" and "Stale" in stale["reason"]


def test_translation_is_latin_first_self_assessment_not_grading() -> None:
    selected = exercise("phrase_translation")
    result = render_exercise(candidate(), selected, PROFILE)
    assert result["prompt"].startswith("cum amico")
    assert "Freund" not in result["prompt"]
    assert "Selbsteinschätzung" in result["answer"]
    assert "mit einem Freund" in result["answer"]
    assert result["grading"] == "self-assessment"


def test_ambiguous_readings_retained_single_case_withheld() -> None:
    selected = exercise()
    data = selected.reference.model_dump(mode="json")
    reading = data["readings"][0]
    reading["translations"] = ["gute Geschenke"]
    reading["explanation"] = "Ohne Satzkontext bleiben mehrere Kasus möglich."
    reading["contributions"] = {"object": "Neutrales Plural mit Adjektivkongruenz."}
    data["readings"] = [
        reading | {"features": {"object": {"case": case, "number": "plural", "gender": "neuter"}}}
        for case in ("nominative", "accusative", "vocative")
    ]
    selected = selected.model_copy(update={"reference": Reference.model_validate(data)})
    phrase = candidate("dona bona")
    phrase["constituents"] = [{"role": "object", "form": {"text": "dona"}}]
    withheld = render_exercise(phrase, selected, PROFILE)
    assert withheld["review_state"] == "withheld"
    assert "unresolved" in withheld["reason"]
    assert len(withheld["readings"]) == 3
    assert "nominativus" in withheld["answer"] and "accusativus" in withheld["answer"]
    assert withheld["enabled"] == ""
    alternatives = selected.model_copy(update={"distinction": "analysis"})
    claim = contextual_claim(phrase, alternatives, PROFILE)
    assert len(claim.alternatives) == 3
    assert '"case":"vocative"' in claim.alternatives[2]
    decision = review_claim(claim, status="accepted", reviewer="Teacher", reason="All contextual alternatives checked")
    assert render_exercise(phrase, alternatives, PROFILE, decision)["review_state"] == "reviewed"


def test_target_distinction_keys_and_layout_compatibility() -> None:
    from latinitas_cards.cards import TEMPLATE_REGISTRY
    from latinitas_cards.reference_templates import REFERENCE_CARD_TEMPLATES, REFERENCE_NOTE_TYPE_FIELDS

    subject = exercise(target="subject")
    obj = exercise(target="object")
    assert subject.semantic_key != obj.semantic_key
    assert obj.semantic_key != obj.model_copy(update={"distinction": "number"}).semantic_key
    assert (
        obj.semantic_key == obj.model_copy(update={"reference": exercise("phrase_translation").reference}).semantic_key
    )
    assert PHRASE_LAYOUT_VERSION == 1
    assert tuple(t.slot for t in PHRASE_LAYOUT[:10]) == TEMPLATE_REGISTRY
    assert PHRASE_LAYOUT[:10] == REFERENCE_CARD_TEMPLATES
    assert [t.slot.ordinal for t in PHRASE_LAYOUT[-2:]] == [10, 11]
    assert PHRASE_LAYOUT_FIELDS[: len(REFERENCE_NOTE_TYPE_FIELDS)] == REFERENCE_NOTE_TYPE_FIELDS


def test_explicit_plan_limits_and_invalid_targets() -> None:
    selected = exercise()
    with pytest.raises(ValueError):
        RecipePlan(version=1, max_exercises=1, exercises=(selected, exercise("phrase_translation")))
    with pytest.raises(ValueError):
        exercise("both")
    with pytest.raises(ValueError):
        exercise("phrase_translation", target="object")
    with pytest.raises(ValueError):
        render_exercise(candidate(), exercise(target="subject"), PROFILE)


def test_isolated_possibilities_are_not_contextual_evidence() -> None:
    isolated = exercise().reference.model_dump(mode="json")
    isolated["readings"] = [
        isolated["readings"][0] | {"features": {"object": {"case": case, "number": "singular", "gender": "masculine"}}}
        for case in ("dative", "ablative")
    ]
    ambiguous = exercise(reference=isolated)
    assert render_exercise(candidate("amico"), ambiguous, PROFILE)["review_state"] == "withheld"
    contextual = render_exercise(candidate(), exercise(), PROFILE)
    assert "dativus" not in contextual["answer"] and "ablativus" in contextual["answer"]
    # liber can address the book (vocative) or name it (nominative). The noun
    # amici remains genitive singular here; its isolated nominative plural is
    # NOT evidence for agreement with singular liber, nor a contextual reading.
    phrase = candidate("liber amici")
    phrase["constituents"] = [
        {"role": role, "form": {"text": text}} for role, text in (("head", "liber"), ("dependent", "amici"))
    ]
    reference = {
        "readings": [
            {
                "translations": [translation],
                "explanation": explanation,
                "features": {
                    "head": {"case": case, "number": "singular", "gender": "masculine"},
                    "dependent": {"case": "genitive", "number": "singular", "gender": "masculine"},
                },
                "contributions": {
                    "head": "Benennung oder Anrede des Buches.",
                    "dependent": "Zugehörigkeit zum Freund.",
                },
            }
            for case, translation, explanation in (
                ("nominative", "das Buch des Freundes", "Benennung des Buches."),
                ("vocative", "o Buch des Freundes", "Anrede des Buches; kurzer Kontext entscheidet nicht."),
            )
        ],
        "evidence": exercise().reference.model_dump(mode="json")["evidence"],
    }
    selected = exercise(target="dependent", reference=reference)
    rendered = render_exercise(phrase, selected, PROFILE)
    assert "genetivus" in rendered["answer"] and "pluralis" not in rendered["answer"]
    assert "o Buch" in rendered["answer"] and len(rendered["readings"]) == 2
    assert render_exercise(phrase, exercise(target="head", reference=reference), PROFILE)["enabled"] == ""


def test_clause_targets_all_features_and_html_safety() -> None:
    phrase = candidate("puella amicum videt")
    phrase["constituents"] = [
        {"role": role, "form": {"text": text}}
        for role, text in (("subject", "puella"), ("object", "amicum"), ("verb", "videt"))
    ]
    reference = exercise().reference.model_dump(mode="json")
    reference["readings"] = [
        {
            "translations": ["Das Mädchen sieht einen Freund."],
            "explanation": "Das Subjekt sieht das Objekt. <script>unsafe</script>",
            "features": {
                "subject": {"case": "nominative", "number": "singular", "gender": "feminine"},
                "object": {"case": "accusative", "number": "singular", "gender": "masculine"},
                "verb": {
                    "person": "3",
                    "number": "singular",
                    "tense": "present",
                    "mood": "indicative",
                    "voice": "active",
                },
            },
            "contributions": {"subject": "Subjekt", "object": "Direktes Objekt", "verb": "Prädikat der Wahrnehmung"},
        }
    ]
    subject = render_exercise(phrase, exercise(target="subject", reference=reference), PROFILE)
    obj = render_exercise(phrase, exercise(reference=reference), PROFILE)
    assert subject["semantic_key"] != obj["semantic_key"]
    assert "puella [subject]" in subject["prompt"] and "amicum [object]" in obj["prompt"]
    selected = exercise(target="verb", distinction="tense", reference=reference)
    latin = render_exercise(phrase, selected, PROFILE)
    for expected in ("Persona: tertia", "Numerus: singularis", "Tempus: praesens", "Modus: indicativus", "Vox: activa"):
        assert expected in latin["answer"]
    assert "<script>" not in latin["answer_html"] and "&lt;script&gt;" in latin["answer_html"]
