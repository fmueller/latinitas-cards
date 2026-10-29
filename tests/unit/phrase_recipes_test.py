"""Phrase recipe contract: explicit context, not inferred disambiguation."""

import pytest
from pydantic import ValidationError

from latinitas_cards.claim_review import ReviewDecision, review_claim
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
    assert proposed["answer"].startswith(
        "Proposed reference — not asserted\nLesart 1: mit einem Freund / mit dem Freund\n"
    )
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
    english = profile.apply_overrides({"language_tag": "en", "morphology": {"terminology": {"language": "user"}}})
    english_card = render_exercise(phrase, in_english(selected), english, accept(phrase, in_english(selected), english))
    assert english_card["prompt"].endswith("[object]: Case.")
    assert "Case: Ablative" in english_card["answer"] and "Number: Singular" in english_card["answer"]
    assert english_card["semantic_key"] == reviewed["semantic_key"]
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


ENGLISH = PROFILE.apply_overrides({"language_tag": "en-US", "morphology": {"terminology": {"language": "user"}}})
FRENCH_LATIN = PROFILE.apply_overrides({"language_tag": "fr", "morphology": {"terminology": {"language": "latin"}}})


def accept(phrase: dict[str, object], selected: Exercise, profile: DeckProfile) -> ReviewDecision:
    claim = contextual_claim(phrase, selected, profile)
    return review_claim(claim, status="accepted", reviewer="Latin teacher", reason="Checked")


def in_english(selected: Exercise) -> Exercise:
    data = selected.reference.model_dump(mode="json")
    data["language"] = "en"
    data["readings"][0].update(
        translations=["with a friend"],
        contributions={"object": "Accompaniment; cum takes the ablative."},
        explanation="The preposition cum governs the case here.",
    )
    return selected.model_copy(update={"reference": Reference.model_validate(data)})


@pytest.mark.parametrize("recipe", ["phrase_translation", "phrase_recognition"])
def test_english_cards_need_their_own_accepted_english_reference(recipe: str) -> None:
    phrase = candidate()
    german = exercise(recipe)
    english = in_english(german)
    # A German reference, even accepted under the English profile, never reaches an English card.
    leaked = render_exercise(phrase, german, ENGLISH, accept(phrase, german, ENGLISH))
    assert (leaked["review_state"], leaked["enabled"]) == ("withheld", "")
    assert "'de'" in leaked["reason"] and "'en'" in leaked["reason"]
    for text in ("Freund", "cum regiert", "Präposition"):
        assert text not in leaked["answer"] + leaked["answer_html"]
    proposed = render_exercise(phrase, english, ENGLISH)
    assert (proposed["review_state"], proposed["enabled"]) == ("proposed", "")
    assert proposed["answer"].startswith("Proposed reference — not asserted\nReading 1: with a friend\n")
    assert (
        contextual_claim(phrase, english, ENGLISH).fingerprint != contextual_claim(phrase, german, ENGLISH).fingerprint
    )
    reviewed = render_exercise(phrase, english, ENGLISH, accept(phrase, english, ENGLISH))
    assert (reviewed["review_state"], reviewed["enabled"]) == ("reviewed", "1")
    assert reviewed["semantic_key"] == german.semantic_key
    assert reviewed["answer"].startswith("Reviewed reference\nReading 1: with a friend\n")
    if recipe == "phrase_translation":
        assert reviewed["prompt"] == "cum amico\nTranslate, then compare with the reference."
        assert reviewed["answer"].endswith("\nSelf-assessment: compare the meaning, not the exact wording.")
    else:
        assert reviewed["prompt"] == "cum amico\nIdentify amico [object]: Case."
        assert "\nCase: Ablative\nNumber: Singular\nGender: Masculine\n" in reviewed["answer"]
    for text in ("Lesart", "Übersetze", "Bestimme", "Selbsteinschätzung"):
        assert text not in reviewed["prompt"] + reviewed["answer"]
    # An English reference never appears on a German card either.
    swapped = render_exercise(phrase, english, PROFILE, accept(phrase, english, PROFILE))
    assert swapped["answer"] == (
        "Withheld: Reference language 'en' differs from card language 'de'\nProposed reference — not asserted\n"
    )
    # Tags other than en keep German, so a German reference stays valid on a French-tagged profile.
    french = render_exercise(phrase, german, FRENCH_LATIN, accept(phrase, german, FRENCH_LATIN))
    assert french["review_state"] == "reviewed" and "Lesart 1: mit einem Freund" in french["answer"]


def test_english_analysis_prompt_and_latin_terminology() -> None:
    selected = in_english(exercise(distinction="analysis"))
    assert render_exercise(candidate(), selected, ENGLISH)["prompt"].endswith("[object]: Analysis.")
    latin = ENGLISH.apply_overrides({"morphology": {"terminology": {"language": "latin"}}})
    assert render_exercise(candidate(), in_english(exercise()), latin)["prompt"].endswith(
        "Identify amico [object]: Casus."
    )
    assert render_exercise(candidate(), exercise(distinction="analysis"), PROFILE)["prompt"].endswith(": Analyse.")


def test_reference_language_names_supported_tags() -> None:
    data = exercise().reference.model_dump(mode="json") | {"language": "fr"}
    with pytest.raises(ValidationError, match="'de' or 'en'"):
        Reference.model_validate(data)


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
    # A language mismatch adds its own reason without hiding the unresolved distinction.
    both = render_exercise(phrase, selected, ENGLISH)
    assert both["reason"] == (
        "The tested distinction is unresolved in the supplied contextual readings. "
        "Reference language 'de' differs from card language 'en'"
    )
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
