"""Contract tests for conditional sibling cards on learning-object notes.

External expectations are written as literal data instead of being derived from
the production helpers, so a drift in registry order, slot naming, ownership or
eligibility cannot silently re-authorize itself.
"""

import hashlib
import json
from pathlib import Path

from latinitas_cards.cards import (
    TEMPLATE_REGISTRY,
    TEMPLATE_REGISTRY_VERSION,
    render_cards,
    slot_for_recipe_role,
)
from latinitas_cards.generation import generate_learning_object_notes
from latinitas_cards.notes import (
    AUTHORITATIVE_NOTE_FIELDS,
    CSV_EXPORT_FIELD_NAMES,
    GENERATED_NOTE_FIELD_NAMES,
    TAGS_CSV_COLUMN,
    GeneratedNote,
    GeneratedNoteProvenance,
    GenerationMetadata,
    ManagedNoteContent,
)
from latinitas_cards.principal_parts import ParsedPrincipalParts, PrincipalPartValue
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig
from latinitas_cards.sources import CanonicalSourceRecord, SourceProvenance

EXPECTED_REGISTRY = (
    ("principal_part_completion:present_1s", "Completion Present", 0),
    ("principal_part_completion:present_infinitive", "Completion Infinitive", 1),
    ("principal_part_completion:perfect_1s", "Completion Perfect", 2),
    ("principal_part_completion:perfect_passive_participle", "Completion PPP", 3),
    ("principal_part_completion:supine", "Completion Supine", 4),
    ("principal_part_recognition:present_1s", "Recognition Present", 5),
    ("principal_part_recognition:present_infinitive", "Recognition Infinitive", 6),
    ("principal_part_recognition:perfect_1s", "Recognition Perfect", 7),
    ("principal_part_recognition:perfect_passive_participle", "Recognition PPP", 8),
    ("principal_part_recognition:supine", "Recognition Supine", 9),
)

EXPECTED_REGISTRY_ROLES = (
    "present_1s",
    "present_infinitive",
    "perfect_1s",
    "perfect_passive_participle",
    "supine",
)

EXPECTED_RECIPES = ("principal_part_completion", "principal_part_recognition")

EXPECTED_CARD_FIELD_NAMES = (
    "CompletionPresentEnabled",
    "CompletionPresentPrompt",
    "CompletionPresentAnswer",
    "CompletionInfinitiveEnabled",
    "CompletionInfinitivePrompt",
    "CompletionInfinitiveAnswer",
    "CompletionPerfectEnabled",
    "CompletionPerfectPrompt",
    "CompletionPerfectAnswer",
    "CompletionPPPEnabled",
    "CompletionPPPPrompt",
    "CompletionPPPAnswer",
    "CompletionSupineEnabled",
    "CompletionSupinePrompt",
    "CompletionSupineAnswer",
    "RecognitionPresentEnabled",
    "RecognitionPresentPrompt",
    "RecognitionPresentAnswer",
    "RecognitionInfinitiveEnabled",
    "RecognitionInfinitivePrompt",
    "RecognitionInfinitiveAnswer",
    "RecognitionPerfectEnabled",
    "RecognitionPerfectPrompt",
    "RecognitionPerfectAnswer",
    "RecognitionPPPEnabled",
    "RecognitionPPPPrompt",
    "RecognitionPPPAnswer",
    "RecognitionSupineEnabled",
    "RecognitionSupinePrompt",
    "RecognitionSupineAnswer",
)

EXPECTED_BASE_FIELD_NAMES = (
    "LatinitasID",
    "Lemma",
    "Principal Parts",
    "Meaning",
    "Tags",
    "Source ID",
    "Source Scope",
    "Source Kind",
    "Source Location",
    "Source Path",
    "Note Schema",
    "Generator",
    "Profile",
)


def _profile(
    *,
    roles: tuple[str, ...] = ("present_1s", "present_infinitive", "perfect_1s", "supine"),
    separators: tuple[str, ...] = (" — ",),
    recipes: tuple[str, ...] = ("principal_part_completion", "principal_part_recognition"),
) -> DeckProfile:
    return DeckProfile.default(
        note_type="Latin vocabulary",
        lexical_entry_field="Lemma",
        principal_parts_field="Principal parts",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="Source ID"),
        principal_part_roles=roles,
        separators=separators,
        selected_recipes=recipes,
    )


def _record(
    principal_parts: str,
    *,
    lexical_entry: str = "ferō",
    meaning: str = "tragen",
    source_identity: str = "entry-1",
    profile: DeckProfile | None = None,
) -> CanonicalSourceRecord:
    resolved = profile or _profile()
    return CanonicalSourceRecord(
        source_kind="csv",
        note_type=resolved.note_type,
        fields={
            "Source ID": source_identity,
            resolved.fields.lexical_entry_field: lexical_entry,
            resolved.fields.principal_parts_field: principal_parts,
            "German gloss": meaning,
        },
        provenance=SourceProvenance(source_path=Path("fixture.csv"), location="row 2", row_number=2),
        source_identity=source_identity,
    )


def _generate_one(principal_parts: str, profile: DeckProfile, *, meaning: str = "tragen") -> GeneratedNote:
    result = generate_learning_object_notes(
        (_record(principal_parts, profile=profile, meaning=meaning),),
        profile,
        source_scope="scope-cards",
    )
    assert len(result.notes) == 1
    return result.notes[0]


def test_template_registry_freezes_slots_for_both_recipes() -> None:
    assert [(slot.semantic_key, slot.template_name, slot.ordinal) for slot in TEMPLATE_REGISTRY] == [
        (key, name, ordinal) for key, name, ordinal in EXPECTED_REGISTRY
    ]
    for slot in TEMPLATE_REGISTRY:
        short_role = slot.semantic_key.split(":")[1]
        assert slot.recipe in {"principal_part_completion", "principal_part_recognition"}
        assert slot.role == short_role
    completion_present = slot_for_recipe_role("principal_part_completion", "present_1s")
    assert completion_present is not None
    assert completion_present.enabled_field == "CompletionPresentEnabled"
    assert completion_present.prompt_field == "CompletionPresentPrompt"
    assert completion_present.answer_field == "CompletionPresentAnswer"
    assert slot_for_recipe_role("principal_part_completion", "form_x") is None
    assert slot_for_recipe_role("unknown_recipe", "present_1s") is None


def test_template_registry_digest_is_independently_reproducible() -> None:
    from latinitas_cards.cards import TEMPLATE_REGISTRY_DIGEST

    independent = hashlib.sha256(
        json.dumps(
            [
                {
                    "key": key,
                    "recipe": key.split(":")[0],
                    "role": key.split(":")[1],
                    "template": name,
                    "ordinal": ordinal,
                }
                for key, name, ordinal in EXPECTED_REGISTRY
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert TEMPLATE_REGISTRY_VERSION == 1
    assert f"card-registry-sha256:{independent}" == TEMPLATE_REGISTRY_DIGEST


def test_supported_recipes_and_card_keys_bind_to_the_profile_and_identity_authorities() -> None:
    from typing import get_args

    from latinitas_cards.cards import SUPPORTED_RECIPES
    from latinitas_cards.identity import derive_card_semantic_key
    from latinitas_cards.profile import RecipeName

    assert SUPPORTED_RECIPES == EXPECTED_RECIPES
    assert set(SUPPORTED_RECIPES) == set(get_args(RecipeName))
    assert {slot.semantic_key for slot in TEMPLATE_REGISTRY} == {
        derive_card_semantic_key(recipe, role) for recipe in EXPECTED_RECIPES for role in EXPECTED_REGISTRY_ROLES
    }


def test_authoritative_schema_extends_card_fields_without_repurposing_existing_slots() -> None:
    field_names = tuple(field.name for field in AUTHORITATIVE_NOTE_FIELDS)
    assert field_names == EXPECTED_BASE_FIELD_NAMES + EXPECTED_CARD_FIELD_NAMES + ("Personal Notes",)
    ownership = {field.name: field.ownership for field in AUTHORITATIVE_NOTE_FIELDS}
    assert ownership["LatinitasID"] == "managed"
    assert ownership["Tags"] == "transport"
    assert ownership["Personal Notes"] == "personal"
    assert all(ownership[name] == "managed" for name in EXPECTED_CARD_FIELD_NAMES)
    exported = tuple(field.name for field in AUTHORITATIVE_NOTE_FIELDS if field.exported)
    assert exported == CSV_EXPORT_FIELD_NAMES
    assert "Personal Notes" not in CSV_EXPORT_FIELD_NAMES
    assert TAGS_CSV_COLUMN == 5
    assert GENERATED_NOTE_FIELD_NAMES[: len(EXPECTED_BASE_FIELD_NAMES)] == EXPECTED_BASE_FIELD_NAMES


def test_card_keys_follow_registry_order_not_profile_order() -> None:
    reversed_profile = _profile(
        roles=("supine", "perfect_1s", "present_infinitive", "present_1s"),
        recipes=("principal_part_recognition", "principal_part_completion"),
    )

    note = _generate_one("lātum — tulī — ferre — ferō", reversed_profile)

    assert note.card_keys == (
        "principal_part_completion:present_1s",
        "principal_part_completion:present_infinitive",
        "principal_part_completion:perfect_1s",
        "principal_part_recognition:present_1s",
        "principal_part_recognition:present_infinitive",
        "principal_part_recognition:perfect_1s",
    )


def test_deselected_recipes_and_unsupported_roles_never_become_eligible_cards() -> None:
    recognition_only = _profile(recipes=("principal_part_recognition",))
    unsupported_roles = _profile(roles=("stem_a", "stem_b"), separators=(" — ",))

    note = _generate_one("ferō — ferre — tulī — lātum", recognition_only)
    unsupported = _generate_one("ferō — ferre", unsupported_roles)

    assert note.card_keys == tuple(key for key in note.card_keys if key.startswith("principal_part_recognition:"))
    eligible_guards = {card.slot.enabled_field: card.guard for card in note.cards if card.eligible}
    assert all(card.guard == "" for card in note.cards if not card.eligible)
    assert eligible_guards
    assert unsupported.card_keys == ()
    assert all(card.guard == "" and card.prompt == "" and card.answer == "" for card in unsupported.cards)


def test_explicit_omissions_guard_cards_without_shifting_positions() -> None:
    profile = _profile(roles=("present_1s", "present_infinitive", "perfect_1s", "perfect_passive_participle"))

    note = _generate_one("ferō — ferre —  — ", profile)

    eligible = {card.slot.semantic_key for card in note.cards if card.eligible}
    assert eligible == {
        "principal_part_completion:present_1s",
        "principal_part_completion:present_infinitive",
        "principal_part_recognition:present_1s",
        "principal_part_recognition:present_infinitive",
    }
    ppp_cards = [card for card in note.cards if card.slot.role == "perfect_passive_participle"]
    assert len(ppp_cards) == 2
    assert all(card.guard == "" and card.prompt == "" and card.answer == "" for card in ppp_cards)


def test_ppp_and_supine_remain_distinct_slots() -> None:
    profile = _profile(roles=("present_1s", "present_infinitive", "perfect_1s", "perfect_passive_participle"))
    supine_profile = _profile(roles=("present_1s", "present_infinitive", "perfect_1s", "supine"))

    ppp_note = _generate_one("ferō — ferre — tulī — lātum", profile)
    supine_note = _generate_one("ferō — ferre — tulī — lātum", supine_profile)

    ppp_key = "principal_part_completion:perfect_passive_participle"
    ppp_completion = next(card for card in ppp_note.cards if card.slot.semantic_key == ppp_key)
    supine_completion = next(
        card for card in supine_note.cards if card.slot.semantic_key == "principal_part_completion:supine"
    )
    assert not ppp_completion.eligible and not supine_completion.eligible
    assert ppp_completion.prompt == supine_completion.prompt == ""
    assert ppp_completion.slot.template_name == "Completion PPP"
    assert supine_completion.slot.template_name == "Completion Supine"


def test_completion_requires_meaningful_remaining_context_and_an_answer() -> None:
    parts = (
        PrincipalPartValue(role="present_1s", display="ferō", comparison="fero", raw="ferō"),
        PrincipalPartValue(role="perfect_1s", display=None, comparison=None, raw=""),
    )
    parsed = ParsedPrincipalParts(lexical_entry="ferō", lexical_entry_comparison="fero", parts=parts)

    cards = render_cards(parsed, selected_recipes=("principal_part_completion", "principal_part_recognition"))

    by_key = {card.slot.semantic_key: card for card in cards}
    assert by_key["principal_part_completion:perfect_1s"].eligible is False
    assert by_key["principal_part_recognition:perfect_1s"].eligible is False
    assert by_key["principal_part_completion:present_1s"].eligible is False
    assert by_key["principal_part_recognition:present_1s"].eligible is True


def test_recognition_prompt_uses_target_and_lemma_and_gloss_is_never_invented() -> None:
    profile = _profile()
    with_gloss = _generate_one("ferō — ferre — tulī — lātum", profile, meaning="tragen")
    without_gloss = _generate_one("ferō — ferre — tulī — lātum", profile, meaning="")

    recognition = next(
        card for card in with_gloss.cards if card.slot.semantic_key == "principal_part_recognition:perfect_1s"
    )
    recognition_bare = next(
        card for card in without_gloss.cards if card.slot.semantic_key == "principal_part_recognition:perfect_1s"
    )
    completion = next(card for card in with_gloss.cards if card.slot.semantic_key == "principal_part_completion:supine")

    assert recognition.eligible
    assert "tulī" in recognition.prompt
    assert "ferō" in recognition.prompt
    assert "tragen" in recognition.prompt
    assert recognition_bare.eligible
    assert "tragen" not in recognition_bare.prompt
    assert not completion.eligible
    assert completion.prompt == completion.answer == ""


def test_card_prompts_render_the_series_with_the_target_blanked() -> None:
    note = _generate_one("ferō — ferre — tulī — lātum", _profile())

    completion = next(card for card in note.cards if card.slot.semantic_key == "principal_part_completion:perfect_1s")

    assert "ferō" in completion.prompt
    assert "ferre" in completion.prompt
    assert "lātum" not in completion.prompt
    assert "unresolved source evidence" in completion.prompt
    assert "tulī" not in completion.prompt
    assert "____" in completion.prompt
    assert completion.answer.startswith("tulī<div")


def test_card_content_is_escaped_and_guards_are_binary() -> None:
    profile = _profile()

    note = _generate_one("ferō — ferre — tulī &amp; poët. — lātum", profile)

    eligible = [card for card in note.cards if card.eligible]
    assert {card.guard for card in eligible} == {"1"}
    assert all("<b>" not in card.prompt and "<b>" not in card.answer for card in note.cards)
    assert any("tulī &amp; poët." in card.prompt for card in eligible)


def test_note_identity_is_independent_of_recipe_selection_and_eligibility() -> None:
    profile = _profile()
    reduced = _profile(recipes=("principal_part_recognition",))
    degraded = _profile()

    baseline = _generate_one("ferō — ferre — tulī — lātum", profile)
    fewer_recipes = _generate_one("ferō — ferre — tulī — lātum", reduced)
    lost_card = _generate_one("ferō — ferre —  — lātum", degraded)

    assert fewer_recipes.latinitas_id == baseline.latinitas_id
    assert lost_card.latinitas_id == baseline.latinitas_id
    assert len(lost_card.card_keys) < len(baseline.card_keys)


def test_generated_note_exposes_card_fields_matching_the_authoritative_schema() -> None:
    note = _generate_one("ferō — ferre — tulī — lātum", _profile())

    fields = dict(note.to_anki_fields())
    card_field_values = {name: value for name, value in fields.items() if name in EXPECTED_CARD_FIELD_NAMES}
    assert set(card_field_values) == set(EXPECTED_CARD_FIELD_NAMES)
    assert tuple(fields) == GENERATED_NOTE_FIELD_NAMES
    assert all(
        card_field_values[slot.enabled_field] == ("1" if slot.semantic_key in note.card_keys else "")
        for slot in TEMPLATE_REGISTRY
    )
    assert all(
        card_field_values[slot.prompt_field] and card_field_values[slot.answer_field]
        for slot in TEMPLATE_REGISTRY
        if slot.semantic_key in note.card_keys
    )


def test_cards_survive_managed_content_and_metadata_updates() -> None:
    note = _generate_one("ferō — ferre — tulī — lātum", _profile())
    provenance = GeneratedNoteProvenance(
        source_kind="csv",
        location="row 2",
        source_identity="entry-1",
        source_scope="scope-x",
    )
    rebuilt = GeneratedNote.create(
        source_identity="entry-1",
        source_scope="scope-x",
        provenance=provenance,
        object_key=note.object_key,
        metadata=GenerationMetadata(profile_digest="profile-sha256:other"),
        content=ManagedNoteContent(lemma="ferō", principal_parts="ferō — ferre", meaning="tragen"),
        cards=note.cards,
    )

    assert rebuilt.cards == note.cards
    assert rebuilt.card_keys == note.card_keys
