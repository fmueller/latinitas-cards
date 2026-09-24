from dataclasses import replace
from pathlib import Path

from latinitas_cards.generation import PrincipalPartGenerationResult, generate_principal_part_study_cards
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig
from latinitas_cards.sources import CanonicalSourceRecord, SourceProvenance

UNIVERSITY_ROLES = (
    "present_infinitive",
    "present_1s",
    "perfect_1s",
    "perfect_passive_participle",
)


def _profile(
    *,
    roles: tuple[str, ...] = UNIVERSITY_ROLES,
    separators: tuple[str, ...] = (",",),
    recipes: tuple[str, ...] = ("principal_part_completion", "principal_part_recognition"),
    tags: tuple[str, ...] = ("latinitas",),
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
        tags=tags,
    )


def _record(
    principal_parts: str,
    *,
    lexical_entry: str = "dīcō",
    meaning: str = "sagen",
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
            "German gloss": meaning,
        },
        provenance=SourceProvenance(source_path=Path("fixture.csv"), location="row 2", row_number=2),
        source_identity=source_identity,
    )


def test_completion_uses_only_confirmed_recipe_and_answers_with_form_and_role() -> None:
    profile = _profile(recipes=("principal_part_completion",), tags=("custom-principal-parts",))

    result = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, dictum", profile=profile),),
        profile,
    )

    assert isinstance(result, PrincipalPartGenerationResult)
    assert not result.skips
    assert len(result.notes) == 4
    assert {note.recipe.recipe_identity for note in result.notes} == {"principal_part_completion"}
    assert {note.recipe.exercise_key for note in result.notes} == set(UNIVERSITY_ROLES)
    assert all(note.provenance.source_identity == "entry-17" for note in result.notes)
    assert all(note.provenance.location == "row 2" for note in result.notes)
    assert all(note.provenance.source_path is None for note in result.notes)
    assert all(note.content.tags == ("custom-principal-parts",) for note in result.notes)
    assert all(("Tags", "custom-principal-parts") in note.to_anki_fields() for note in result.notes)

    perfect_note = next(note for note in result.notes if note.recipe.exercise_key == "perfect_1s")
    assert "Ergänze die fehlende Stammform." in perfect_note.content.prompt
    assert "Stammformen" in perfect_note.content.prompt
    assert "Bedeutung" in perfect_note.content.prompt
    assert "dīxī" in perfect_note.content.answer
    assert "Perfekt, 1. Person Singular" in perfect_note.content.answer
    assert "Welche Stammform" not in perfect_note.content.prompt


def test_generated_completion_and_recognition_html_keep_section_boundaries() -> None:
    profile = _profile()
    result = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, dictum", profile=profile),),
        profile,
    )

    completion = next(
        note
        for note in result.notes
        if note.recipe.recipe_identity == "principal_part_completion" and note.recipe.exercise_key == "perfect_1s"
    )
    recognition = next(
        note
        for note in result.notes
        if note.recipe.recipe_identity == "principal_part_recognition" and note.recipe.exercise_key == "perfect_1s"
    )

    assert completion.content.prompt == (
        "<div>Ergänze die fehlende Stammform.</div>"
        "<div><strong>Stammformen</strong></div>"
        "<div><strong>Infinitiv:</strong> dīcere<br>"
        "<strong>Präsens, 1. Person Singular:</strong> dīcō<br>"
        "<strong>Perfekt, 1. Person Singular:</strong> _____<br>"
        "<strong>Partizip Perfekt Passiv (PPP):</strong> dictum</div>"
        "<div><strong>Bedeutung:</strong> sagen</div>"
    )
    assert completion.content.answer == (
        "<div><strong>Fehlende Stammform:</strong> dīxī</div>"
        "<div><strong>Rolle:</strong> Perfekt, 1. Person Singular</div>"
    )
    assert recognition.content.answer == (
        "<div><strong>Lemma:</strong> dīcō</div>"
        "<div><strong>Stammformen</strong></div>"
        "<div><strong>Infinitiv:</strong> dīcere<br>"
        "<strong>Präsens, 1. Person Singular:</strong> dīcō<br>"
        "<strong>Perfekt, 1. Person Singular:</strong> dīxī<br>"
        "<strong>Partizip Perfekt Passiv (PPP):</strong> dictum</div>"
        "<div><strong>Rolle:</strong> Perfekt, 1. Person Singular</div>"
        "<div><strong>Bedeutung:</strong> sagen</div>"
    )


def test_recognition_maps_each_latin_form_to_lemma_full_parts_role_and_gloss() -> None:
    profile = _profile(recipes=("principal_part_recognition",))

    result = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, dictum", profile=profile),),
        profile,
    )

    assert not result.skips
    assert len(result.notes) == 4
    perfect_note = next(note for note in result.notes if note.recipe.exercise_key == "perfect_1s")
    assert perfect_note.content.prompt.startswith("Welche Stammform ist „dīxī“?")
    assert "Lemma" in perfect_note.content.answer
    assert "dīcō" in perfect_note.content.answer
    assert "Stammformen" in perfect_note.content.answer
    assert "dīcere" in perfect_note.content.answer
    assert "dīxī" in perfect_note.content.answer
    assert "dictum" in perfect_note.content.answer
    assert "Rolle" in perfect_note.content.answer
    assert "Perfekt, 1. Person Singular" in perfect_note.content.answer
    assert "Bedeutung" in perfect_note.content.answer
    assert "sagen" in perfect_note.content.answer


def test_generation_preserves_old_supine_profiles_and_position_specific_layouts() -> None:
    old_profile = _profile(
        roles=("present_1s", "present_infinitive", "perfect_1s", "supine"),
        separators=(" — ",),
        recipes=("principal_part_recognition",),
    )
    old_result = generate_principal_part_study_cards(
        (_record("ferō — ferre — tulī — lātum", lexical_entry="ferō", profile=old_profile),),
        old_profile,
    )

    assert not old_result.skips
    supine_note = next(note for note in old_result.notes if note.recipe.exercise_key == "supine")
    assert "Supinum" in supine_note.content.answer
    assert "Partizip Perfekt Passiv (PPP)" not in supine_note.content.answer

    position_profile = _profile(
        separators=(", ", "; ", " / "),
        recipes=("principal_part_completion",),
    )
    position_result = generate_principal_part_study_cards(
        (_record("dīcere, dīcō; dīxī / dictum", profile=position_profile),),
        position_profile,
    )

    assert not position_result.skips
    assert {note.recipe.exercise_key for note in position_result.notes} == set(UNIVERSITY_ROLES)
    assert any("Partizip Perfekt Passiv (PPP)" in note.content.answer for note in position_result.notes)


def test_explicit_omissions_skip_only_the_unavailable_role_without_shifting_roles() -> None:
    profile = _profile(
        roles=("present_1s", "present_infinitive", "perfect_1s", "supine"),
        separators=(" — ",),
    )

    result = generate_principal_part_study_cards(
        (_record("sum — esse — fuī — ", lexical_entry="sum", profile=profile),),
        profile,
    )

    assert len(result.notes) == 6
    assert {note.recipe.exercise_key for note in result.notes} == {"present_1s", "present_infinitive", "perfect_1s"}
    assert {skip.code for skip in result.skips} == {"omitted_principal_part"}
    assert {skip.recipe_identity for skip in result.skips} == {
        "principal_part_completion",
        "principal_part_recognition",
    }
    assert all(skip.source_identity == "entry-17" for skip in result.skips)


def test_parser_failures_become_structured_skips_without_guessed_cards() -> None:
    profile = _profile()
    records = (
        _record("", source_identity="incomplete"),
        _record("dīcere — dīcō — dīxī — dictum", source_identity="unsupported"),
        _record("dīcō, dīcere, dīxī", source_identity="ambiguous"),
    )

    result = generate_principal_part_study_cards(records, profile)

    assert result.notes == ()
    assert {skip.source_identity for skip in result.skips} == {"incomplete", "unsupported", "ambiguous"}
    assert {skip.code for skip in result.skips} == {
        "missing_principal_parts",
        "separator_mismatch",
        "unmarked_omission",
    }
    assert all("dīcere" not in skip.message for skip in result.skips)


def test_wording_and_gloss_changes_retain_identity_but_change_managed_content() -> None:
    profile = _profile(recipes=("principal_part_recognition",))
    original = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, dictum", profile=profile),),
        profile,
    )
    revised = generate_principal_part_study_cards(
        (_record("dicere, dico, dixi, dictum", meaning="sagen; aussprechen", profile=profile),),
        profile,
    )

    original_by_key = {note.recipe.exercise_key: note for note in original.notes}
    revised_by_key = {note.recipe.exercise_key: note for note in revised.notes}
    assert set(original_by_key) == set(revised_by_key)
    for key in original_by_key:
        assert revised_by_key[key].latinitas_id == original_by_key[key].latinitas_id
    assert revised_by_key["perfect_1s"].content != original_by_key["perfect_1s"].content


def test_recipe_and_source_semantic_keys_never_collide() -> None:
    profile = _profile()
    result = generate_principal_part_study_cards(
        (
            _record("dīcere, dīcō, dīxī, dictum", source_identity="entry-a", profile=profile),
            _record("ferre, ferō, tulī, lātum", lexical_entry="ferō", source_identity="entry-b", profile=profile),
        ),
        profile,
    )

    assert len(result.notes) == 16
    assert len({note.latinitas_id for note in result.notes}) == len(result.notes)
    semantic_keys = {
        (
            note.provenance.source_identity,
            note.recipe.recipe_identity,
            note.recipe.exercise_key,
        )
        for note in result.notes
    }
    assert len(semantic_keys) == 16


def test_manifest_identity_is_used_for_note_identity_and_provenance() -> None:
    profile = _profile(recipes=("principal_part_completion",)).apply_overrides(
        {"source_identity": {"strategy": "manifest"}}
    )
    record = _record("dīcere, dīcō, dīxī, dictum", profile=profile)
    record = replace(record, source_identity=None)
    other_note_type = replace(record, note_type="Other note type")

    result = generate_principal_part_study_cards(
        (other_note_type, record),
        profile,
        manifest_identities=("ignored", "manifest-17"),
    )

    assert [skip.code for skip in result.skips] == ["note_type_mismatch"]
    assert result.notes
    assert all(note.provenance.source_identity == "manifest-17" for note in result.notes)


def test_untrusted_source_text_is_escaped_before_generated_markup() -> None:
    profile = _profile(recipes=("principal_part_recognition",))
    result = generate_principal_part_study_cards(
        (
            _record(
                "<img src=x onerror=alert(1)>, <script>dīcō</script>, dīxī, dictum",
                lexical_entry="<script>alert(1)</script>",
                meaning="<b>sagen</b>",
                profile=profile,
            ),
        ),
        profile,
    )

    rendered = "\n".join(note.content.prompt + note.content.answer for note in result.notes)
    assert "<script>" not in rendered
    assert "<img" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;img" in rendered
    assert "&lt;b&gt;sagen&lt;/b&gt;" in rendered


def test_generated_provenance_does_not_expose_an_absolute_source_path() -> None:
    profile = _profile(recipes=("principal_part_recognition",))
    record = _record("dīcere, dīcō, dīxī, dictum", profile=profile)
    record = replace(
        record,
        provenance=replace(record.provenance, source_path=Path("/home/alice/private/decks/source.csv")),
    )

    result = generate_principal_part_study_cards((record,), profile)

    assert result.notes
    assert all(note.provenance.source_path is None for note in result.notes)
    assert all(("Source Path", "") in note.to_anki_fields() for note in result.notes)
    assert all("alice" not in field_value for note in result.notes for _, field_value in note.to_anki_fields())
