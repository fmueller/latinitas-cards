from dataclasses import replace
from pathlib import Path

from latinitas_cards.generation import PrincipalPartGenerationResult, generate_principal_part_study_cards
from latinitas_cards.notes import GeneratedNote
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
    source_tags: tuple[str, ...] = (),
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
        source_tags=source_tags,
    )


def test_generated_notes_inherit_all_valid_parent_tags_additively_for_both_recipes() -> None:
    profile = _profile(tags=("latinitas", "latin"))
    records = (
        _record(
            "dīcere, dīcō, dīxī, dictum",
            source_identity="entry-a",
            profile=profile,
            source_tags=("latin", "verb::irregular", "Vokabeln-Übung", "latin"),
        ),
        _record("amāre, amō, amāvī, amātum", source_identity="entry-b", profile=profile, source_tags=("grammar",)),
        _record("ferre, ferō, tulī, lātum", source_identity="entry-c", profile=profile),
    )

    result = generate_principal_part_study_cards(records, profile)

    assert not result.skips
    expected_tags_by_parent = {
        "entry-a": ("latin", "verb::irregular", "Vokabeln-Übung", "latinitas"),
        "entry-b": ("grammar", "latinitas", "latin"),
        "entry-c": ("latinitas", "latin"),
    }
    notes_by_parent: dict[str, list[GeneratedNote]] = {}
    for note in result.notes:
        notes_by_parent.setdefault(note.provenance.source_identity or "", []).append(note)
    assert set(notes_by_parent) == set(expected_tags_by_parent)
    for parent_identity, expected_tags in expected_tags_by_parent.items():
        parent_notes = notes_by_parent[parent_identity]
        assert len(parent_notes) == 8
        assert {note.recipe.recipe_identity for note in parent_notes} == {
            "principal_part_completion",
            "principal_part_recognition",
        }
        assert all(note.content.tags == expected_tags for note in parent_notes)
        assert all(("Tags", " ".join(expected_tags)) in note.to_anki_fields() for note in parent_notes)


def test_tag_membership_and_order_changes_never_change_identities() -> None:
    profile = _profile(recipes=("principal_part_recognition",), tags=("configured",))
    before = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, dictum", source_identity="entry-a", profile=profile, source_tags=("a", "b")),),
        profile,
    )
    after = generate_principal_part_study_cards(
        (
            _record(
                "dīcere, dīcō, dīxī, dictum",
                source_identity="entry-a",
                profile=profile,
                source_tags=("c", "b", "a"),
            ),
        ),
        profile,
    )

    assert [note.latinitas_id for note in after.notes] == [note.latinitas_id for note in before.notes]
    assert [note.provenance.source_identity for note in after.notes] == [
        note.provenance.source_identity for note in before.notes
    ]
    assert {note.content.tags for note in after.notes} == {("c", "b", "a", "configured")}


def test_invalid_inherited_tags_skip_the_parent_with_actionable_diagnostics() -> None:
    profile = _profile()
    records = (
        _record(
            "dīcere, dīcō, dīxī, dictum",
            source_identity="entry-invalid",
            profile=profile,
            source_tags=("latin", "bad\x9btag"),
        ),
        _record("ferre, ferō, tulī, lātum", source_identity="entry-ok", profile=profile, source_tags=("grammar",)),
    )

    result = generate_principal_part_study_cards(records, profile)

    assert [note.provenance.source_identity for note in result.notes] == ["entry-ok"] * 8
    assert all(note.content.tags == ("grammar", "latinitas") for note in result.notes)
    invalid_skips = [skip for skip in result.skips if skip.code == "invalid_source_tags"]
    assert len(invalid_skips) == 1
    skip = invalid_skips[0]
    assert skip.status == "unsupported"
    assert skip.source_identity == "entry-invalid"
    assert skip.source_location == "row 2"
    assert "tag" in skip.message.lower()
    assert "position 2" in skip.message
    assert "bad" not in skip.message


def test_invalid_inherited_tag_whitespace_is_flagged_not_silently_split() -> None:
    profile = _profile(recipes=("principal_part_recognition",))

    result = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, dictum", source_identity="entry-tab", profile=profile, source_tags=("a\tb",)),),
        profile,
    )

    assert result.notes == ()
    skip = result.skips[0]
    assert skip.code == "invalid_source_tags"
    assert "whitespace" in skip.message or "control" in skip.message
    assert "a\tb" not in skip.message


def test_markup_unsafe_inherited_tags_are_skipped_without_live_markup() -> None:
    profile = _profile()
    records = (
        _record(
            "dīcere, dīcō, dīxī, dictum",
            source_identity="entry-hostile",
            profile=profile,
            source_tags=("<img/src=x/onerror=alert(1)>",),
        ),
        _record("ferre, ferō, tulī, lātum", source_identity="entry-amp", profile=profile, source_tags=("rock&roll",)),
        _record("amāre, amō, amāvī, amātum", source_identity="entry-ok", profile=profile, source_tags=("latin",)),
    )

    result = generate_principal_part_study_cards(records, profile)

    assert [note.provenance.source_identity for note in result.notes] == ["entry-ok"] * 8
    hostile_skips = [skip for skip in result.skips if skip.code == "invalid_source_tags"]
    assert len(hostile_skips) == 2
    assert {skip.source_identity for skip in hostile_skips} == {"entry-hostile", "entry-amp"}
    for skip in hostile_skips:
        assert "export" in skip.message.lower()
        assert "'&', '<', or '>'" in skip.message
    rendered = "\n".join(skip.message for skip in hostile_skips)
    assert "onerror" not in rendered
    assert "rock" not in rendered
    exported = "\n".join(str(tuple(note.to_anki_fields())) for note in result.notes)
    assert "<img" not in exported


def test_configured_tags_keep_existing_markup_semantics_for_trusted_input() -> None:
    profile = _profile(recipes=("principal_part_recognition",), tags=("rock&roll", "latinitas"))

    result = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, dictum", source_identity="entry-17", profile=profile),),
        profile,
    )

    assert all(note.content.tags == ("rock&roll", "latinitas") for note in result.notes)
    assert all(("Tags", "rock&roll latinitas") in note.to_anki_fields() for note in result.notes)


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


def test_source_html_meaning_renders_as_readable_text_with_line_boundaries() -> None:
    profile = _profile()
    result = generate_principal_part_study_cards(
        (
            _record(
                "dīcere, dīcō, dīxī, dictum",
                meaning="erste Bedeutung<div>zweite <b>Bedeutung</b></div>",
                profile=profile,
            ),
        ),
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
    expected = "<div><strong>Bedeutung:</strong> erste Bedeutung<br>zweite Bedeutung</div>"

    assert completion.content.prompt.endswith(expected)
    assert recognition.content.answer.endswith(expected)
    assert "&lt;div&gt;" not in completion.content.prompt
    assert "&lt;b&gt;" not in recognition.content.answer


def test_source_entities_in_meaning_decode_to_readable_text() -> None:
    profile = _profile()
    result = generate_principal_part_study_cards(
        (
            _record(
                "dīcere, dīcō, dīxī, dictum",
                meaning="sagen&nbsp;&amp;&nbsp;machen&lt;br&gt;prüfen",
                profile=profile,
            ),
        ),
        profile,
    )

    recognition = next(
        note
        for note in result.notes
        if note.recipe.recipe_identity == "principal_part_recognition" and note.recipe.exercise_key == "perfect_1s"
    )

    assert recognition.content.answer.endswith("<div><strong>Bedeutung:</strong> sagen &amp; machen<br>prüfen</div>")
    assert "&amp;nbsp;" not in recognition.content.answer
    assert "&lt;br&gt;" not in recognition.content.answer


def test_plain_text_source_meaning_renders_byte_identical() -> None:
    profile = _profile(recipes=("principal_part_recognition",))
    result = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, dictum", meaning="führen, dīcere — prüfen", profile=profile),),
        profile,
    )

    recognition = next(note for note in result.notes if note.recipe.exercise_key == "perfect_1s")

    assert recognition.content.answer.endswith("<div><strong>Bedeutung:</strong> führen, dīcere — prüfen</div>")


def test_lexical_and_principal_part_display_html_renders_as_readable_text() -> None:
    profile = _profile(recipes=("principal_part_recognition",))
    result = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, <b>dīctum</b>", lexical_entry="<i>dīcō</i>", profile=profile),),
        profile,
    )

    participle = next(note for note in result.notes if note.recipe.exercise_key == "perfect_passive_participle")

    assert participle.content.prompt == "Welche Stammform ist „dīctum“?"
    assert "<div><strong>Lemma:</strong> dīcō</div>" in participle.content.answer
    assert "&lt;i&gt;" not in participle.content.answer
    assert "&lt;b&gt;" not in participle.content.answer


def test_meaning_markup_normalization_keeps_identity_and_equivalent_content() -> None:
    profile = _profile(recipes=("principal_part_recognition",))
    plain = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, dictum", meaning="sagen", profile=profile),),
        profile,
    )
    marked = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, dictum", meaning="<div>sagen</div>", profile=profile),),
        profile,
    )

    assert {note.latinitas_id for note in marked.notes} == {note.latinitas_id for note in plain.notes}
    assert marked.notes == plain.notes


def test_entity_encoded_hostile_markup_in_meaning_is_removed() -> None:
    profile = _profile(recipes=("principal_part_recognition",))
    result = generate_principal_part_study_cards(
        (
            _record(
                "dīcere, dīcō, dīxī, dictum",
                meaning="sagen&lt;script&gt;alert(1)&lt;/script&gt;",
                profile=profile,
            ),
        ),
        profile,
    )

    recognition = next(note for note in result.notes if note.recipe.exercise_key == "perfect_1s")

    assert recognition.content.answer.endswith("<div><strong>Bedeutung:</strong> sagen</div>")
    assert "alert(1)" not in recognition.content.answer
    assert "script" not in recognition.content.answer


def test_untrusted_source_text_is_escaped_before_generated_markup() -> None:
    profile = _profile()
    result = generate_principal_part_study_cards(
        (
            _record(
                "<img src=x onerror=alert(1)>, <script>dīcō</script>, dīxī, dictum",
                lexical_entry="<script>alert(1)</script>",
                meaning="<b>sagen</b><script>steal()</script>",
                profile=profile,
            ),
        ),
        profile,
    )

    rendered = "\n".join(note.content.prompt + note.content.answer for note in result.notes)
    assert "<script>" not in rendered
    assert "<img" not in rendered
    assert "&lt;script&gt;" not in rendered
    assert "&lt;img" not in rendered
    assert "&lt;b&gt;" not in rendered
    assert "alert(1)" not in rendered
    assert "steal()" not in rendered
    assert "sagen" in rendered


def test_meaning_entities_decode_exactly_once_in_both_recipes() -> None:
    profile = _profile()
    result = generate_principal_part_study_cards(
        (_record("dīcere, dīcō, dīxī, dictum", meaning="a &amp;amp; b", profile=profile),),
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
    expected = "<div><strong>Bedeutung:</strong> a &amp;amp; b</div>"

    assert completion.content.prompt.endswith(expected)
    assert recognition.content.answer.endswith(expected)


def test_meaning_attribute_markup_with_quoted_greater_than_renders_text_only() -> None:
    profile = _profile()
    result = generate_principal_part_study_cards(
        (
            _record(
                "dīcere, dīcō, dīxī, dictum",
                meaning='<span title="a > b">sagen</span>',
                profile=profile,
            ),
        ),
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
    expected = "<div><strong>Bedeutung:</strong> sagen</div>"

    assert completion.content.prompt.endswith(expected)
    assert recognition.content.answer.endswith(expected)
    rendered = "\n".join(
        (completion.content.prompt, completion.content.answer, recognition.content.prompt, recognition.content.answer)
    )
    assert "title" not in rendered
    assert "&quot;&gt;" not in rendered


def test_encoded_comments_are_dropped_from_meaning_in_both_recipes() -> None:
    profile = _profile()
    result = generate_principal_part_study_cards(
        (
            _record(
                "dīcere, dīcō, dīxī, dictum",
                meaning="erste Bedeutung&lt;!-- versteckt --&gt; weitere Bedeutung",
                profile=profile,
            ),
        ),
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
    expected = "<div><strong>Bedeutung:</strong> erste Bedeutung weitere Bedeutung</div>"

    assert completion.content.prompt.endswith(expected)
    assert recognition.content.answer.endswith(expected)
    for content in (completion.content.prompt, recognition.content.answer):
        assert "versteckt" not in content
        assert "&lt;!--" not in content


def test_multiline_lexical_and_part_display_render_line_breaks_in_both_recipes() -> None:
    profile = _profile()
    result = generate_principal_part_study_cards(
        (
            _record(
                "dīcere, dīcō, dīxī<br>poet., dictum",
                lexical_entry="dīcō<div>alt</div>",
                profile=profile,
            ),
        ),
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

    assert "Welche Stammform ist „dīxī<br>poet.“?" in recognition.content.prompt
    assert "<div><strong>Lemma:</strong> dīcō<br>alt</div>" in recognition.content.answer
    assert "<strong>Perfekt, 1. Person Singular:</strong> dīxī<br>poet." in recognition.content.answer
    assert "<div><strong>Fehlende Stammform:</strong> dīxī<br>poet.</div>" in completion.content.answer
    assert "<strong>Perfekt, 1. Person Singular:</strong> _____" in completion.content.prompt


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
