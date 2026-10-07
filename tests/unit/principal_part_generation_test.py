from dataclasses import replace
from html import unescape
from pathlib import Path
from typing import Any

import pytest

from latinitas_cards.generation import (
    SINGLE_LEXEME_OBJECT_KEY,
    LearningObjectGenerationResult,
    generate_learning_object_notes,
)
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig
from latinitas_cards.sources import CanonicalSourceRecord, SourceProvenance

UNIVERSITY_ROLES = (
    "present_infinitive",
    "present_1s",
    "perfect_1s",
    "perfect_passive_participle",
)


def _generate(
    records: tuple[CanonicalSourceRecord, ...],
    profile: DeckProfile,
    **kwargs: Any,
) -> LearningObjectGenerationResult:
    if profile.source_identity.strategy != "manifest":
        kwargs.setdefault("source_scope", "scope-generation")
    return generate_learning_object_notes(records, profile, **kwargs)


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


def test_one_source_note_yields_one_learning_object_note_with_knowledge_and_card_keys() -> None:
    profile = _profile()

    result = _generate((_record("dīcere, dīcō, dīxī, dictum", profile=profile),), profile)

    assert isinstance(result, LearningObjectGenerationResult)
    assert [skip.code for skip in result.skips] == ["linguistic_review_required"]
    assert len(result.notes) == 1
    note = result.notes[0]
    assert note.object_key == SINGLE_LEXEME_OBJECT_KEY
    assert note.content.lemma == "dīcō"
    assert "<strong>Infinitiv:</strong> dīcere" in note.content.principal_parts
    assert note.content.meaning == "sagen"
    assert note.provenance.source_identity == "entry-17"
    assert note.provenance.location == "row 2"
    assert note.provenance.source_path is None


def test_card_keys_cover_selected_recipes_and_available_roles_only() -> None:
    both = _generate((_record("dīcere, dīcō, dīxī, dictum", profile=_profile()),), _profile())
    recognition_only = _generate(
        (_record("dīcere, dīcō, dīxī, dictum", profile=_profile()),),
        _profile(recipes=("principal_part_recognition",)),
    )

    assert both.notes[0].card_keys == (
        "principal_part_completion:present_1s",
        "principal_part_completion:present_infinitive",
        "principal_part_completion:perfect_1s",
        "principal_part_recognition:present_1s",
        "principal_part_recognition:present_infinitive",
        "principal_part_recognition:perfect_1s",
    )
    assert recognition_only.notes[0].card_keys == tuple(
        key for key in both.notes[0].card_keys if key.startswith("principal_part_recognition:")
    )
    assert all("latinitas" not in key for key in both.notes[0].card_keys)


def test_distinct_sources_stay_distinct_objects_even_for_related_words() -> None:
    profile = _profile()
    result = _generate(
        (
            _record("ferre, ferō, tulī, lātum", lexical_entry="ferō", source_identity="entry-fero", profile=profile),
            _record("esse, sum, fuī, ", lexical_entry="sum", source_identity="entry-sum", profile=profile),
        ),
        profile,
    )

    assert len(result.notes) == 2
    assert len({note.latinitas_id for note in result.notes}) == 2
    assert {note.provenance.source_identity for note in result.notes} == {"entry-fero", "entry-sum"}


def test_wording_gloss_and_recipe_changes_enrich_the_same_note_identity() -> None:
    profile = _profile()
    original = _generate((_record("dīcere, dīcō, dīxī, dictum", profile=profile),), profile)
    revised = _generate(
        (_record("dicere, dico, dixi, dictum", meaning="sagen; aussprechen", profile=profile),), profile
    )
    enriched = _generate(
        (_record("dīcere, dīcō, dīxī, dictum", profile=profile),),
        _profile(recipes=("principal_part_recognition", "principal_part_completion")),
    )

    assert revised.notes[0].latinitas_id == original.notes[0].latinitas_id
    assert revised.notes[0].content != original.notes[0].content
    assert enriched.notes[0].latinitas_id == original.notes[0].latinitas_id
    assert len(enriched.notes[0].card_keys) == 6


def test_generated_notes_inherit_all_valid_parent_tags_additively() -> None:
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

    result = _generate(records, profile)

    assert [skip.code for skip in result.skips] == ["linguistic_review_required"] * 3
    expected_tags_by_parent = {
        "entry-a": ("latin", "verb::irregular", "Vokabeln-Übung", "latinitas"),
        "entry-b": ("grammar", "latinitas", "latin"),
        "entry-c": ("latinitas", "latin"),
    }
    assert {note.provenance.source_identity: note.content.tags for note in result.notes} == expected_tags_by_parent
    assert all(("Tags", " ".join(note.content.tags)) in note.to_anki_fields() for note in result.notes)


def test_tag_membership_and_order_changes_never_change_identities() -> None:
    profile = _profile(tags=("configured",))
    before = _generate(
        (_record("dīcere, dīcō, dīxī, dictum", source_identity="entry-a", profile=profile, source_tags=("a", "b")),),
        profile,
    )
    after = _generate(
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
    assert after.notes[0].content.tags == ("c", "b", "a", "configured")


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

    result = _generate(records, profile)

    assert [note.provenance.source_identity for note in result.notes] == ["entry-ok"]
    assert result.notes[0].content.tags == ("grammar", "latinitas")
    invalid_skips = [skip for skip in result.skips if skip.code == "invalid_source_tags"]
    assert len(invalid_skips) == 1
    skip = invalid_skips[0]
    assert skip.status == "unsupported"
    assert skip.source_identity == "entry-invalid"
    assert skip.source_location == "row 2"
    assert "tag" in skip.message.lower()
    assert "position 2" in skip.message
    assert "bad" not in skip.message


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

    result = _generate(records, profile)

    assert [note.provenance.source_identity for note in result.notes] == ["entry-ok"]
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


# Tab-separated lemmas are intentionally not listed: HTML text extraction
# normalizes tabs to spaces, so after normalization they are indistinguishable
# from legitimate multi-word lemmas, and discriminating them would require new
# linguistic rules that v0.1.0 explicitly excludes (v0.2.0 input).
@pytest.mark.parametrize(
    "lexical_entry",
    ["ferō\nsum", "ferō — sum", "ferō - sum", "ferō / sum", "ferō, sum", "ferō; sum", "ferō|sum"],
)
def test_multi_object_lexical_entries_are_reported_for_review_not_merged(lexical_entry: str) -> None:
    profile = _profile()

    result = _generate(
        (
            _record(
                "dīcere, dīcō, dīxī, dictum",
                lexical_entry=lexical_entry,
                source_identity="entry-multi",
                profile=profile,
            ),
            _record("ferre, ferō, tulī, lātum", lexical_entry="ferō", source_identity="entry-single", profile=profile),
            _record(
                "amāre, amō, amāvī, amātum",
                lexical_entry="amāre|amare",
                source_identity="entry-variant",
                profile=profile,
            ),
        ),
        profile,
    )

    assert {note.provenance.source_identity for note in result.notes} == {"entry-single", "entry-variant"}
    multi_skip = next(skip for skip in result.skips if skip.code == "multi_object_source")
    assert multi_skip.status == "ambiguous"
    assert multi_skip.source_identity == "entry-multi"
    assert "coherent" in multi_skip.message or "object" in multi_skip.message


def test_multiline_single_lexeme_display_variants_stay_one_object() -> None:
    profile = _profile()

    result = _generate(
        (
            _record(
                "dīcere, dīcō, dīxī, dictum",
                lexical_entry="dīcō\ndīcō",
                source_identity="entry-multiline-variant",
                profile=profile,
            ),
        ),
        profile,
    )

    assert len(result.notes) == 1
    assert result.notes[0].provenance.source_identity == "entry-multiline-variant"
    assert result.notes[0].content.lemma == "dīcō<br>dīcō"


def test_explicit_omissions_keep_the_object_note_and_report_the_omitted_roles_once() -> None:
    profile = _profile(
        roles=("present_1s", "present_infinitive", "perfect_1s", "supine"),
        separators=(" — ",),
    )

    result = _generate(
        (_record("sum — esse — fuī — ", lexical_entry="sum", profile=profile),),
        profile,
    )

    assert len(result.notes) == 1
    note = result.notes[0]
    assert note.provenance.source_identity == "entry-17"
    assert note.card_keys == (
        "principal_part_completion:present_1s",
        "principal_part_completion:present_infinitive",
        "principal_part_completion:perfect_1s",
        "principal_part_recognition:present_1s",
        "principal_part_recognition:present_infinitive",
        "principal_part_recognition:perfect_1s",
    )
    omission_skips = [skip for skip in result.skips if skip.code == "omitted_principal_part"]
    assert len(omission_skips) == 1
    assert omission_skips[0].status == "incomplete"
    assert omission_skips[0].source_identity == "entry-17"
    assert "supine" in omission_skips[0].message


def test_parser_failures_become_structured_skips_without_guessed_objects() -> None:
    profile = _profile()
    records = (
        _record("", source_identity="incomplete"),
        _record("dīcere — dīcō — dīxī — dictum", source_identity="unsupported"),
        _record("dīcō, dīcere, dīxī", source_identity="ambiguous"),
    )

    result = _generate(records, profile)

    assert result.notes == ()
    assert {skip.source_identity for skip in result.skips} == {"incomplete", "unsupported", "ambiguous"}
    assert {skip.code for skip in result.skips} == {
        "missing_principal_parts",
        "separator_mismatch",
        "unmarked_omission",
    }
    assert all("dīcere" not in skip.message for skip in result.skips)


def test_manifest_identity_and_scope_are_used_for_note_identity_and_provenance() -> None:
    profile = _profile().apply_overrides({"source_identity": {"strategy": "manifest"}})
    record = _record("dīcere, dīcō, dīxī, dictum", profile=profile)
    record = replace(record, source_identity=None)
    other_note_type = replace(record, note_type="Other note type")

    result = _generate(
        (other_note_type, record),
        profile,
        manifest_identities=("ignored", "manifest-17"),
        source_scope="scope-alpha",
    )

    assert [skip.code for skip in result.skips] == ["note_type_mismatch", "linguistic_review_required"]
    assert result.notes
    assert all(note.provenance.source_identity == "manifest-17" for note in result.notes)
    assert all(note.provenance.source_scope == "scope-alpha" for note in result.notes)


def test_duplicate_source_identities_become_collision_skips() -> None:
    profile = _profile()

    result = _generate(
        (
            _record("dīcere, dīcō, dīxī, dictum", source_identity="entry-dup", profile=profile),
            _record("ferre, ferō, tulī, lātum", source_identity="entry-dup", profile=profile),
        ),
        profile,
    )

    assert result.notes == ()
    assert {skip.code for skip in result.skips} == {"duplicate_source_identity"}
    assert all(skip.status == "collision" for skip in result.skips)


def test_source_html_knowledge_renders_as_safe_readable_text() -> None:
    profile = _profile()
    result = _generate(
        (
            _record(
                "dīcere, <i>dīcō</i>, dīxī, dictum",
                lexical_entry="<b>dīcō</b>",
                meaning="erste Bedeutung<div>zweite <b>Bedeutung</b>&lt;br&gt;dritte</div>"
                "<script>alert(1)</script><img src=x onerror=alert(2)>",
                profile=profile,
            ),
        ),
        profile,
    )

    note = result.notes[0]
    rendered = note.content.lemma + note.content.principal_parts + note.content.meaning

    assert note.content.lemma == "dīcō"
    assert "<strong>Präsens, 1. Person Singular:</strong> dīcō" in note.content.principal_parts
    assert note.content.meaning == "erste Bedeutung<br>zweite Bedeutung<br>dritte"
    assert "&lt;div&gt;" not in rendered
    assert "&lt;b&gt;" not in rendered
    assert "&lt;br&gt;" not in rendered
    assert "alert(1)" not in rendered
    assert "alert(2)" not in rendered
    assert "<script" not in rendered
    assert "<img" not in rendered


def test_source_entities_in_meaning_decode_exactly_once() -> None:
    profile = _profile()
    result = _generate(
        (
            _record(
                "dīcere, dīcō, dīxī, dictum",
                meaning="sagen&nbsp;&amp;&nbsp;machen&lt;br&gt;prüfen",
                profile=profile,
            ),
        ),
        profile,
    )

    assert result.notes[0].content.meaning == "sagen &amp; machen<br>prüfen"


def test_encoded_comments_are_dropped_from_meaning() -> None:
    profile = _profile()
    result = _generate(
        (
            _record(
                "dīcere, dīcō, dīxī, dictum",
                meaning="erste Bedeutung&lt;!-- versteckt --&gt; weitere Bedeutung",
                profile=profile,
            ),
        ),
        profile,
    )

    meaning = result.notes[0].content.meaning
    assert meaning == "erste Bedeutung weitere Bedeutung"
    assert "versteckt" not in meaning


def test_multiline_lexical_and_part_display_render_line_breaks() -> None:
    profile = _profile()
    result = _generate(
        (
            _record(
                "dīcere, dīcō, dīxī<br>poet., dictum",
                lexical_entry="dīcō<div>dīcō</div>",
                profile=profile,
            ),
        ),
        profile,
    )

    note = result.notes[0]
    assert note.content.lemma == "dīcō<br>dīcō"
    assert "<strong>Perfekt, 1. Person Singular:</strong> dīxī<br>poet." in note.content.principal_parts


def test_generated_provenance_does_not_expose_an_absolute_source_path() -> None:
    profile = _profile()
    record = _record("dīcere, dīcō, dīxī, dictum", profile=profile)
    record = replace(
        record,
        provenance=replace(record.provenance, source_path=Path("/home/alice/private/decks/source.csv")),
    )

    result = _generate((record,), profile)

    assert result.notes
    assert all(note.provenance.source_path is None for note in result.notes)
    assert all(("Source Path", "") in note.to_anki_fields() for note in result.notes)
    assert all("alice" not in field_value for note in result.notes for _, field_value in note.to_anki_fields())


def test_generated_notes_are_deterministic_for_identical_input() -> None:
    profile = _profile()
    records = (
        _record("dīcere, dīcō, dīxī, dictum", source_identity="entry-a", profile=profile),
        _record("ferre, ferō, tulī, lātum", lexical_entry="ferō", source_identity="entry-b", profile=profile),
    )

    first = _generate(records, profile)
    second = _generate(records, profile)

    assert first == second


def test_generated_note_knowledge_uses_role_labels_for_every_confirmed_role() -> None:
    profile = _profile(
        roles=("present_1s", "present_infinitive", "perfect_1s", "supine"),
        separators=(" — ",),
    )

    result = _generate(
        (_record("ferō — ferre — tulī — lātum", lexical_entry="ferō", profile=profile),),
        profile,
    )

    principal_parts = result.notes[0].content.principal_parts
    evidence, visible = principal_parts.split("</span>", 1)
    assert 'class="source-extraction"' in evidence
    assert visible.startswith(
        "<strong>Präsens, 1. Person Singular:</strong> ferō<br>"
        "<strong>Infinitiv:</strong> ferre<br>"
        "<strong>Perfekt, 1. Person Singular:</strong> tulī<br>"
        "<strong>Supinum:</strong> lātum <em>(linguistic review required; target withheld)</em>"
    )
    assert (result.generated_count, result.skipped_count, result.generated_warning_count) == (1, 0, 1)
    assert [skip.code for skip in result.skips] == ["linguistic_review_required"]
    assert result.skips[0].evidence is not None
    assert result.skips[0].evidence.candidates is not None
    assert result.skips[0].evidence.candidates[-1] == ("lātum",)
    assert replace(result, source_entry_count=None).skipped_count == 0
    assert all("Linguistic review required" in card.answer for card in result.notes[0].cards if card.eligible)
    completion_cards = [
        card for card in result.notes[0].cards if card.eligible and card.slot.recipe == "principal_part_completion"
    ]
    assert len(completion_cards) == 3
    for card in completion_cards:
        assert "— (linguistic review required)" in card.prompt
        assert "unresolved source evidence" not in card.prompt


def _markup_only_profile() -> DeckProfile:
    return _profile(
        roles=("present_1s", "present_infinitive", "perfect_1s", "supine"),
        separators=(" — ",),
    )


def _markup_only_record(
    principal_parts: str,
    *,
    lexical_entry: str = "amō",
    source_identity: str,
) -> CanonicalSourceRecord:
    return _record(
        principal_parts,
        lexical_entry=lexical_entry,
        source_identity=source_identity,
        profile=_markup_only_profile(),
    )


def test_markup_only_part_keeps_the_note_but_omits_blank_answers_and_cards() -> None:
    profile = _markup_only_profile()

    result = _generate(
        (_markup_only_record("amo — amare — <b></b> — amatum", source_identity="entry-amo"),),
        profile,
    )

    assert len(result.notes) == 1
    note = result.notes[0]
    omission_skips = [skip for skip in result.skips if skip.code == "omitted_principal_part"]
    assert len(omission_skips) == 1
    assert omission_skips[0].status == "incomplete"
    assert omission_skips[0].source_identity == "entry-amo"
    assert "perfect_1s" in omission_skips[0].message
    assert note.card_keys == (
        "principal_part_completion:present_1s",
        "principal_part_completion:present_infinitive",
        "principal_part_recognition:present_1s",
        "principal_part_recognition:present_infinitive",
    )
    assert "<strong>Perfekt, 1. Person Singular:</strong> —" in note.content.principal_parts
    for expected_answer in ("amo", "amare", "amatum"):
        assert expected_answer in note.content.principal_parts
    assert "<b>" not in note.content.principal_parts


def test_historical_markup_only_batch_never_emits_blank_perfect_answers() -> None:
    profile = _markup_only_profile()
    records = (
        _markup_only_record("amo — amare — <b></b> — amatum", source_identity="entry-1"),
        _markup_only_record("amō — amāre — <b class='x'></b> — amātum", source_identity="entry-2"),
        _markup_only_record("amō — amāre — <!-- unbekannt --> — amātum", source_identity="entry-3"),
        _markup_only_record("amō — amāre — &nbsp; — amātum", source_identity="entry-4"),
        _markup_only_record("moneō — monēre — <i></i> — monitum", lexical_entry="moneō", source_identity="entry-5"),
        _markup_only_record("regō — regere — <span> </span> — rectum", lexical_entry="regō", source_identity="entry-6"),
        _markup_only_record(
            "audiō — audīre — <b></b><!-- x --> — audītum", lexical_entry="audiō", source_identity="entry-7"
        ),
        _markup_only_record("dīcō — dīcere — dīxī — dictum", lexical_entry="dīcō", source_identity="entry-8"),
    )

    result = _generate(records, profile)

    assert len(result.notes) == 8
    omission_skips = [skip for skip in result.skips if skip.code == "omitted_principal_part"]
    assert len(omission_skips) == 7
    assert all(skip.status == "incomplete" for skip in omission_skips)
    assert {skip.source_identity for skip in omission_skips} == {f"entry-{index}" for index in range(1, 8)}
    for note in result.notes:
        perfect_lines = [line for line in note.content.principal_parts.split("<br>") if "Perfekt" in line]
        assert len(perfect_lines) == 1
        perfect_answer = perfect_lines[0].split("</strong>")[-1]
        identity = note.provenance.source_identity or ""
        if identity == "entry-8":
            assert perfect_answer == " dīxī"
            assert "principal_part_completion:perfect_1s" in note.card_keys
        else:
            assert perfect_answer == " —"
            assert "principal_part_completion:perfect_1s" not in note.card_keys
            assert "principal_part_recognition:perfect_1s" not in note.card_keys
        rendered_answers = [line.split("</strong>")[-1].strip() for line in note.content.principal_parts.split("<br>")]
        eligible_answers = [answer for answer in rendered_answers if answer != "—"]
        assert eligible_answers and all(answer for answer in eligible_answers)


def test_markup_only_lexical_entry_is_a_structured_skip_not_a_crash() -> None:
    profile = _markup_only_profile()

    result = _generate(
        (
            _markup_only_record(
                "legō — legere — lēgī — lēctum", lexical_entry="<i></i>", source_identity="entry-blank-lemma"
            ),
            _markup_only_record("dīcō — dīcere — dīxī — dictum", lexical_entry="dīcō", source_identity="entry-ok"),
        ),
        profile,
    )

    assert [note.provenance.source_identity for note in result.notes] == ["entry-ok"]
    blank_lemma_skips = [skip for skip in result.skips if skip.code == "missing_lexical_entry"]
    assert len(blank_lemma_skips) == 1
    assert blank_lemma_skips[0].status == "incomplete"
    assert blank_lemma_skips[0].source_identity == "entry-blank-lemma"


def test_rendering_does_not_double_decode_normalized_display_text() -> None:
    profile = _markup_only_profile()

    result = _generate(
        (_markup_only_record("discō — discere — did&amp;#x12B;cī — doctum", source_identity="entry-double-entity"),),
        profile,
    )

    note = result.notes[0]
    perfect_line = next(line for line in note.content.principal_parts.split("<br>") if "Perfekt" in line)
    assert perfect_line.endswith("did&amp;#x12B;cī")
    assert unescape(perfect_line).endswith("did&#x12B;cī")
    assert "didīcī" not in unescape(perfect_line)
    assert "didīcī" not in note.content.principal_parts


@pytest.mark.parametrize("recipe", ["principal_part_completion", "principal_part_recognition"])
def test_unresolved_source_evidence_never_becomes_a_recipe_target(recipe: str) -> None:
    profile = _profile(recipes=(recipe,)).apply_overrides(
        {"principal_parts": {"pipe_alternatives": True, "trailing_poet_hint": True}}
    )
    result = _generate(
        (
            _record("amāre|amare, amō|amo, amāvī, amātum", source_identity="alternatives", profile=profile),
            _record("monēre, moneō, mōnī<br>poet., monitum", source_identity="hint", profile=profile),
            _record("dīcere, dīcō, dīxī, dictum<br>supine (not PPP)", source_identity="conflict", profile=profile),
            _record("amāre, amō, , amātum", source_identity="omission", profile=profile),
        ),
        profile,
    )
    assert result.generated_count == 3
    assert result.skipped_count == 1
    assert result.generated_warning_count == 3
    assert [len(note.card_keys) for note in result.notes] == (
        [0, 2, 2] if recipe == "principal_part_completion" else [1, 2, 2]
    )
    for note in result.notes:
        assert 'class="source-extraction"' in note.content.principal_parts
    assert "amāre | amare" in result.notes[0].content.principal_parts
    assert "unresolved" in result.notes[0].content.principal_parts
    assert "poet." in result.notes[1].content.principal_parts
    if recipe == "principal_part_completion":
        for card in result.notes[0].cards:
            if card.eligible:
                assert "amāre" not in card.prompt
                assert "unresolved source evidence" in card.prompt
    conflict = next(skip for skip in result.skips if skip.code == "unconfirmed_source_evidence")
    assert conflict.evidence is not None
    assert conflict.evidence.raw == "dīcere, dīcō, dīxī, dictum<br>supine (not PPP)"
    assert conflict.evidence.hints[0].position == 4
