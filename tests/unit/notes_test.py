from latinitas_cards.identity import derive_latinitas_id
from latinitas_cards.notes import (
    CSV_EXPORT_FIELD_NAMES,
    GENERATED_NOTE_FIELD_NAMES,
    GeneratedNote,
    GeneratedNoteProvenance,
    ManagedNoteContent,
    RecipeMetadata,
)


def _note() -> GeneratedNote:
    return GeneratedNote.create(
        source_identity="source-17",
        provenance=GeneratedNoteProvenance(source_kind="csv", location="row 2"),
        recipe=RecipeMetadata(
            recipe_identity="principal_part_completion",
            exercise_key="perfect_1s",
            recipe_version="1",
        ),
        content=ManagedNoteContent(prompt="dico — _____", answer="dixi", tags=("latinitas",)),
        personal_notes="Review this next week",
    )


def test_generated_note_separates_identity_managed_content_provenance_recipe_and_personal_notes() -> None:
    note = _note()

    assert note.latinitas_id == derive_latinitas_id("source-17", "principal_part_completion", "perfect_1s")
    fields = note.to_anki_fields()
    assert fields[0] == ("LatinitasID", note.latinitas_id)
    assert fields[-1] == ("Personal Notes", "Review this next week")
    assert ("Source ID", "source-17") in fields
    assert ("Recipe", "principal_part_completion") in fields
    assert ("Exercise Key", "perfect_1s") in fields


def test_note_type_retains_personal_notes_while_csv_export_omits_the_user_owned_field() -> None:
    assert GENERATED_NOTE_FIELD_NAMES[-1] == "Personal Notes"
    assert "Personal Notes" not in CSV_EXPORT_FIELD_NAMES
    assert CSV_EXPORT_FIELD_NAMES == (
        "LatinitasID",
        "Prompt",
        "Answer",
        "Tags",
        "Source ID",
        "Source Kind",
        "Source Location",
        "Source Path",
        "Recipe",
        "Exercise Key",
        "Recipe Version",
    )
    assert CSV_EXPORT_FIELD_NAMES[0] == "LatinitasID"


def test_mutable_content_tags_gloss_html_and_recipe_version_do_not_change_identity() -> None:
    note = _note()

    changed = note.with_managed_content(
        ManagedNoteContent(prompt="<b>dico</b> — _____", answer="dixi (sagen)", tags=("updated",))
    ).with_recipe_metadata(
        RecipeMetadata(
            recipe_identity="principal_part_completion",
            exercise_key="perfect_1s",
            recipe_version="2",
        )
    )

    assert changed.latinitas_id == note.latinitas_id
    assert changed.personal_notes == note.personal_notes
    assert changed.content != note.content
    assert changed.recipe.recipe_version == "2"


def test_generated_note_rejects_an_identity_that_does_not_match_its_logical_exercise() -> None:
    note = _note()

    try:
        GeneratedNote(
            latinitas_id="latinitas-wrong",
            content=note.content,
            provenance=note.provenance,
            recipe=note.recipe,
            personal_notes=note.personal_notes,
        )
    except ValueError as error:
        assert "LatinitasID" in str(error)
    else:
        raise AssertionError("a mismatched logical identity must be rejected")
