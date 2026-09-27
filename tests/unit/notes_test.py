from latinitas_cards.identity import derive_latinitas_id
from latinitas_cards.notes import (
    AUTHORITATIVE_NOTE_FIELDS,
    CSV_EXPORT_FIELD_NAMES,
    GENERATED_NOTE_FIELD_NAMES,
    TAGS_CSV_COLUMN,
    FieldOwnership,
    GeneratedNote,
    GeneratedNoteProvenance,
    GenerationMetadata,
    ManagedNoteContent,
)


def _note(**overrides: object) -> GeneratedNote:
    values: dict[str, object] = {
        "source_identity": "source-17",
        "source_scope": None,
        "object_key": "lexeme-1",
        "provenance": GeneratedNoteProvenance(source_kind="csv", location="row 2"),
        "metadata": GenerationMetadata(profile_digest="profile-sha256:abc123"),
        "content": ManagedNoteContent(
            lemma="dīcō",
            principal_parts="<strong>Infinitiv:</strong> dīcere",
            meaning="sagen",
            tags=("latinitas",),
        ),
        "personal_notes": "Review this next week",
    }
    values.update(overrides)
    return GeneratedNote.create(**values)  # type: ignore[arg-type]


def test_authoritative_schema_declares_ownership_for_every_field() -> None:
    ownership_by_name: dict[str, FieldOwnership] = {field.name: field.ownership for field in AUTHORITATIVE_NOTE_FIELDS}

    assert ownership_by_name["LatinitasID"] == "managed"
    assert ownership_by_name["Lemma"] == "managed"
    assert ownership_by_name["Principal Parts"] == "managed"
    assert ownership_by_name["Meaning"] == "managed"
    assert ownership_by_name["Source ID"] == "managed"
    assert ownership_by_name["Source Scope"] == "managed"
    assert ownership_by_name["Source Kind"] == "managed"
    assert ownership_by_name["Source Location"] == "managed"
    assert ownership_by_name["Source Path"] == "managed"
    assert ownership_by_name["Note Schema"] == "managed"
    assert ownership_by_name["Generator"] == "managed"
    assert ownership_by_name["Profile"] == "managed"
    assert ownership_by_name["Tags"] == "transport"
    assert ownership_by_name["Personal Notes"] == "personal"


def test_note_type_fields_and_csv_export_are_derived_from_the_authoritative_schema() -> None:
    assert tuple(field.name for field in AUTHORITATIVE_NOTE_FIELDS) == GENERATED_NOTE_FIELD_NAMES
    assert GENERATED_NOTE_FIELD_NAMES[0] == "LatinitasID"
    assert GENERATED_NOTE_FIELD_NAMES[-1] == "Personal Notes"

    exported = tuple(field.name for field in AUTHORITATIVE_NOTE_FIELDS if field.exported)
    assert exported == CSV_EXPORT_FIELD_NAMES
    assert "Personal Notes" not in CSV_EXPORT_FIELD_NAMES
    assert CSV_EXPORT_FIELD_NAMES[0] == "LatinitasID"


def test_tags_is_transport_metadata_with_a_derived_csv_directive_position() -> None:
    tags_field = next(field for field in AUTHORITATIVE_NOTE_FIELDS if field.name == "Tags")

    assert tags_field.ownership == "transport"
    assert tags_field.exported
    assert CSV_EXPORT_FIELD_NAMES.index("Tags") + 1 == TAGS_CSV_COLUMN
    assert TAGS_CSV_COLUMN == 5


def test_generated_note_separates_identity_knowledge_provenance_metadata_and_personal_notes() -> None:
    note = _note()

    assert note.latinitas_id == derive_latinitas_id("source-17", "lexeme-1")
    fields = dict(note.to_anki_fields())
    assert fields["LatinitasID"] == note.latinitas_id
    assert fields["Lemma"] == "dīcō"
    assert fields["Personal Notes"] == "Review this next week"
    assert fields["Source ID"] == "source-17"
    assert fields["Source Scope"] == ""
    assert fields["Tags"] == "latinitas"
    assert fields["Note Schema"] == note.metadata.note_schema
    assert fields["Generator"] == note.metadata.generator
    assert fields["Profile"] == "profile-sha256:abc123"


def test_scoped_source_identity_participates_in_note_identity_and_provenance() -> None:
    scoped = _note(
        source_identity="csv-source-000001",
        source_scope="scope-alpha",
        provenance=GeneratedNoteProvenance(source_kind="csv", location="row 2", source_scope="scope-alpha"),
    )

    assert scoped.latinitas_id == derive_latinitas_id("csv-source-000001", "lexeme-1", source_scope="scope-alpha")
    assert dict(scoped.to_anki_fields())["Source Scope"] == "scope-alpha"


def test_mutable_knowledge_wording_tags_and_profile_do_not_change_identity() -> None:
    note = _note()

    changed = note.with_managed_content(
        ManagedNoteContent(
            lemma="dīcō (emended)",
            principal_parts="<strong>Infinitiv:</strong> dīcere<br><strong>Perfekt:</strong> dīxī",
            meaning="sagen; aussprechen",
            tags=("updated",),
        )
    ).with_metadata(
        GenerationMetadata(profile_digest="profile-sha256:def456"),
    )

    assert changed.latinitas_id == note.latinitas_id
    assert changed.personal_notes == note.personal_notes
    assert changed.content != note.content
    assert changed.metadata.profile_digest == "profile-sha256:def456"
    assert dict(changed.to_anki_fields())["Profile"] == "profile-sha256:def456"


def test_generated_note_rejects_a_provenance_scope_dropped_by_the_caller() -> None:
    scoped_provenance = GeneratedNoteProvenance(
        source_kind="csv",
        location="row 2",
        source_scope="scope-alpha",
    )

    try:
        GeneratedNote.create(
            source_identity="csv-source-000001",
            provenance=scoped_provenance,
            object_key="lexeme-1",
            metadata=GenerationMetadata(profile_digest="profile-sha256:abc123"),
            content=ManagedNoteContent(lemma="dīcō", principal_parts="dīcere", meaning="", tags=()),
        )
    except ValueError as error:
        assert "source scope" in str(error)
    else:
        raise AssertionError("dropping a carried source scope must be rejected")


def test_generated_note_rejects_an_identity_that_does_not_match_its_object() -> None:
    note = _note()

    try:
        GeneratedNote(
            latinitas_id="latinitas-v2-wrong",
            object_key=note.object_key,
            content=note.content,
            provenance=note.provenance,
            metadata=note.metadata,
            personal_notes=note.personal_notes,
        )
    except ValueError as error:
        assert "LatinitasID" in str(error)
    else:
        raise AssertionError("a mismatched logical identity must be rejected")


def test_generated_note_rejects_empty_lemma_or_principal_parts_knowledge() -> None:
    for knowledge in (
        {"lemma": " ", "principal_parts": "dīcere"},
        {"lemma": "dīcō", "principal_parts": ""},
    ):
        try:
            ManagedNoteContent(tags=(), meaning="", **knowledge)
        except ValueError as error:
            assert "lemma" in str(error) or "principal parts" in str(error)
        else:
            raise AssertionError("empty managed knowledge must be rejected")
