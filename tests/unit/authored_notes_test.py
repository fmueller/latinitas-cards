"""Dedicated authored note contracts, directions, and stable occurrence identity."""

from pathlib import Path

import pytest
from authored_import_test import row, write_rows

from latinitas_cards.authored_identity import reconcile_authored_import
from latinitas_cards.authored_import import load_authored_import
from latinitas_cards.authored_notes import (
    AUTHORED_NOTE_TYPES,
    RenderedAuthoredNote,
    reference_authored_note_types,
    render_authored_note,
)
from latinitas_cards.notes import GenerationMetadata


@pytest.mark.parametrize(
    "kind,name,content,front,back",
    [
        ("vocab", "Latinitas Authored Vocabulary", ("Lemma", "Dictionary Form", "Meaning"), "Lemma", "Meaning"),
        (
            "form",
            "Latinitas Authored Form",
            ("Text Form", "Base Form", "Analysis", "Translation", "Context"),
            "Text Form",
            "Base Form",
        ),
        ("qa", "Latinitas Authored QA", ("Question", "Answer"), "Question", "Answer"),
    ],
)
def test_contract_and_empty_optionals(
    tmp_path: Path, kind: str, name: str, content: tuple[str, ...], front: str, back: str
) -> None:
    (identified,) = reconcile_authored_import(
        "course", load_authored_import(write_rows(tmp_path / "notes.jsonl", [row(kind, tags=["lesson::1"])]))
    ).require_valid()
    note = render_authored_note(identified, GenerationMetadata(profile_digest="profile-test"))
    schema = AUTHORED_NOTE_TYPES[kind]
    assert note.note_type == schema and schema.name == name
    assert schema.field_names == (
        "LatinitasID",
        *content,
        "Document",
        "Section",
        "Reference",
        "Source Key",
        "Source Kind",
        "Language",
        "Note Schema",
        "Generator",
        "Profile",
        "Personal Notes",
    )
    assert schema.fields[-1].ownership == "personal" and not schema.fields[-1].exported
    values = dict(note.to_export_fields())
    assert tuple(values) == schema.field_names[:-1]
    assert values["LatinitasID"] == identified.latinitas_id
    assert values["Document"] == "Vorlesung ä" and values["Section"] == "  Freier Abschnitt  "
    assert values["Reference"] == "" and values["Source Key"] == f"lesson:{kind}:1"
    assert values["Source Kind"] == kind and values["Language"] == "de-DE"
    assert values["Note Schema"] == "authored-1" and values["Profile"] == "profile-test"
    assert values["Generator"] == "3" and note.tags == ("lesson::1",)
    assert "Personal Notes" not in values and "Tags" not in schema.field_names
    assert "{{" + front + "}}" in schema.front_template
    assert "{{" + back + "}}" in schema.back_template
    assert "{{Personal Notes}}" in schema.back_template
    assert "{{LatinitasID}}" not in schema.front_template
    if kind == "vocab":
        assert values["Lemma"] == "amō" and values["Meaning"] == "lieben" and values["Dictionary Form"] == ""
        assert "Meaning" not in schema.front_template
    elif kind == "form":
        assert [values[f] for f in content] == ["amāvērunt", "amō", "Perfekt, 3. Pl.", "sie liebten", ""]
        assert "{{#Context}}" in schema.front_template
        assert "{{Analysis}}" in schema.back_template and "{{Translation}}" in schema.back_template
        assert "Base Form" not in schema.front_template
    else:
        assert values["Question"] == "Was ist ein Ablativ?" and values["Answer"] == "Ein Kasus."
        assert "Answer" not in schema.front_template


def test_optional_content_and_context_keys(tmp_path: Path) -> None:
    rows = [
        row(
            "form",
            key="occurrence:1",
            context="Multī amāvērunt.",
            provenance={"document": "A", "section": "B", "reference": "1"},
        ),
        row(
            "form",
            key="occurrence:2",
            context="Paucī amāvērunt.",
            provenance={"document": "A", "section": "B", "reference": "2"},
        ),
        row("vocab", dictionary_form="amō, amāre"),
        row("qa", provenance={"document": "A", "section": "B", "reference": "3"}),
    ]

    def render() -> dict[str, RenderedAuthoredNote]:
        identified = reconcile_authored_import(
            "course", load_authored_import(write_rows(tmp_path / "notes.jsonl", rows))
        ).require_valid()
        return {n.item.key: render_authored_note(n, GenerationMetadata(profile_digest="p")) for n in identified}

    notes = render()
    first = notes["occurrence:1"]
    second = notes["occurrence:2"]
    assert first.latinitas_id != second.latinitas_id
    assert dict(first.to_export_fields())["Context"] == "Multī amāvērunt."
    assert dict(second.to_export_fields())["Context"] == "Paucī amāvērunt."
    assert dict(notes["lesson:vocab:1"].to_export_fields())["Dictionary Form"] == "amō, amāre"
    assert dict(notes["lesson:qa:1"].to_export_fields())["Reference"] == "3"
    rows[0]["provenance"] = {"document": "A", "section": "B", "reference": "citation corrected"}
    edited = render()["occurrence:1"]
    assert edited.latinitas_id == first.latinitas_id
    assert dict(edited.to_export_fields())["Reference"] == "citation corrected"


def test_managed_text_is_html_safe_and_preserves_lines(tmp_path: Path) -> None:
    (identified,) = reconcile_authored_import(
        "course",
        load_authored_import(
            write_rows(tmp_path / "notes.jsonl", [row("qa", question='<script>alert("x")</script>\nA & B')])
        ),
    ).require_valid()
    fields = dict(render_authored_note(identified, GenerationMetadata(profile_digest="p")).to_export_fields())
    assert fields["Question"] == "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;<br>A &amp; B"


def test_published_reference_matches_contract() -> None:
    published = Path("docs/authored-note-types.md").read_text(encoding="utf-8")
    assert reference_authored_note_types() in published


@pytest.mark.parametrize("kind", ["vocab", "form", "qa"])
@pytest.mark.parametrize(
    "value,expected",
    [
        ("left\rright", "left<br>right"),
        ("left\r\nright", "left<br>right"),
        ("left\nright", "left<br>right"),
        ('<ä>\r\n\r&\n"𐌀"\r\nend', "&lt;ä&gt;<br><br>&amp;<br>&quot;𐌀&quot;<br>end"),
    ],
)
def test_logical_newlines_in_content_and_provenance(tmp_path: Path, kind: str, value: str, expected: str) -> None:
    schema = AUTHORED_NOTE_TYPES[kind]
    content = {attribute: value for _, attribute in schema.content_fields}
    source = write_rows(
        tmp_path / "notes.jsonl",
        [row(kind, **content, provenance={"document": value, "section": value, "reference": value})],
    )
    before = source.read_bytes()
    (identified,) = reconcile_authored_import("course", load_authored_import(source)).require_valid()
    fields = dict(render_authored_note(identified, GenerationMetadata(profile_digest="p")).to_export_fields())
    for field, attribute in schema.content_fields:
        assert fields[field] == expected
        assert getattr(identified.item, attribute) == value
    for field in ("Document", "Section", "Reference"):
        assert fields[field] == expected
        assert getattr(identified.item.provenance, field.lower()) == value
    assert source.read_bytes() == before
