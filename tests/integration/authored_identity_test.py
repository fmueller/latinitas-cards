"""Authored keys, not mutable note content, own identity."""

from pathlib import Path

import pytest
from authored_import_test import row, write_rows

from latinitas_cards.authored_identity import derive_authored_id, normalize_authored_key, reconcile_authored_import
from latinitas_cards.authored_import import AuthoredImportError, load_authored_import


def test_normalization_and_identity_contract(tmp_path: Path) -> None:
    assert normalize_authored_key("  le\u0301sson\t 1  ") == "lésson 1"
    rows = [
        row(key="lésson 1"),
        row(
            key=" le\u0301sson\t1 ",
            meaning="changed",
            status="skip",
            tags=["new"],
            provenance={"document": "other", "section": "new", "reference": "citation"},
        ),
    ]
    items = load_authored_import(write_rows(tmp_path / "notes.jsonl", rows)).require_valid()
    first = derive_authored_id("course", items[0])
    assert first == derive_authored_id("course", items[1])
    assert first != derive_authored_id("other", items[0])
    qa = load_authored_import(write_rows(tmp_path / "qa.jsonl", [row("qa", key="lésson 1")])).require_valid()[0]
    assert first != derive_authored_id("course", qa)
    assert first != derive_authored_id("course", items[0].model_copy(update={"key": "Lésson 1"}))
    with pytest.raises(ValueError):
        derive_authored_id(" ", items[0])


@pytest.mark.parametrize(
    "kind,optional,value",
    [("vocab", "dictionary_form", "amāre"), ("form", "context", "Multī amāvērunt."), ("qa", None, None)],
)
def test_duplicates_merge_independently_of_order(
    tmp_path: Path, kind: str, optional: str | None, value: str | None
) -> None:
    rows = [
        row(kind, key=" shared ", tags=["z", "a"], **({optional: ""} if optional else {})),
        row(
            kind,
            key="shared",
            status="skip",
            tags=["b", "a"],
            provenance={"document": "Vorlesung ä", "section": "  Freier Abschnitt  ", "reference": " Lk 1,28 "},
            **({optional: value} if optional else {}),
        ),
        row(kind, key="shared"),
    ]
    results = []
    for ordered in (rows, list(reversed(rows))):
        result = reconcile_authored_import(
            "course", load_authored_import(write_rows(tmp_path / "notes.jsonl", ordered))
        )
        (note,) = result.require_valid()
        assert result.merged_duplicates == ((1, 2, 3),)
        assert note.item.key == "shared" and note.item.tags == ("a", "b", "z") and note.item.status == "skip"
        assert note.item.provenance.reference == " Lk 1,28 "
        if optional:
            assert getattr(note.item, optional) == value
        results.append(note)
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "kind,changes,field",
    [
        ("vocab", {"lemma": "other"}, "lemma"),
        ("vocab", {"meaning": "other"}, "meaning"),
        ("form", {"text_form": "other"}, "text_form"),
        ("form", {"base_form": "other"}, "base_form"),
        ("form", {"analysis": "other"}, "analysis"),
        ("form", {"translation": "other"}, "translation"),
        ("qa", {"answer": "other"}, "answer"),
        ("qa", {"question": "other"}, "question"),
        ("qa", {"language_tag": "la"}, "language_tag"),
        ("qa", {"provenance": {"document": "other", "section": "new"}}, "provenance.document"),
        ("vocab", {"dictionary_form": "other"}, "dictionary_form"),
        ("form", {"context": "other"}, "context"),
        (
            "qa",
            {"provenance": {"document": "Vorlesung ä", "section": "  Freier Abschnitt  ", "reference": "other"}},
            "provenance.reference",
        ),
    ],
)
def test_conflicts_report_actual_lines_and_fields(
    tmp_path: Path, kind: str, changes: dict[str, object], field: str
) -> None:
    original = row(
        kind,
        key="shared",
        **({"dictionary_form": "original"} if kind == "vocab" else {"context": "original"} if kind == "form" else {}),
    )
    original["provenance"] = {"document": "Vorlesung ä", "section": "  Freier Abschnitt  ", "reference": "original"}
    rows = [row(kind, key="unrelated"), original, row(kind, key=" shared ", **changes)]
    for ordered in (rows, [rows[0], rows[2], rows[1]]):
        result = reconcile_authored_import(
            "course", load_authored_import(write_rows(tmp_path / "notes.jsonl", ordered))
        )
        with pytest.raises(AuthoredImportError, match=f"line 3.*{field}.*line 2"):
            result.require_valid()
        if field == "provenance.document":
            assert "provenance.section" in str(result.errors)


def test_optional_conflict_names_contributing_line_and_retains_loader_errors(tmp_path: Path) -> None:
    rows: list[dict[str, object] | str] = [row(), row(dictionary_form="one"), row(dictionary_form="two"), "not JSON"]
    result = reconcile_authored_import("course", load_authored_import(write_rows(tmp_path / "notes.jsonl", rows)))
    assert any(error.line_number == 4 for error in result.errors)
    assert any(error.line_number == 3 and "line 2" in error.assumption for error in result.errors)
    with pytest.raises(AuthoredImportError):
        result.require_valid()
