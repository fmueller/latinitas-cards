"""The authored import boundary validates every row before selection."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from latinitas_cards.authored_import import AuthoredImportError, FormItem, QaItem, VocabItem, load_authored_import


def row(item_kind: str = "vocab", /, **changes: object) -> dict[str, object]:
    content = {
        "vocab": {"lemma": "amō", "meaning": "lieben"},
        "form": {
            "text_form": "amāvērunt",
            "base_form": "amō",
            "analysis": "Perfekt, 3. Pl.",
            "translation": "sie liebten",
        },
        "qa": {"question": "Was ist ein Ablativ?", "answer": "Ein Kasus."},
    }[item_kind]
    return {
        "schema_version": 1,
        "kind": item_kind,
        "key": f"lesson:{item_kind}:1",
        "provenance": {"document": "Vorlesung ä", "section": "  Freier Abschnitt  "},
        "status": "include",
        "language_tag": "de-DE",
        **content,
        **changes,
    }


def write_rows(path: Path, rows: Sequence[dict[str, object] | str]) -> Path:
    path.write_text(
        "\n".join(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False) for value in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_typed_kinds_and_opaque_provenance(tmp_path: Path) -> None:
    refs = (" Lk 1,28 ", "Cic. Cat. 1,1", "Lektion 3")
    rows = [
        row(kind, provenance={"document": "Ä", "section": "  ?!  ", "reference": ref})
        for kind, ref in zip(("vocab", "form", "qa"), refs, strict=True)
    ]
    rows[0].update(dictionary_form="amō, amāre, amāvī, amātum", tags=["lesson::3", "ä"])
    rows[1].update(context="Multī amāvērunt.", status="skip")
    result = load_authored_import(write_rows(tmp_path / "notes.jsonl", rows))
    vocab, form, qa = result.require_valid()
    assert isinstance(vocab, VocabItem) and vocab.lemma == "amō" and vocab.meaning == "lieben"
    assert vocab.dictionary_form == "amō, amāre, amāvī, amātum" and vocab.tags == ("lesson::3", "ä")
    assert isinstance(form, FormItem) and form.context == "Multī amāvērunt." and form.status == "skip"
    assert form.text_form == "amāvērunt" and form.base_form == "amō"
    assert form.analysis == "Perfekt, 3. Pl." and form.translation == "sie liebten"
    assert isinstance(qa, QaItem) and qa.answer == "Ein Kasus." and qa.tags == ()
    assert [item.provenance.reference for item in result.diagnostic_items] == list(refs)
    assert all(item.provenance.section == "  ?!  " and item.language_tag == "de-DE" for item in result.require_valid())
    assert [item.line_number for item in result.require_valid()] == [1, 2, 3]


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": True}, "schema_version"),
        ({"schema_version": "1"}, "schema_version"),
        ({"kind": "cloze"}, "kind"),
        ({"key": "  "}, "key"),
        ({"lemma": None}, "lemma"),
        ({"status": "other"}, "status"),
        ({"language_tag": "German!"}, "language_tag"),
        ({"tags": "lesson"}, "tags"),
        ({"tags": [3]}, "tags"),
        ({"tags": ["two words"]}, "tags"),
        ({"provenance": {"document": "x"}}, "section"),
        ({"extra": "typo"}, "extra"),
    ],
)
def test_invalid_assumptions_have_line_numbers(tmp_path: Path, changes: dict[str, object], field: str) -> None:
    result = load_authored_import(write_rows(tmp_path / "notes.jsonl", [row(), row(**changes)]))
    assert result.errors and all(error.line_number == 2 for error in result.errors)
    assert field in str(result.errors[0])
    with pytest.raises(AuthoredImportError, match=f"line 2.*{field}"):
        result.require_valid()


@pytest.mark.parametrize(
    "kind,field",
    [
        ("vocab", "lemma"),
        ("vocab", "meaning"),
        ("form", "text_form"),
        ("form", "base_form"),
        ("form", "analysis"),
        ("form", "translation"),
        ("qa", "question"),
        ("qa", "answer"),
        ("vocab", "schema_version"),
        ("vocab", "kind"),
        ("vocab", "key"),
        ("vocab", "provenance"),
        ("vocab", "status"),
        ("vocab", "language_tag"),
    ],
)
def test_required_fields(tmp_path: Path, kind: str, field: str) -> None:
    value = row(kind)
    del value[field]
    result = load_authored_import(write_rows(tmp_path / "notes.jsonl", [value]))
    assert result.errors and field in str(result.errors[0])


def test_errors_aggregate_across_skipped_rows_and_recover_after_bad_json(tmp_path: Path) -> None:
    result = load_authored_import(
        write_rows(
            tmp_path / "notes.jsonl",
            [row(status="skip", meaning=""), "{bad", row("qa"), row(language_tag="!"), "[]", ""],
        )
    )
    assert [error.line_number for error in result.errors] == [1, 2, 4, 5, 6]
    assert len(result.diagnostic_items) == 1 and result.diagnostic_items[0].line_number == 3
    with pytest.raises(AuthoredImportError) as caught:
        result.require_valid()
    assert caught.value.errors == result.errors
    assert "meaning" in str(caught.value) and "JSON" in str(caught.value)


def test_invalid_utf8_is_reported_and_next_row_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "notes.jsonl"
    path.write_bytes(b"\xff\n" + json.dumps(row()).encode("utf-8") + b"\n")
    result = load_authored_import(path)
    assert result.errors[0].line_number == 1 and "UTF-8" in str(result.errors[0])
    assert result.diagnostic_items[0].line_number == 2
    with pytest.raises(AuthoredImportError):
        result.require_valid()


@pytest.mark.parametrize(
    "duplicate,field",
    [
        ('"status": "include", "status": "skip"', "status"),
        ('"status": "skip", "status": "include"', "status"),
        ('"document": "first", "document": "second"', "document"),
    ],
)
def test_duplicate_json_keys_are_invalid_and_later_rows_recover(tmp_path: Path, duplicate: str, field: str) -> None:
    text = json.dumps(row())
    if field == "status":
        text = text.replace('"status": "include"', duplicate)
    else:
        text = text.replace('"document": "Vorlesung \\u00e4"', duplicate)
    result = load_authored_import(write_rows(tmp_path / "notes.jsonl", [text, row("qa")]))
    assert result.errors and result.errors[0].line_number == 1
    assert "duplicate" in str(result.errors[0]) and field in str(result.errors[0])
    assert [item.line_number for item in result.diagnostic_items] == [2]
    with pytest.raises(AuthoredImportError):
        result.require_valid()
