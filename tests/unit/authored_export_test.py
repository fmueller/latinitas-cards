import csv
import io
from pathlib import Path

import pytest
from authored_import_test import row, write_rows
from typer.testing import CliRunner, Result

from latinitas_cards.authored_notes import AUTHORED_NOTE_TYPES
from latinitas_cards.cli import app


def export(source: Path, destination: Path, *filters: str) -> Result:
    return CliRunner().invoke(
        app,
        [
            "authored",
            "export",
            str(source),
            "--namespace",
            "course",
            "--output-dir",
            str(destination),
            "--deck",
            "Latin::Authored",
            *filters,
        ],
    )


def data(payload: bytes) -> list[list[str]]:
    return list(
        csv.reader(
            io.StringIO("\n".join(line for line in payload.decode("utf-8").splitlines() if not line.startswith("#")))
        )
    )


def test_all_kinds_deterministic_edited_identity_and_personal_omission(tmp_path: Path) -> None:
    source = write_rows(
        tmp_path / "notes.jsonl",
        [row(kind, tags=["lesson::1", "ä"]) for kind in ("vocab", "form", "qa")] + [row(key="skipped", status="skip")],
    )
    out = tmp_path / "out"
    out.mkdir()
    result = export(source, out)
    assert result.exit_code == 0, result.output
    assert "Effective selection: 3" in result.output
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    assert set(before) == {"vocab.csv", "form.csv", "qa.csv"}
    for kind, schema in AUTHORED_NOTE_TYPES.items():
        payload = before[f"{kind}.csv"]
        fields = tuple(field.name for field in schema.fields if field.exported) + ("Tags",)
        assert f"#columns:{','.join(fields)}\n".encode() in payload
        assert f"#tags column:{len(fields)}\n".encode() in payload
        assert f"#notetype:{schema.name}\n#deck:Latin::Authored\n".encode() in payload
        assert b"Personal Notes" not in payload
        rows = data(payload)
        assert len(rows) == 1 and len(rows[0]) == len(fields)
        assert rows[0][-1] == "lesson::1 ä"
        assert rows[0][fields.index("Document")] == "Vorlesung ä"
    assert export(source, out).exit_code == 0
    assert before == {path.name: path.read_bytes() for path in out.iterdir()}
    write_rows(
        source, [row("vocab", meaning='neu, "Ä"\nZeile'), row("form", translation="neu"), row("qa", answer="neu")]
    )
    assert export(source, out).exit_code == 0
    for kind in AUTHORED_NOTE_TYPES:
        assert data(before[f"{kind}.csv"])[0][0] == data((out / f"{kind}.csv").read_bytes())[0][0]
        assert before[f"{kind}.csv"] != (out / f"{kind}.csv").read_bytes()


@pytest.mark.parametrize(
    "bad", [row("qa", status="skip", answer=""), row("form", analysis=""), "not json", row(key="conflict", meaning="b")]
)
def test_invalid_whole_file_leaves_existing_and_new_outputs_untouched(
    tmp_path: Path, bad: dict[str, object] | str
) -> None:
    source = write_rows(tmp_path / "notes.jsonl", [row(), row(key="conflict", meaning="a"), bad])
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "vocab.csv"
    existing.write_bytes(b"prior output")
    result = export(source, out, "--kind", "vocab", "--tag", "absent")
    assert result.exit_code == 1, result.output
    assert existing.read_bytes() == b"prior output"
    assert list(out.iterdir()) == [existing]
    missing = tmp_path / "missing"
    assert export(source, missing, "--kind", "vocab").exit_code == 1
    assert not missing.exists()


def test_combined_filters_and_empty_selection_no_writes(tmp_path: Path) -> None:
    provenance = {"document": "D", "section": "S", "reference": "R"}
    source = write_rows(
        tmp_path / "notes.jsonl",
        [
            row(tags=["chosen"], provenance=provenance),
            row("qa", tags=["chosen"], provenance=provenance),
            row(key="other", tags=["other"]),
            row("form", status="skip", tags=["chosen"], provenance=provenance),
        ],
    )
    out = tmp_path / "out"
    out.mkdir()
    result = export(
        source, out, "--kind", "vocab", "--kind", "form", "--section", "S", "--reference", "R", "--tag", "chosen"
    )
    assert result.exit_code == 0, result.output
    assert "Effective selection: 1" in result.output
    assert [path.name for path in out.iterdir()] == ["vocab.csv"]
    before = (out / "vocab.csv").read_bytes()
    result = export(source, out, "--tag", "missing")
    assert result.exit_code == 0 and "Empty selection" in result.output
    assert (out / "vocab.csv").read_bytes() == before
    missing = tmp_path / "missing"
    assert export(source, missing, "--tag", "missing").exit_code == 0
    assert not missing.exists()
