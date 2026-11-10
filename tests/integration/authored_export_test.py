import csv
import errno
import io
import json
import os
import tempfile
from pathlib import Path
from typing import IO

import pytest
from authored_import_test import row, write_rows
from typer.testing import CliRunner, Result

from latinitas_cards.authored_notes import AUTHORED_NOTE_TYPES
from latinitas_cards.cli import app


@pytest.mark.parametrize("fail_at", [1, 2, 3])
@pytest.mark.parametrize("present", [(), (0, 1, 2), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("failure", ["create", "write"])
def test_early_staging_failure_preserves_durable_directory_and_successful_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int, present: tuple[int, ...], failure: str
) -> None:
    source = write_rows(tmp_path / "notes.jsonl", [row(kind) for kind in ("vocab", "form", "qa")])
    source_before = source.read_bytes()
    out = tmp_path / "out"
    out.mkdir()
    for index, kind in enumerate(("vocab", "form", "qa")):
        if index in present:
            (out / f"{kind}.csv").write_bytes(bytes([index, 255, 0, 128]) + b"prior\r\n")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    calls = 0
    original_mkstemp = tempfile.mkstemp
    original_fdopen = os.fdopen

    def fail_creation(*, prefix: str, suffix: str, dir: Path) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError(errno.ENOSPC, "injected staging creation failure")
        return original_mkstemp(prefix=prefix, suffix=suffix, dir=dir)

    def fail_write(payload: bytes) -> int:
        raise OSError(errno.ENOSPC, "injected staging write failure")

    def open_stage(descriptor: int, mode: str) -> IO[bytes]:
        nonlocal calls
        calls += 1
        stream = original_fdopen(descriptor, mode)
        if calls == fail_at:
            monkeypatch.setattr(stream, "write", fail_write)
        return stream

    with monkeypatch.context() as injection:
        if failure == "create":
            injection.setattr("latinitas_cards.preview_export.tempfile.mkstemp", fail_creation)
        else:
            injection.setattr("latinitas_cards.preview_export.os.fdopen", open_stage)
        result = export(source, out)
    assert calls == fail_at
    assert result.exit_code == 1, result.output
    assert "No output or committed state was changed" in result.output
    assert source.read_bytes() == source_before
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert export(source, out).exit_code == 0
    successful = {p.name: p.read_bytes() for p in out.iterdir()}
    assert set(successful) == {"vocab.csv", "form.csv", "qa.csv"}
    assert all(len(data(payload)) == 1 for payload in successful.values())
    assert export(source, out).exit_code == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == successful
    assert source.read_bytes() == source_before


def test_staging_failure_does_not_remove_concurrently_created_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_rows(tmp_path / "notes.jsonl", [row(kind) for kind in ("vocab", "form", "qa")])
    source_before = source.read_bytes()
    out = tmp_path / "out"
    out.mkdir()
    (out / "vocab.csv").write_bytes(b"prior\x00\xff")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    concurrent_bytes = b"concurrent\x00\xfe"
    calls = 0
    original_mkstemp = tempfile.mkstemp

    def create_concurrently_then_fail(*, prefix: str, suffix: str, dir: Path) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            (out / "form.csv").write_bytes(concurrent_bytes)
        if calls == 2:
            raise OSError(errno.ENOSPC, "injected staging failure after concurrent creation")
        return original_mkstemp(prefix=prefix, suffix=suffix, dir=dir)

    monkeypatch.setattr("latinitas_cards.preview_export.tempfile.mkstemp", create_concurrently_then_fail)
    result = export(source, out)
    assert calls == 2
    assert result.exit_code == 1, result.output
    assert "No output or committed state was changed" in result.output
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before | {"form.csv": concurrent_bytes}
    assert source.read_bytes() == source_before


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


@pytest.mark.parametrize(
    ("directory", "display"),
    [
        ("ordinary ä", "ordinary ä"),
        ("out\x1b]0;OWNED\x07", r"out\x1b]0;OWNED\x07"),
        ("out\n\r\t\u2028\u2029", r"out\x0a\x0d\x09\u2028\u2029"),
    ],
)
def test_export_status_escapes_path_without_changing_destination_or_csv(
    tmp_path: Path, directory: str, display: str
) -> None:
    source = write_rows(tmp_path / "notes.jsonl", [row(kind) for kind in ("vocab", "form", "qa")])
    source_bytes = source.read_bytes()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    assert export(source, baseline).exit_code == 0
    out = tmp_path / directory
    out.mkdir()
    result = export(source, out)
    assert result.exit_code == 0, repr(result.output)
    expected_status = [f"Wrote {tmp_path / display / f'{kind}.csv'}" for kind in ("vocab", "form", "qa")]
    assert result.output.splitlines()[-3:] == expected_status, repr(result.output)
    assert all(character not in result.output for character in "\x1b\x07\r\t\u2028\u2029")
    assert {path.name for path in out.iterdir()} == {"vocab.csv", "form.csv", "qa.csv"}
    for path in baseline.iterdir():
        assert (out / path.name).read_bytes() == path.read_bytes()
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize("command", ["validate", "preview", "export"])
@pytest.mark.parametrize("existing", [False, True])
def test_surrogate_cli_errors_aggregate_before_selection_and_preserve_bytes(
    tmp_path: Path, command: str, existing: bool
) -> None:
    source = write_rows(
        tmp_path / "notes.jsonl",
        [
            json.dumps(row(meaning="\ud800", status="skip")),
            json.dumps(row("form", context="\udfff")),
            row("qa", key="later"),
        ],
    )
    before = source.read_bytes()
    out = tmp_path / "out"
    out.mkdir()
    if existing:
        (out / "vocab.csv").write_bytes(b"prior output\x00\xff")
    outputs = {p.name: p.read_bytes() for p in out.iterdir()}
    args = ["authored", command, str(source), "--namespace", "course"]
    if command != "validate":
        args += ["--kind", "qa"]
    if command == "export":
        args += ["--output-dir", str(out), "--deck", "Latin"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    assert "line 1, vocab.meaning" in result.output
    assert "line 2, form.context" in result.output
    assert "Invalid: 2 errors" in result.output and "not exportable" in result.output
    assert "'qa': 1" in result.output
    assert "Traceback" not in result.output and "Authored export error" not in result.output
    result.output.encode("utf-8")
    assert source.read_bytes() == before
    assert {p.name: p.read_bytes() for p in out.iterdir()} == outputs


@pytest.mark.parametrize("escaped", [False, True])
def test_unicode_cli_roundtrip(tmp_path: Path, escaped: bool) -> None:
    value = "𐌀🌹"
    source = write_rows(tmp_path / "notes.jsonl", [json.dumps(row(meaning=value), ensure_ascii=escaped)])
    before = source.read_bytes()
    out = tmp_path / "out"
    out.mkdir()
    for command in ("validate", "preview"):
        result = CliRunner().invoke(app, ["authored", command, str(source), "--namespace", "course"])
        assert result.exit_code == 0, result.output
        assert "Valid. Effective selection: 1" in result.output
    assert export(source, out).exit_code == 0
    assert value in (out / "vocab.csv").read_text(encoding="utf-8")
    assert source.read_bytes() == before


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
