"""Public corpus validation contract, including recovery and read-only input."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from latinitas_cards.cli import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/corpus"
CASES = json.loads((FIXTURES / "invalid/cases.json").read_text())


def invoke(path: Path, machine: bool = True) -> Any:
    before = (path.read_bytes(), path.stat().st_mtime_ns) if path.exists() else None
    result = CliRunner().invoke(app, ["corpus", "validate", str(path), *(["--json"] if machine else [])])
    if before is not None:
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    return result


@pytest.mark.parametrize("name", ["ordinal", "usfx-bcv", "wsl"])
def test_valid(name: str) -> None:
    result = invoke(FIXTURES / f"valid/{name}.jsonl")
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.stdout)
    assert envelope == {
        "format": "latinitas-cli-result",
        "version": 1,
        "command": "corpus validate",
        "ok": True,
        "exit_code": 0,
        "diagnostics": [],
        "result": {"counts": {"header": 1, "corpus": 1, "work": 1 if name == "ordinal" else 2, "passage": 3}},
    }
    assert result.stderr == ""


@pytest.mark.parametrize("name", CASES)
def test_one_defect(name: str) -> None:
    path = FIXTURES / "invalid" / name
    result = invoke(path)
    assert result.exit_code == CASES[name]["exit"], result.output
    diagnostics = json.loads(result.stdout)["diagnostics"]
    assert {d["code"] for d in diagnostics} == set(CASES[name]["codes"])
    assert all(d["path"] == str(path) and d["item"] is None for d in diagnostics)
    assert result.stderr == ""


@pytest.mark.parametrize(
    "content,code",
    [
        (None, "unreadable"),
        ("[]\n", "invalid_json"),
        ("", "missing_header"),
        ('{"record":"corpus"}\n', "missing_header"),
        ('{"record":"header","format":"other","version":1}\n', "unsupported_format"),
        ('{"record":"header","format":"latinitas-corpus","version":2}\n', "unsupported_version"),
        ('{"record":"header","format":"latinitas-corpus","version":1.0}\n', "unsupported_version"),
    ],
)
def test_gate(tmp_path: Path, content: str | None, code: str) -> None:
    path = tmp_path / "corpus.jsonl"
    if content is not None:
        path.write_text(content)
    result = invoke(path)
    assert result.exit_code == 3, result.output
    assert [d["code"] for d in json.loads(result.stdout)["diagnostics"]] == [code]


def rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (FIXTURES / "valid/ordinal.jsonl").read_text().splitlines()]


def write(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))


@pytest.mark.parametrize("late_gate", [False, True])
def test_all_findings_context_and_text(tmp_path: Path, late_gate: bool) -> None:
    records = rows()
    records[3]["passage_hash"] = "sha256:" + "0" * 64
    records[3]["norm"] = "wrong"
    records[4]["citation"]["parts"][0]["value"] = 7
    records[4]["passage_id"] = "unusable"
    records[5]["norm"] = "wrong"
    path = tmp_path / "multi.jsonl"
    write(path, records)
    if late_gate:
        with path.open("a") as stream:
            stream.write("not JSON\n")
    result = invoke(path)
    assert result.exit_code == (3 if late_gate else 1), result.output
    diagnostics = json.loads(result.stdout)["diagnostics"]
    assert [d["code"] for d in diagnostics] == [
        "hash_mismatch",
        "norm_mismatch",
        "invalid_citation",
        "passage_id_mismatch",
        "norm_mismatch",
        *(["invalid_json"] if late_gate else []),
    ]
    assert [d["ref"] for d in diagnostics[:5]] == [
        "test-corpus:p000001",
        "test-corpus:p000001",
        "p000002",
        "p000002",
        "test-corpus:p000003",
    ]
    text = invoke(path, False)
    assert text.exit_code == result.exit_code
    assert text.stdout == f"{len(diagnostics)} errors, 0 warnings\n"
    assert text.stderr == "".join(
        f"{d['severity']} {d['code']} {d['ref'] or '-'}: {d['message']}\n" for d in diagnostics
    )


def test_encoded_refs_and_streaming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = rows()
    for row, ref in zip(records[3:], ["a.%2E", "a.%252E", "%C3%A9"], strict=True):
        row.update(native_ref=ref, passage_id=f"test-corpus:{ref}")
    path = tmp_path / "encoded.jsonl"
    write(path, records)
    assert invoke(path).exit_code == 0

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("whole-file read")

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    result = CliRunner().invoke(app, ["corpus", "validate", str(FIXTURES / "valid/usfx-bcv.jsonl"), "--json"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("work", [[], {}, None, 17])
def test_invalid_work_is_a_finding_not_internal(tmp_path: Path, work: Any) -> None:
    records = rows()
    records[3]["work"] = work
    path = tmp_path / "bad-work.jsonl"
    write(path, records)
    result = invoke(path)
    assert result.exit_code == 1, result.output
    assert {d["code"] for d in json.loads(result.stdout)["diagnostics"]} == {"invalid_record"}


def test_distinct_same_code_findings_are_retained(tmp_path: Path) -> None:
    records = rows()
    records[3]["citation"]["scheme"] = 2
    records[3]["citation"]["parts"][0]["value"] = 1
    records[4]["work"] = "unknown-work"
    records[4]["order"] = 99
    path = tmp_path / "same-code.jsonl"
    write(path, records)
    result = invoke(path)
    assert result.exit_code == 1, result.output
    diagnostics = json.loads(result.stdout)["diagnostics"]
    assert [d["code"] for d in diagnostics] == [
        "invalid_citation",
        "invalid_citation",
        "invalid_record",
        "invalid_record",
    ]
    assert "citation.scheme" in diagnostics[0]["message"]
    assert "citation.parts.0.value" in diagnostics[1]["message"]
    assert "unknown work" in diagnostics[2]["message"]
    assert "order 2" in diagnostics[3]["message"]


def test_float_does_not_hide_independent_schema_errors(tmp_path: Path) -> None:
    records = rows()
    records[3]["source_extra"] = {"number": 1.5}
    records[3]["citation"]["scheme"] = 2
    path = tmp_path / "float-and-schema.jsonl"
    write(path, records)
    result = invoke(path)
    assert result.exit_code == 1, result.output
    assert [d["code"] for d in json.loads(result.stdout)["diagnostics"]] == ["float_value", "invalid_citation"]


@pytest.mark.parametrize("field", ["refs", "text", "notes", "source_extra"])
@pytest.mark.parametrize("machine", [False, True])
def test_lone_surrogate_is_a_safe_input_finding(tmp_path: Path, field: str, machine: bool) -> None:
    records = rows()
    if field == "refs":
        records[3].update(native_ref="\ud800", passage_id="test-corpus:\ud800")
    elif field == "notes":
        records[3][field] = ["\ud800"]
    elif field == "source_extra":
        records[3][field] = {"\ud800": "value"}
    else:
        records[3][field] = "\ud800"
    path = tmp_path / "surrogate.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in records))
    result = invoke(path, machine)
    assert result.exit_code == 1, (result.output, result.exception)
    (result.stdout + result.stderr).encode("utf-8")
    if machine:
        envelope = json.loads(result.stdout)
        assert "invalid_record" in {d["code"] for d in envelope["diagnostics"]}
        assert all(d["path"] == str(path) for d in envelope["diagnostics"])
        assert result.stderr == ""
    else:
        assert "invalid_record" in result.stderr
        assert "internal" not in result.stderr
