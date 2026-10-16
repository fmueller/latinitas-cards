"""Real segment command, retry/timestamp and persistence contracts."""

import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from latinitas_cards.annotation_sidecar import annotation_id
from latinitas_cards.atomic_io import writer_lock
from latinitas_cards.canonical import canonical_json
from latinitas_cards.cli import app
from latinitas_cards.cli_contract import CliResult

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CORPUS = FIXTURES / "corpus/valid/ordinal.jsonl"
STAMP = "2026-01-01T00:00:00Z"


def segment(output: Path, *extra: str, corpus: Path = CORPUS) -> CliResult:
    result = CliRunner().invoke(app, ["annotations", "segment", str(corpus), "--output", str(output), *extra, "--json"])
    assert result.stdout.count("\n") == 1, result.output
    envelope = CliResult.model_validate_json(result.stdout)
    assert result.exit_code == envelope.exit_code
    assert envelope.command == "annotations segment"
    return envelope


def test_segment_write_retry_and_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    output = tmp_path / "sidecar.jsonl"
    result = segment(output, "--timestamp", STAMP)
    assert result.exit_code == 0
    assert result.result == {"appended": 3, "skipped": 0}
    before, inode = output.read_bytes(), output.stat().st_ino
    rows = [json.loads(line) for line in before.splitlines()]
    assert rows[0] == {"format": "latinitas-annotations", "record": "header", "version": 1}
    assert [row["anchor"]["passage_id"] for row in rows[1:]] == [f"test-corpus:p00000{i}" for i in (1, 2, 3)]
    assert rows[2]["value"] == {
        "tokenizer": "latin-tok/1",
        "sentences": [{"continued": False, "continues": False, "span": [0, 18], "tokens": [0, 4]}],
        "tokens": [
            {"kind": "word", "span": [0, 5]},
            {"kind": "word", "span": [6, 12]},
            {"kind": "word", "span": [13, 17]},
            {"kind": "punct", "span": [17, 18]},
        ],
    }
    for row in rows[1:]:
        assert row["producer"] == {"kind": "rule", "name": "latin-seg", "version": "1"}
        assert row["plan"] is None and row["created"] == STAMP
        assert set(row["anchor"]) == {"passage_id", "passage_hash"}
    assert before == (FIXTURES / "annotations/ordinal.annotations.jsonl").read_bytes()
    assert segment(output, "--timestamp", STAMP).result == {"appended": 0, "skipped": 3}
    assert segment(output).exit_code == 0  # no-op needs no clock
    assert output.read_bytes() == before and output.stat().st_ino == inode
    other = tmp_path / "other.jsonl"
    assert segment(other, "--timestamp", "2027-06-30T12:00:00Z").exit_code == 0
    assert [json.loads(line)["annotation_id"] for line in other.read_text().splitlines()[1:]] == [
        row["annotation_id"] for row in rows[1:]
    ]
    assert not {"cltk", "stanza", "torch"}.intersection(sys.modules)


@pytest.mark.parametrize("stamp", [None, "2026-01-01", "2026-01-01T01:00:00+01:00", "2026-02-30T00:00:00Z"])
def test_timestamp_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stamp: str | None) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    output = tmp_path / "out.jsonl"
    result = segment(output, *(["--timestamp", stamp] if stamp is not None else []))
    assert result.exit_code == 2
    assert result.diagnostics[0].code == "timestamp_required"
    if stamp:
        assert stamp in result.diagnostics[0].message
    assert not output.exists()


@pytest.mark.parametrize("epoch", ["0", "-1", "1.0", "", "+1", "999999999999999999999999"])
def test_source_date_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, epoch: str) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", epoch)
    output = tmp_path / "out.jsonl"
    result = segment(output)
    if epoch == "0":
        assert result.exit_code == 0
        assert json.loads(output.read_text().splitlines()[1])["created"] == "1970-01-01T00:00:00Z"
    else:
        assert result.exit_code == 2
        assert epoch in result.diagnostics[0].message
        assert not output.exists()
    assert segment(output, "--timestamp", STAMP).exit_code == 0  # explicit flag wins


def test_default_output_and_continuation(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("arma\nvirum.\n")
    corpus = tmp_path / "corpus.jsonl"
    result = CliRunner().invoke(
        app, ["corpus", "normalize", str(source), "--adapter", "text", "--corpus-id", "poem", "--output", str(corpus)]
    )
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(app, ["annotations", "segment", str(corpus), "--timestamp", STAMP, "--json"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in corpus.with_suffix(".annotations.jsonl").read_text().splitlines()[1:]]
    assert rows[0]["value"]["sentences"][0]["continues"] is True
    assert rows[1]["value"]["sentences"][0]["continued"] is True


def test_busy_even_on_noop(tmp_path: Path) -> None:
    output = tmp_path / "sidecar.jsonl"
    assert segment(output, "--timestamp", STAMP).exit_code == 0
    before = output.read_bytes()
    with writer_lock(output):
        result = segment(output, "--timestamp", STAMP)
    assert result.exit_code == 4 and result.diagnostics[0].code == "busy"
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "kind", ["direct", "symlink", "hardlink", "claims", "manifest", "lock-symlink", "lock-hardlink"]
)
def test_alias_no_write(tmp_path: Path, kind: str) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(CORPUS.read_bytes())
    output = tmp_path / "sidecar.jsonl"
    if kind == "direct":
        output = corpus
    elif kind == "symlink":
        output.symlink_to(corpus)
    elif kind == "hardlink":
        os.link(corpus, output)
    elif kind == "claims":
        output = Path(f"{corpus}.claims.json")
    elif kind == "manifest":
        output = corpus.with_suffix(".manifest.json")
    elif kind == "lock-symlink":
        Path(f"{output}.lock").symlink_to(corpus)
    else:
        os.link(corpus, Path(f"{output}.lock"))
    before = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    result = segment(output, "--timestamp", STAMP, corpus=corpus)
    assert result.exit_code == 3 and result.diagnostics[0].code == "alias"
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not output.exists() or output in before


def test_replace_failure_keeps_previous_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "sidecar.jsonl"
    output.write_text('{"format":"latinitas-annotations","record":"header","version":1}\n')
    before = output.read_bytes()

    def fail_replace(source: object, target: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr("latinitas_cards.atomic_io.os.replace", fail_replace)
    result = segment(output, "--timestamp", STAMP)
    assert result.exit_code == 3 and result.diagnostics[0].code == "unreadable"
    assert output.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sidecar.jsonl", "sidecar.jsonl.lock"]


@pytest.mark.parametrize(
    ("key", "value", "code"),
    [
        ("format", "foreign", "unsupported_format"),
        ("version", 2, "unsupported_version"),
        ("version", 1.0, "unsupported_version"),
    ],
)
def test_unsupported_sidecar_no_write(tmp_path: Path, key: str, value: object, code: str) -> None:
    output = tmp_path / "sidecar.jsonl"
    output.write_text(
        json.dumps({"format": "latinitas-annotations", "record": "header", "version": 1, key: value}) + "\n"
    )
    before = output.read_bytes()
    result = segment(output, "--timestamp", STAMP)
    assert result.exit_code == 3 and result.diagnostics[0].code == code
    assert output.read_bytes() == before


def test_invalid_corpus_layered_diagnostics(tmp_path: Path) -> None:
    output = tmp_path / "sidecar.jsonl"
    result = segment(output, "--timestamp", STAMP, corpus=FIXTURES / "corpus/invalid/hash.jsonl")
    assert result.exit_code == 3
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["invalid_corpus", "hash_mismatch"]
    assert "hash_mismatch" in result.diagnostics[0].message
    assert not output.exists()


def test_corpus_gate_preserves_earlier_findings(tmp_path: Path) -> None:
    corpus = tmp_path / "bad.jsonl"
    corpus.write_text((FIXTURES / "corpus/invalid/hash.jsonl").read_text() + "{broken\n")
    result = segment(tmp_path / "sidecar.jsonl", "--timestamp", STAMP, corpus=corpus)
    assert result.exit_code == 3
    assert [d.code for d in result.diagnostics] == ["invalid_corpus", "hash_mismatch", "invalid_json"]


def test_retains_history_and_warns_on_different_current_segmentation(tmp_path: Path) -> None:
    output = tmp_path / "sidecar.jsonl"
    assert segment(output, "--timestamp", STAMP).exit_code == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    # A valid alternate sentence partition is a current proposal, not permission to replace it.
    row = rows[1]
    row["value"]["sentences"][0]["continues"] = True
    row["annotation_id"] = annotation_id(row)
    output.write_text("".join(canonical_json(row) + "\n" for row in rows))
    before = output.read_bytes()
    result = segment(output)
    assert result.exit_code == 0 and result.result == {"appended": 0, "skipped": 3}
    assert [d.code for d in result.diagnostics] == ["segmentation_differs"]
    assert result.diagnostics[0].severity == "warning"
    assert output.read_bytes() == before
    # Retract the alternate record: the standard record may now be appended, after all history.
    retraction = {key: value for key, value in row.items() if key != "value"}
    retraction.update(record="retraction", retracts=row["annotation_id"])
    retraction["annotation_id"] = annotation_id(retraction)
    with output.open("a") as stream:
        stream.write(canonical_json(retraction) + "\n")
    before = output.read_bytes()
    result = segment(output, "--timestamp", STAMP)
    assert result.exit_code == 0 and result.result == {"appended": 1, "skipped": 2}
    assert output.read_bytes().startswith(before)


def test_retracted_duplicate_id_is_skipped_without_timestamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    output = tmp_path / "sidecar.jsonl"
    assert segment(output, "--timestamp", STAMP).exit_code == 0
    row = json.loads(output.read_text().splitlines()[1])
    retraction = {key: value for key, value in row.items() if key != "value"}
    retraction.update(record="retraction", retracts=row["annotation_id"])
    retraction["annotation_id"] = annotation_id(retraction)
    with output.open("a") as stream:
        stream.write(canonical_json(retraction) + "\n")
    before = output.read_bytes()
    result = segment(output)
    assert result.exit_code == 0 and result.result == {"appended": 0, "skipped": 3}
    assert output.read_bytes() == before


def test_invalid_explicit_timestamp_rejected_even_on_noop(tmp_path: Path) -> None:
    output = tmp_path / "sidecar.jsonl"
    assert segment(output, "--timestamp", STAMP).exit_code == 0
    before = output.read_bytes()
    result = segment(output, "--timestamp", "2026-01-01")
    assert result.exit_code == 2 and result.diagnostics[0].code == "timestamp_required"
    assert output.read_bytes() == before
