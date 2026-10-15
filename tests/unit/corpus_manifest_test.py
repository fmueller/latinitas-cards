"""Ref lifetime and two-file recovery through the public normalization command."""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from latinitas_cards import atomic_io
from latinitas_cards.cli import app


def invoke(source: Path, output: Path, *flags: str) -> Result:
    return CliRunner().invoke(
        app,
        [
            "corpus",
            "normalize",
            str(source),
            "--adapter",
            "text",
            "--corpus-id",
            "test-corpus",
            "--output",
            str(output),
            "--json",
            *flags,
        ],
    )


@pytest.mark.parametrize("failure", [0, 1, 2])
def test_approved_transition_recovers_without_consuming_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: int,
) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "corpus.jsonl"
    manifest = tmp_path / "corpus.manifest.json"
    source.write_text("arma\nvirum\ncano\n")
    assert invoke(source, output).exit_code == 0
    before = output.read_bytes(), manifest.read_bytes()
    source.write_text("arma\narma\ncano mutatum\n")
    flags = (
        "--approve-reuse",
        "0=p000001",
        "--approve-allocation",
        "1",
        "--approve-reuse",
        "2=p000003",
        "--approve-removal",
        "p000002",
    )
    assert invoke(source, output, *flags).exit_code == 0
    expected = output.read_bytes(), manifest.read_bytes()
    output.write_bytes(before[0])
    manifest.write_bytes(before[1])
    original = atomic_io.replace_staged
    calls = 0

    def interrupt(path: Path, staged: Path) -> None:
        nonlocal calls
        if calls == failure:
            if failure == 2:
                raise AssertionError("unreachable")
            raise OSError("injected replacement failure")
        calls += 1
        original(path, staged)
        if failure == 2 and calls == 2:
            raise OSError("injected post-corpus failure")

    monkeypatch.setattr(atomic_io, "replace_staged", interrupt)
    result = invoke(source, output, *flags)
    assert result.exit_code == 3, result.output
    if failure == 0:
        assert (output.read_bytes(), manifest.read_bytes()) == before
    monkeypatch.setattr(atomic_io, "replace_staged", original)
    result = invoke(source, output, *flags)
    assert result.exit_code == 0, result.output
    assert (output.read_bytes(), manifest.read_bytes()) == expected
    committed = json.loads(manifest.read_bytes())
    assert committed["next_ref"] == 5
    assert [e["ref"] for e in committed["entries"] if not e["active"]] == ["p000002"]
    inodes = output.stat().st_ino, manifest.stat().st_ino
    assert invoke(source, output, *flags).exit_code == 0
    assert (output.stat().st_ino, manifest.stat().st_ino) == inodes
    source.write_text("arma\narma\ncano mutatum\nnovum\n")
    assert (
        invoke(
            source, output, "--approve-reuse", "0=p000001", "--approve-reuse", "1=p000004", "--approve-allocation", "3"
        ).exit_code
        == 0
    )
    assert json.loads(manifest.read_bytes())["next_ref"] == 6


def test_review_does_not_change_files_and_retirement_is_permanent(tmp_path: Path) -> None:
    source, output = tmp_path / "source.txt", tmp_path / "corpus.jsonl"
    source.write_text("a\nb\nc\n")
    assert invoke(source, output).exit_code == 0
    manifest = output.with_suffix(".manifest.json")
    before = output.read_bytes(), manifest.read_bytes()
    source.write_text("a\nc\nnew\n")
    result = invoke(source, output)
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["result"]["review_items"] == [
        {"kind": "inserted", "ordinal": 2, "candidates": []},
        {"kind": "removed", "ordinal": None, "candidates": ["p000002"]},
    ]
    assert (output.read_bytes(), manifest.read_bytes()) == before
    assert invoke(source, output, "--approve-allocation", "2", "--approve-removal", "p000002").exit_code == 0
    assert json.loads(manifest.read_bytes())["next_ref"] == 5
    result = invoke(source, output, "--approve-reuse", "2=p000002")
    assert result.exit_code == 3
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "retired_ref"


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize(
    "change", ["source", "basename", "descriptor", "id", "language", "approvals", "bytes", "digest", "none"]
)
def test_pending_binding_and_first_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first: bool,
    change: str,
) -> None:
    source, output = tmp_path / "source.txt", tmp_path / "out.jsonl"
    manifest = output.with_suffix(".manifest.json")
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text('{"format":"latinitas-corpus-descriptor","version":1}')
    flags = ["--descriptor", str(descriptor)]
    source.write_text("a\nb\n")
    if not first:
        assert invoke(source, output, *flags).exit_code == 0
        source.write_text("a\nbb\n")
        flags += ["--approve-reuse", "1=p000002"]
    original = atomic_io.replace_staged

    def fail_corpus(path: Path, staged: Path) -> None:
        if path == output:
            raise OSError("interrupted")
        original(path, staged)

    monkeypatch.setattr(atomic_io, "replace_staged", fail_corpus)
    result = invoke(source, output, *flags)
    assert result.exit_code == 3, result.output
    assert "identical request" in result.output
    monkeypatch.setattr(atomic_io, "replace_staged", original)
    if change == "source":
        source.write_text("changed\n")
    elif change == "basename":
        renamed = tmp_path / "renamed.txt"
        renamed.write_bytes(source.read_bytes())
        source = renamed
    elif change == "descriptor":
        descriptor.write_text('{"format":"latinitas-corpus-descriptor","version":1,"title":"changed"}')
    elif change == "id":
        flags += ["--corpus-id", "other-corpus"]
    elif change == "language":
        flags += ["--language", "hr"]
    elif change == "approvals":
        flags += ["--approve-allocation", "0"]
    elif change == "bytes":
        output.write_text("foreign bytes")
    elif change == "digest":
        stored = json.loads(manifest.read_bytes())
        stored["materialization"]["corpus_digest"] = "sha256:" + "0" * 64
        manifest.write_text(json.dumps(stored))
    before = (output.read_bytes() if output.exists() else None), manifest.read_bytes()
    result = invoke(source, output, *flags)
    if change == "none":
        assert result.exit_code == 0, result.output
        assert json.loads(manifest.read_bytes())["next_ref"] == 3
        assert CliRunner().invoke(app, ["corpus", "validate", str(output)]).exit_code == 0
    else:
        assert result.exit_code == 4, result.output
        assert json.loads(result.stdout)["diagnostics"][0]["code"] == "binding_drift"
        assert ((output.read_bytes() if output.exists() else None), manifest.read_bytes()) == before
    assert not list(tmp_path.glob(".out.*"))


def test_equivalent_flags_and_relocated_paths_recover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "source.txt", tmp_path / "out.jsonl"
    source.write_text("a\nb\nc\n")
    assert invoke(source, output).exit_code == 0
    source.write_text("aa\nbb\n")
    flags = ["--approve-reuse", "0=p000001", "--approve-reuse", "1=p000002", "--approve-removal", "p000003"]
    original = atomic_io.replace_staged

    def fail(path: Path, staged: Path) -> None:
        if path == output:
            raise OSError("interrupted")
        original(path, staged)

    monkeypatch.setattr(atomic_io, "replace_staged", fail)
    assert invoke(source, output, *flags).exit_code == 3
    monkeypatch.setattr(atomic_io, "replace_staged", original)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for path in (source, output, output.with_suffix(".manifest.json")):
        (relocated / path.name).write_bytes(path.read_bytes())
    result = invoke(
        relocated / source.name,
        relocated / output.name,
        "--approve-removal",
        "p000003",
        "--approve-reuse",
        "1=p000002",
        "--approve-reuse",
        "0=p000001",
    )
    assert result.exit_code == 0, result.output
    assert json.loads((relocated / "out.manifest.json").read_bytes())["next_ref"] == 4


@pytest.mark.parametrize(
    "flags",
    [
        ["--approve-reuse", "0=p000001", "--approve-reuse", "0=p000001"],
        ["--approve-allocation", "0", "--approve-allocation", "0"],
        ["--approve-removal", "p000001", "--approve-removal", "p000001"],
        ["--approve-reuse", "0=p000001", "--approve-allocation", "0"],
        ["--approve-reuse", "0=p000001", "--approve-removal", "p000001"],
        ["--approve-reuse", "0=unknown"],
        ["--approve-allocation", "9"],
        ["--approve-reuse", "bad"],
    ],
)
def test_invalid_approvals_leave_committed_state(tmp_path: Path, flags: list[str]) -> None:
    source, output = tmp_path / "source.txt", tmp_path / "out.jsonl"
    source.write_text("a\nb\n")
    assert invoke(source, output).exit_code == 0
    manifest = output.with_suffix(".manifest.json")
    before = output.read_bytes(), manifest.read_bytes()
    result = invoke(source, output, *flags)
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "invalid_approval"
    assert (output.read_bytes(), manifest.read_bytes()) == before


def test_manifest_directory_fsync_failure_still_instructs_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "source.txt", tmp_path / "out.jsonl"
    source.write_text("a\n")
    original = atomic_io.replace_staged

    def fail_after_manifest(path: Path, staged: Path) -> None:
        original(path, staged)
        raise OSError("directory fsync failed after rename")

    monkeypatch.setattr(atomic_io, "replace_staged", fail_after_manifest)
    result = invoke(source, output)
    assert result.exit_code == 3
    assert "rerun the identical request" in result.output
    assert output.with_suffix(".manifest.json").exists() and not output.exists()
    monkeypatch.setattr(atomic_io, "replace_staged", original)
    assert invoke(source, output).exit_code == 0


def test_failed_second_staging_fsync_preserves_both_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "source.txt", tmp_path / "out.jsonl"
    source.write_text("a\nb\n")
    assert invoke(source, output).exit_code == 0
    manifest = output.with_suffix(".manifest.json")
    before = output.read_bytes(), manifest.read_bytes()
    source.write_text("a\nbb\n")
    original = os.fsync
    calls = 0

    def fail_second(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("second fsync failed")
        original(fd)

    monkeypatch.setattr(os, "fsync", fail_second)
    result = invoke(source, output, "--approve-reuse", "1=p000002")
    assert result.exit_code == 3, result.output
    assert (output.read_bytes(), manifest.read_bytes()) == before
    assert not list(tmp_path.glob(".out.*"))


@pytest.mark.parametrize("missing", ["format", "version"])
def test_manifest_requires_versioned_envelope(tmp_path: Path, missing: str) -> None:
    source, output = tmp_path / "source.txt", tmp_path / "out.jsonl"
    source.write_text("a\nb\n")
    assert invoke(source, output).exit_code == 0
    manifest = output.with_suffix(".manifest.json")
    stored = json.loads(manifest.read_bytes())
    del stored[missing]
    manifest.write_text(json.dumps(stored))
    before = output.read_bytes(), manifest.read_bytes()
    result = invoke(source, output)
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "invalid_manifest"
    assert (output.read_bytes(), manifest.read_bytes()) == before


@pytest.mark.parametrize("change", ["shorter", "longer", "hash"])
def test_unreproducible_committed_assignments_are_binding_drift(tmp_path: Path, change: str) -> None:
    source, output = tmp_path / "source.txt", tmp_path / "out.jsonl"
    source.write_text("a\nb\n")
    assert invoke(source, output).exit_code == 0
    manifest = output.with_suffix(".manifest.json")
    stored = json.loads(manifest.read_bytes())
    if change == "shorter":
        stored["entries"].pop()
        stored["snapshot"].pop()
    elif change == "longer":
        stored["entries"].append({**stored["entries"][0], "ref": "p000003", "ordinal": 2})
        stored["snapshot"].append(stored["snapshot"][0])
        stored["next_ref"] = 4
    else:
        stored["snapshot"][0] = stored["entries"][0]["passage_hash"] = "sha256:" + "0" * 64
    manifest.write_text(json.dumps(stored))
    output.unlink()  # Pending recovery, including the absent-corpus first-creation case.
    before = manifest.read_bytes()
    result = invoke(source, output)
    assert result.exit_code == 4, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "binding_drift"
    assert not output.exists() and manifest.read_bytes() == before
