"""Real segment command, retry/timestamp and persistence contracts."""

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from latinitas_cards.annotation_sidecar import Annotation, Retraction, annotation_id, append_records
from latinitas_cards.atomic_io import writer_lock
from latinitas_cards.canonical import canonical_json
from latinitas_cards.cli import app
from latinitas_cards.cli_contract import CliResult

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CORPUS = FIXTURES / "corpus/valid/ordinal.jsonl"
STAMP = "2026-01-01T00:00:00Z"


@pytest.mark.parametrize("state", ["held", "released", "absent"])
@pytest.mark.parametrize("location", ["output", "output-lock"])
@pytest.mark.parametrize("kind", ["direct", "symlink", "hardlink", "chained", "directory"])
def test_segment_resolved_default_manifest_lock(tmp_path: Path, state: str, location: str, kind: str) -> None:
    if state == "absent" and kind == "hardlink":
        pytest.skip("Hardlinks require an existing inode")
    source = tmp_path / "source.txt"
    source.write_text("Canis currit.\nFemina librum legit.\n")
    corpus = tmp_path / "corpus.jsonl"
    normalize = [
        "corpus",
        "normalize",
        str(source),
        "--adapter",
        "text",
        "--corpus-id",
        "segment",
        "--output",
        str(corpus),
        "--json",
    ]

    def invoke(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "latinitas_cards", *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    assert invoke(normalize).returncode == 0
    manifest = corpus.with_suffix(".manifest.json")
    data = tmp_path / "manifest-data.json"
    manifest.rename(data)
    manifest.symlink_to(data.name)
    assert invoke(normalize).returncode == 0  # Supported default-manifest DATA symlink.
    reserved = Path(f"{data}.lock")
    control = tmp_path / "control.jsonl"
    segment_args = ["annotations", "segment", str(corpus), "--timestamp", STAMP, "--json", "--output"]
    control_result = invoke([*segment_args, str(control)])
    assert control_result.returncode == 0
    assert json.loads(control_result.stdout)["result"] == {"appended": 2, "skipped": 0}
    seed = control.read_bytes().splitlines(keepends=True)[0]
    if state == "absent":
        reserved.unlink()
    else:
        reserved.write_bytes(seed)  # Valid header makes accidental output appends observable.
    output = tmp_path / "side.jsonl"
    alias = output if location == "output" else Path(f"{output}.lock")
    if kind == "direct":
        output = reserved if location == "output" else data
    elif kind == "hardlink":
        os.link(reserved, alias)
    elif kind == "chained":
        hop = tmp_path / "hop"
        hop.symlink_to(reserved.name)
        alias.symlink_to(hop.name)
    elif kind == "directory":
        directory = tmp_path / "linked-directory"
        directory.symlink_to(tmp_path, target_is_directory=True)
        alias.symlink_to(directory / reserved.name)
    else:
        alias.symlink_to(reserved.name)

    before = {p: (p.read_bytes(), p.stat().st_ino) for p in tmp_path.iterdir() if p.is_file()}
    links = {p: os.readlink(p) for p in tmp_path.iterdir() if p.is_symlink()}
    entries = set(tmp_path.iterdir())
    with reserved.open("ab") if state == "held" else open(os.devnull, "ab") as held:
        if state == "held":
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert invoke(normalize).returncode == 4
        result = invoke([*segment_args, str(output)])
        if state == "held":
            assert invoke(normalize).returncode == 4
            # Unrelated sidecars still work while the manifest writer is excluded.
            ordinary = invoke([*segment_args, str(control)])
            assert ordinary.returncode == 0
            assert json.loads(ordinary.stdout)["result"] == {"appended": 0, "skipped": 2}
    assert result.returncode == 3 and json.loads(result.stdout)["diagnostics"][0]["code"] == "alias"
    assert set(tmp_path.iterdir()) == entries
    assert {p: (p.read_bytes(), p.stat().st_ino) for p in tmp_path.iterdir() if p.is_file()} == before
    assert {p: os.readlink(p) for p in tmp_path.iterdir() if p.is_symlink()} == links
    if state == "absent":
        assert not reserved.exists()
    assert invoke(normalize).returncode == 0  # Release really restores normalizer access.


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


@pytest.mark.parametrize("kind", ["direct", "symlink", "hardlink", "chained", "directory"])
@pytest.mark.parametrize("reserved", ["corpus", "claims", "symlinked-claims", "manifest"])
def test_segment_preserves_held_reserved_lock(tmp_path: Path, reserved: str, kind: str) -> None:
    def run(*args: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "latinitas_cards", *map(str, args), "--json"],
            cwd=tmp_path,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
        )

    source = tmp_path / "source.txt"
    source.write_text("Mater filium amat.\n")
    corpus = tmp_path / "corpus.jsonl"
    normal = ("corpus", "normalize", source, "--adapter", "text", "--corpus-id", "lock-test", "--output", corpus)
    assert run(*normal).returncode == 0
    claims = Path(f"{corpus}.claims.json")
    if reserved == "symlinked-claims":
        decisions = tmp_path / "external.json"
        decisions.write_text('{"version":2,"decisions":[]}')
        claims.symlink_to(decisions)
    lock = Path(f"{corpus if reserved == 'corpus' else claims}.lock")
    if reserved == "manifest":
        lock = Path(f"{corpus.with_suffix('.manifest.json')}.lock")
    seed = b'{"format":"latinitas-annotations","record":"header","version":1}\n'
    lock.write_bytes(seed)
    output = tmp_path / "alias.jsonl"
    if kind == "direct":
        output = lock
    elif kind == "symlink":
        output.symlink_to(lock)
    elif kind == "hardlink":
        os.link(lock, output)
    elif kind == "chained":
        intermediate = tmp_path / "intermediate"
        intermediate.symlink_to(lock)
        output.symlink_to(intermediate)
    else:
        directory = tmp_path / "directory"
        directory.symlink_to(tmp_path, target_is_directory=True)
        output = directory / lock.name
    with lock.open("ab") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inode = os.fstat(held.fileno()).st_ino
        if reserved in {"corpus", "manifest"}:
            assert run(*normal).returncode == 4
        result = run("annotations", "segment", corpus, "--output", output, "--timestamp", STAMP)
        after = lock.read_bytes()
        assert result.returncode == 3, result.stdout
        assert json.loads(result.stdout)["diagnostics"][0]["code"] == "alias"
        assert lock.stat().st_ino == inode and after == seed
        assert output.stat().st_ino == inode
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import fcntl,sys; fcntl.flock(open(sys.argv[1],'ab'),fcntl.LOCK_EX|fcntl.LOCK_NB)",
                str(lock),
            ],
            cwd=tmp_path,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert probe.returncode == 1 and "BlockingIOError" in probe.stderr
        if reserved in {"corpus", "manifest"}:
            assert run(*normal).returncode == 4
        assert (
            run(
                "annotations", "segment", corpus, "--output", tmp_path / "normal.jsonl", "--timestamp", STAMP
            ).returncode
            == 0
        )
        assert lock.stat().st_ino == inode and lock.read_bytes() == seed
    assert run(*normal).returncode == 0
    assert not Path(f"{output.resolve()}.lock").exists()
    if reserved == "symlinked-claims":
        assert claims.is_symlink() and decisions.read_text() == '{"version":2,"decisions":[]}'
        assert not Path(f"{decisions}.lock").exists()


@pytest.mark.parametrize("reserved", ["claims", "manifest"])
@pytest.mark.parametrize("kind", ["direct", "symlink", "chained", "directory", "output-lock"])
def test_segment_rejects_absent_reserved_lock(tmp_path: Path, reserved: str, kind: str) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(CORPUS.read_bytes())
    lock = Path(
        f"{corpus}.claims.json.lock" if reserved == "claims" else f"{corpus.with_suffix('.manifest.json')}.lock"
    )
    output = lock
    if kind in {"symlink", "chained", "output-lock"}:
        output = tmp_path / "alias.jsonl"
        alias = Path(f"{output}.lock") if kind == "output-lock" else output
        if kind == "chained":
            intermediate = tmp_path / "intermediate"
            intermediate.symlink_to(lock)
            alias.symlink_to(intermediate)
        else:
            alias.symlink_to(lock)
    elif kind == "directory":
        directory = tmp_path / "directory"
        directory.symlink_to(tmp_path, target_is_directory=True)
        output = directory / lock.name
    before = set(tmp_path.iterdir())
    result = segment(output, "--timestamp", STAMP, corpus=corpus)
    assert result.exit_code == 3 and result.diagnostics[0].code == "alias"
    assert set(tmp_path.iterdir()) == before and not lock.exists()


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


def test_inspect_provenance_and_check_severity() -> None:
    sidecar = FIXTURES / "annotations/wsl.annotations.jsonl"
    corpus = FIXTURES / "corpus/wsl-three-edited.corpus.jsonl"
    args = ["annotations", "inspect", str(sidecar), "--corpus", str(corpus), "--json"]
    warning = CliRunner().invoke(app, args)
    assert warning.exit_code == 0, warning.output
    checked = CliRunner().invoke(app, [*args, "--check"])
    assert checked.exit_code == 1, checked.output
    report, check = json.loads(warning.stdout), json.loads(checked.stdout)
    assert report["result"] == check["result"]
    originals = [json.loads(line) for line in sidecar.read_text().splitlines()[1:]]
    assert [row["annotation_id"] for row in report["result"]["annotations"]] == [
        row["annotation_id"] for row in originals
    ]
    assert [row["state"] for row in report["result"]["annotations"]] == ["current", "stale", "current"] * 2
    for row, original in zip(report["result"]["annotations"], originals, strict=True):
        assert row == {
            key: original[key] for key in ("annotation_id", "layer", "producer", "plan", "created", "input_hash")
        } | {
            "passage_id": original["anchor"]["passage_id"],
            "source_sha256": "sha256:871786f7c9077bb41e8cd000abee5e76b6c1046a5bbc27c9db1f8af3a634fd2b",
            "state": "stale" if original["anchor"]["passage_id"] == "wsl:poem.1.2" else "current",
            "superseded": False,
            "retracted": False,
        }
    refs = [originals[i]["annotation_id"] for i in (1, 4)]
    assert report["diagnostics"] == [
        {
            "code": "stale",
            "severity": "warning",
            "ref": ref,
            "path": str(sidecar),
            "item": None,
            "message": f"Annotation {ref} on passage wsl:poem.1.2: passage hash differs from the anchor",
        }
        for ref in refs
    ]
    assert check["diagnostics"] == [{**row, "severity": "error"} for row in report["diagnostics"]]
    control = CliRunner().invoke(
        app,
        [
            "annotations",
            "inspect",
            str(sidecar),
            "--corpus",
            str(FIXTURES / "corpus/wsl-three.corpus.jsonl"),
            "--check",
            "--json",
        ],
    )
    assert control.exit_code == 0 and json.loads(control.stdout)["diagnostics"] == []


@pytest.mark.parametrize(
    ("source", "code", "line"),
    [
        ("wsl-three-edited", "stale", "2"),
        ("wsl-two", "unmatched", "3"),
        ("supersede", "orphaned", "1"),
    ],
)
def test_real_anchor_workflow(tmp_path: Path, source: str, code: str, line: str) -> None:
    corpus, sidecar = tmp_path / "corpus.jsonl", tmp_path / "annotations.jsonl"
    normalize = [
        "corpus",
        "normalize",
        str(FIXTURES / "corpus/text/wsl-three.txt"),
        "--adapter",
        "text",
        "--citation-template",
        "wsl:work.section.line",
        "--corpus-id",
        "wsl",
        "--output",
        str(corpus),
        "--json",
    ]
    assert CliRunner().invoke(app, normalize).exit_code == 0
    assert corpus.read_bytes() == (FIXTURES / "corpus/wsl-three.corpus.jsonl").read_bytes()
    assert segment(sidecar, "--timestamp", STAMP, corpus=corpus).exit_code == 0
    morphs = [
        Annotation.model_validate_json(row)
        for row in (FIXTURES / "annotations/wsl-morphology.jsonl").read_text().splitlines()
    ]
    append_records(sidecar, lambda _: morphs, [corpus])
    assert sidecar.read_bytes() == (FIXTURES / "annotations/wsl.annotations.jsonl").read_bytes()
    if source == "supersede":
        replacement = Annotation.model_validate_json(
            (FIXTURES / "annotations/wsl-superseded.annotations.jsonl").read_text().splitlines()[-1]
        )
        append_records(sidecar, lambda _: [replacement], [corpus])
        assert sidecar.read_bytes() == (FIXTURES / "annotations/wsl-superseded.annotations.jsonl").read_bytes()
    else:
        normalize[2] = str(FIXTURES / f"corpus/text/{source}.txt")
        assert CliRunner().invoke(app, normalize).exit_code == 0
        if code == "stale":
            assert corpus.read_bytes() == (FIXTURES / "corpus/wsl-three-edited.corpus.jsonl").read_bytes()
    # Remove writer-created locks so the readers must prove they create none.
    for lock in tmp_path.glob("*.lock"):
        lock.unlink()
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()}
    args = ["annotations", "inspect", str(sidecar), "--corpus", str(corpus), "--json"]
    checked = CliRunner().invoke(app, [*args, "--check"])
    assert checked.exit_code == 1, checked.output
    report = json.loads(checked.stdout)
    original = [json.loads(row) for row in sidecar.read_text().splitlines()[1:]]
    affected = [
        row
        for row in original
        if row["anchor"]["passage_id"] == f"wsl:poem.1.{line}" and (code != "orphaned" or row["layer"] == "morphology")
    ]
    messages = {
        "stale": "passage hash differs from the anchor",
        "unmatched": "anchored passage is missing",
        "orphaned": "anchor segmentation is superseded, retracted, or unknown",
    }
    assert report["diagnostics"] == [
        {
            "code": code,
            "severity": "error",
            "path": str(sidecar),
            "ref": row["annotation_id"],
            "item": None,
            "message": f"Annotation {row['annotation_id']} on passage wsl:poem.1.{line}: {messages[code]}",
        }
        for row in affected
    ]
    affected_ids = {row["annotation_id"] for row in affected}
    assert all(
        row["state"] == (code if row["annotation_id"] in affected_ids else "current")
        for row in report["result"]["annotations"]
    )
    warning = CliRunner().invoke(app, args)
    assert warning.exit_code == 0
    assert json.loads(warning.stdout)["diagnostics"] == [
        {**row, "severity": "warning"} for row in report["diagnostics"]
    ]
    validated = CliRunner().invoke(app, ["corpus", "validate", str(corpus), "--annotations", str(sidecar), "--json"])
    assert validated.exit_code == 1
    assert json.loads(validated.stdout)["diagnostics"] == report["diagnostics"]
    shown = CliRunner().invoke(app, ["corpus", "show", str(corpus), "--annotations", str(sidecar), "--json"])
    assert shown.exit_code == 0
    visible = {row["passage_id"]: row["annotations"] for row in json.loads(shown.stdout)["result"]["passages"]}
    if code == "orphaned":
        assert [row["annotation_id"] for row in visible["wsl:poem.1.1"]] == [replacement.annotation_id]
    elif code == "stale":
        assert visible["wsl:poem.1.2"] == []
    else:
        assert "wsl:poem.1.3" not in visible
    for number in {"1", "2", "3"} - {line}:
        assert [row["annotation_id"] for row in visible[f"wsl:poem.1.{number}"]] == [
            row["annotation_id"] for row in original if row["anchor"]["passage_id"] == f"wsl:poem.1.{number}"
        ]
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("format", "other", "unsupported_format"),
        ("version", 2, "unsupported_version"),
        ("version", 0, "invalid_sidecar"),
    ],
)
def test_read_commands_reject_sidecars_read_only(tmp_path: Path, field: str, value: str | int, code: str) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes((FIXTURES / "corpus/wsl-three.corpus.jsonl").read_bytes())
    sidecar = tmp_path / "annotations.jsonl"
    sidecar.write_text(
        canonical_json({"record": "header", "format": "latinitas-annotations", "version": 1, field: value}) + "\n"
    )
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()}
    for args in (
        ["annotations", "inspect", str(sidecar), "--corpus", str(corpus)],
        ["corpus", "validate", str(corpus), "--annotations", str(sidecar)],
        ["corpus", "show", str(corpus), "--annotations", str(sidecar)],
    ):
        result = CliRunner().invoke(app, [*args, "--json"])
        assert result.exit_code == 3, result.output
        assert json.loads(result.stdout)["diagnostics"][0]["code"] == code
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()} == before


def test_inspect_and_show_retractions_and_safe_text(tmp_path: Path) -> None:
    sidecar = tmp_path / "annotations.jsonl"
    rows = [json.loads(line) for line in (FIXTURES / "annotations/wsl.annotations.jsonl").read_text().splitlines()]
    target = rows[1]
    retract = {key: value for key, value in target.items() if key != "value"}
    retract.update(record="retraction", retracts=target["annotation_id"])
    retract["annotation_id"] = annotation_id(retract)
    sidecar.write_text("".join(canonical_json(row) + "\n" for row in rows))
    append_records(sidecar, lambda _: [Retraction.model_validate(retract)], [])
    # A control in provenance is legal JSON but must never reach the terminal literally.
    rows[5]["created"] = STAMP
    rows[5]["plan"] = "sha256:" + "a" * 64
    rows[5]["producer"]["version"] = "fixture\x1b[31m"
    rows[5]["annotation_id"] = annotation_id(rows[5])
    sidecar.write_text("".join(canonical_json(row) + "\n" for row in [*rows, retract]))
    corpus = FIXTURES / "corpus/wsl-three.corpus.jsonl"
    inspected = CliRunner().invoke(
        app, ["annotations", "inspect", str(sidecar), "--corpus", str(corpus), "--check", "--json"]
    )
    report = json.loads(inspected.stdout)
    assert inspected.exit_code == 1
    assert report["result"]["annotations"][0]["retracted"] is True
    assert report["result"]["annotations"][3]["state"] == "orphaned"
    assert report["result"]["annotations"][4]["plan"] == "sha256:" + "a" * 64
    shown = CliRunner().invoke(app, ["corpus", "show", str(corpus), "--annotations", str(sidecar), "--json"])
    assert shown.exit_code == 0
    assert json.loads(shown.stdout)["result"]["passages"][0]["annotations"] == []
    text = CliRunner().invoke(app, ["annotations", "inspect", str(sidecar), "--corpus", str(corpus)])
    assert text.exit_code == 0 and "\x1b" not in text.output and "\\x1b" in text.output
