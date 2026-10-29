"""Command inputs, aliases, locking and deterministic text/template output."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner, Result

from latinitas_cards.atomic_io import writer_lock
from latinitas_cards.cli import app
from latinitas_cards.cli_contract import Diagnostic
from latinitas_cards.commands import corpus

FIXTURES = Path(__file__).parents[1] / "fixtures/corpus/text"


def normalize(source: Path, output: Path, *flags: str) -> Result:
    return CliRunner().invoke(
        app,
        [
            "corpus",
            "normalize",
            str(source),
            "--adapter",
            "text",
            "--corpus-id",
            "text-test",
            "--output",
            str(output),
            "--json",
            *flags,
        ],
    )


def test_refrain_and_template_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    manifest = output.with_suffix(".manifest.json")
    source = FIXTURES / "refrain.txt"
    assert normalize(source, output).exit_code == 0
    before = output.read_bytes(), manifest.read_bytes(), output.stat().st_ino, manifest.stat().st_ino
    assert normalize(source, output).exit_code == 0
    assert (output.read_bytes(), manifest.read_bytes(), output.stat().st_ino, manifest.stat().st_ino) == before
    manifest.unlink()
    assert normalize(FIXTURES / "wsl.txt", output, "--citation-template", "wsl:work.section.line").exit_code == 0
    assert not manifest.exists()
    records = [json.loads(line) for line in output.read_bytes().splitlines()]
    assert [r["work"] for r in records if r["record"] == "work"] == ["é", "%"]
    passages = [r for r in records if r["record"] == "passage"]
    assert [r["native_ref"] for r in passages] == ["%C3%A9.%25.%252E", "%25.2.3"]
    assert [part["value"] for part in passages[0]["citation"]["parts"]] == ["é", "%", "%2E"]
    assert CliRunner().invoke(app, ["corpus", "validate", str(output)]).exit_code == 0
    assert records[0]["edition"] is None and records[1]["title"] is None


@pytest.mark.parametrize(
    "flags,code",
    [
        (["--adapter", "unknown"], "usage_error"),
        (["--language", "x-latin"], "usage_error"),
        (["--citation-template", "bad"], "usage_error"),
        (["--citation-template", "wsl:work.line", "--manifest", "unused"], "usage_error"),
        (["--citation-template", "wsl:work.line", "--approve-allocation", "0"], "usage_error"),
        (["--approve-allocation", "0"], "invalid_approval"),
    ],
)
def test_rejected_options(tmp_path: Path, flags: list[str], code: str) -> None:
    source, output = tmp_path / "input", tmp_path / "out"
    source.write_text("a\n")
    result = normalize(source, output, *flags)
    assert result.exit_code == (2 if code == "usage_error" else 3), result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == code
    assert not output.exists()


@pytest.mark.parametrize(
    "adapter,flags,message",
    [
        ("nope", ["--language", "x-latin"], "Adapter 'nope' is not yet supported"),
        ("usfx", ["--language", "x-latin", "--manifest", "m"], "Malformed language tag"),
        ("usfx", ["--citation-template", "bad", "--manifest", "m"], "Malformed citation template"),
        ("usfx", ["--manifest", "m"], "USFX cannot use templates, manifests or approvals"),
        ("text", ["--allow-unknown", "--manifest", "m"], "--allow-unknown requires the USFX adapter"),
    ],
)
def test_option_checks_run_in_order(tmp_path: Path, adapter: str, flags: list[str], message: str) -> None:
    source, output = tmp_path / "input", tmp_path / "out"
    source.write_text("a\n")
    result = CliRunner().invoke(
        app,
        ["corpus", "normalize", str(source), "--adapter", adapter, "--corpus-id", "c", "--output", str(output)]
        + ["--json", *flags],
    )
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["message"] == message
    assert not output.exists()


def test_adapter_registry_is_the_only_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "input", tmp_path / "out"
    source.write_text("a\n")
    assert set(corpus.ADAPTERS) == {"text", "usfx"}
    requests: list[corpus.NormalizeRequest] = []

    def fake(request: corpus.NormalizeRequest) -> tuple[JsonValue, list[Diagnostic]]:
        requests.append(request)
        return {"review_items": []}, []

    monkeypatch.setitem(corpus.ADAPTERS, "fake", fake)
    monkeypatch.delitem(corpus.ADAPTERS, "usfx")
    argv = ["corpus", "normalize", str(source), "--corpus-id", "c", "--output", str(output), "--json"]
    result = CliRunner().invoke(app, [*argv, "--adapter", "fake"])
    assert result.exit_code == 0, result.output
    assert [(r.source, r.corpus_id, r.output) for r in requests] == [(source, "c", output)]
    result = CliRunner().invoke(app, [*argv, "--adapter", "usfx"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["message"] == "Adapter 'usfx' is not yet supported"


@pytest.mark.parametrize(
    "content,code",
    [
        (b"\xff", "unreadable"),
        (b"a", "invalid_line"),
        (b"a.b\ttext", "invalid_line"),
        (b"a..b\ttext", "invalid_line"),
        (b"a.b.c\tone\na.b.c\ttwo", "duplicate_source_ref"),
    ],
)
def test_rejected_lines(tmp_path: Path, content: bytes, code: str) -> None:
    source, output = tmp_path / "input", tmp_path / "out"
    source.write_bytes(content)
    result = normalize(source, output, "--citation-template", "wsl:work.section.line")
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == code
    assert not output.exists()


@pytest.mark.parametrize(
    "alias,which,protected",
    [
        (alias, which, protected)
        for alias in ("direct", "symlink", "hardlink")
        for which in ("output", "manifest")
        for protected in ("source", "descriptor", "other-output", "output-lock", "manifest-lock")
        # Renaming a target to its former lock also renames its lock; no self-alias exists.
        if not (alias == "direct" and protected == f"{which}-lock")
    ],
)
def test_output_alias_matrix(tmp_path: Path, which: str, protected: str, alias: str) -> None:
    source, output, manifest, descriptor = [tmp_path / name for name in ("source", "output", "manifest", "descriptor")]
    source.write_text("a\n")
    descriptor.write_text('{"format":"latinitas-corpus-descriptor","version":1}')
    target = output if which == "output" else manifest
    other = manifest if which == "output" else output
    protected_path = {
        "source": source,
        "descriptor": descriptor,
        "other-output": other,
        "output-lock": Path(f"{output}.lock"),
        "manifest-lock": Path(f"{manifest}.lock"),
    }[protected]
    if not protected_path.exists():
        protected_path.write_bytes(b"protected")
    if alias == "direct":
        if which == "output":
            output = protected_path
        else:
            manifest = protected_path
    elif alias == "symlink":
        target.symlink_to(protected_path)
    else:
        target.hardlink_to(protected_path)
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    result = normalize(source, output, "--manifest", str(manifest), "--descriptor", str(descriptor))
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "alias"
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("lock", ["output", "manifest", "shared-manifest"])
def test_busy_writers(tmp_path: Path, lock: str) -> None:
    source, output, manifest = tmp_path / "source", tmp_path / "out", tmp_path / "manifest"
    source.write_text("a\n")
    with writer_lock(output if lock == "output" else manifest):
        result = normalize(source, output, "--manifest", str(manifest))
        assert result.exit_code == 4, result.output
        assert not output.exists() and not manifest.exists()
        if lock == "shared-manifest":
            assert normalize(source, tmp_path / "second", "--manifest", str(manifest)).exit_code == 4


@pytest.mark.parametrize("adapter", ["text", "usfx"])
def test_subprocess_determinism(tmp_path: Path, adapter: str) -> None:
    results = []
    for i, (tz, locale, seed) in enumerate([("UTC", "C", "1"), ("Europe/Zagreb", "C.UTF-8", "777")]):
        directory = tmp_path / str(i)
        directory.mkdir()
        source = directory / "input.txt"
        raw = b"\xef\xbb\xbfa\r\n b\t c \n\na\n"
        if adapter == "usfx":
            raw = (FIXTURES.parent / "usfx/gen-1-1-5.usfx.xml").read_bytes()
        source.write_bytes(raw)
        output = directory / "out.jsonl"
        env = {**os.environ, "TZ": tz, "LC_ALL": locale, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "latinitas_cards",
                "corpus",
                "normalize",
                str(source),
                "--adapter",
                adapter,
                "--corpus-id",
                "test-corpus",
                "--output",
                str(output),
            ],
            cwd=directory,
            env=env,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert source.read_bytes() == raw
        results.append(
            (output.read_bytes(), output.with_suffix(".manifest.json").read_bytes() if adapter == "text" else b"")
        )
    assert results[0] == results[1]
    assert str(tmp_path).encode() not in b"".join(results[0])


def test_both_locks_cover_input_and_old_state_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from latinitas_cards.cli_contract import Code, ContractError

    source, output = tmp_path / "source", tmp_path / "out"
    manifest = output.with_suffix(".manifest.json")
    source.write_text("a\n")
    original = Path.read_bytes
    observed = set()

    def guarded_read(path: Path) -> bytes:
        if path in {source, output, manifest}:
            observed.add(path)
            for target in (output, manifest):
                with pytest.raises(ContractError) as error, writer_lock(target):
                    pytest.fail("read occurred outside both locks")
                assert error.value.diagnostic.code == Code.BUSY
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    assert normalize(source, output).exit_code == 0
    assert observed == {source, output, manifest}


def test_template_without_work_and_descriptor_options(tmp_path: Path) -> None:
    source, output, descriptor = tmp_path / "source", tmp_path / "out", tmp_path / "descriptor"
    source.write_text("1.2\tArma\n1.3\tCano\n")
    descriptor.write_text('{"format":"latinitas-corpus-descriptor","version":1,"title":"Reader","edition":"One"}')
    result = normalize(
        source, output, "--citation-template", "sl:section.line", "--language", "hr-HR", "--descriptor", str(descriptor)
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_bytes().splitlines()]
    assert records[0]["language"] == "hr-HR" and records[0]["edition"] == "One"
    assert records[1]["title"] == "Reader" and records[2]["work"] == "text-test"
    assert [r["work"] for r in records if r["record"] == "passage"] == ["text-test", "text-test"]
