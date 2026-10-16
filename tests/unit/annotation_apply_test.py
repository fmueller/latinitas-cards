"""Atomic planned annotation application through the CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from latinitas_cards.atomic_io import writer_lock
from latinitas_cards.canonical import canonical_json, digest, digest_bytes
from latinitas_cards.cli import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TIMESTAMP = "2026-01-02T03:04:05Z"


def invoke(*args: str) -> dict[str, Any]:
    result = CliRunner().invoke(app, ["annotations", *args, "--json"])
    assert result.stdout.count("\n") == 1, result.output
    envelope: dict[str, Any] = json.loads(result.stdout)
    assert result.exit_code == envelope["exit_code"], result.output
    return envelope


def setup_plan(tmp: Path) -> tuple[Path, Path, Path, Path]:
    corpus = tmp / "corpus.jsonl"
    corpus.write_bytes((FIXTURES / "corpus/valid/ordinal.jsonl").read_bytes())
    sidecar = tmp / "annotations.jsonl"
    sidecar.write_bytes((FIXTURES / "annotations/mixed.base.annotations.jsonl").read_bytes())
    policy = FIXTURES / "policy/mixed.policy.json"
    plan = tmp / "plan.json"
    result = invoke(
        "plan",
        str(FIXTURES / "annotations/mixed-valid.patch.json"),
        "--corpus",
        str(corpus),
        "--sidecar",
        str(sidecar),
        "--policy",
        str(policy),
        "--output",
        str(plan),
    )
    assert result["exit_code"] == 0, result
    return corpus, sidecar, policy, plan


def test_mixed_partial_receipt_retry_and_replan(tmp_path: Path) -> None:
    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    args = ("--corpus", str(corpus), "--sidecar", str(sidecar), "--policy", str(policy))
    before = sidecar.read_bytes()
    first = invoke("apply", str(plan), *args, "--timestamp", TIMESTAMP)
    assert first["exit_code"] == 0, first
    assert first["result"]["appended"] == 1
    assert len(first["result"]["pending"]) == 1
    assert [d["code"] for d in first["diagnostics"]] == ["denied"]
    auto = sidecar.read_bytes()
    assert auto.startswith(before) and auto != before
    receipt = tmp_path / "receipt.json"
    approved = invoke(
        "approve",
        str(plan),
        "--operation",
        first["result"]["pending"][0],
        "--actor",
        "human:reviewer",
        "--review",
        "Checked",
        "--output",
        str(receipt),
    )
    assert approved["exit_code"] == 0
    second = invoke("apply", str(plan), *args, "--receipt", str(receipt), "--timestamp", TIMESTAMP)
    assert second["exit_code"] == 0, second
    assert second["result"]["appended"] == 1 and second["result"]["pending"] == []
    reviewed = sidecar.read_bytes()
    retry = invoke("apply", str(plan), *args, "--receipt", str(receipt))
    assert retry["exit_code"] == 0 and retry["result"]["already_applied"] is True, retry
    assert sidecar.read_bytes() == reviewed
    replanned = tmp_path / "replanned.json"
    assert (
        invoke(
            "plan",
            str(FIXTURES / "annotations/mixed-valid.patch.json"),
            *args,
            "--output",
            str(replanned),
        )["exit_code"]
        == 0
    )
    assert invoke("apply", str(replanned), *args)["result"]["already_applied"] is True
    assert sidecar.read_bytes() == reviewed
    assert len({digest_bytes(data) for data in (before, auto, reviewed)}) == 3


@pytest.mark.parametrize("target", ["corpus", "sidecar", "claims"])
def test_busy_each_lock_releases_earlier_locks(tmp_path: Path, target: str) -> None:
    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    claims = Path(f"{corpus}.claims.json")
    targets = {"corpus": corpus, "sidecar": sidecar, "claims": claims}
    before = sidecar.read_bytes()
    with writer_lock(targets[target]):
        result = invoke("apply", str(plan), "--corpus", str(corpus), "--sidecar", str(sidecar), "--policy", str(policy))
        assert result["exit_code"] == 4 and result["diagnostics"][0]["code"] == "busy", result
        for key, path in targets.items():
            if key != target:
                with writer_lock(path):
                    pass
    assert sidecar.read_bytes() == before
    assert all(Path(f"{path}.lock").exists() for path in targets.values())


@pytest.mark.parametrize(
    "change",
    ["patch", "invalid", "receipt", "unknown", "header", "passage", "missing", "policy", "foreign", "actor", "deny"],
)
def test_no_write_precedence(tmp_path: Path, change: str) -> None:
    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    document = json.loads(plan.read_bytes())
    receipt = tmp_path / "receipt.json"
    approval = {
        "format": "latinitas-annotation-receipt",
        "version": 1,
        "plan_id": document["plan_id"],
        "actor": "agent:test",
        "review": "Checked",
        "operations": [],
    }
    expected, code = 4, "binding_drift"
    if change in {"patch", "invalid"}:
        if change == "patch":
            document["base"]["patch"] = "sha256:" + "0" * 64
            code = "invalid_plan"
        else:
            document["operations"][0]["route"] = "invalid"
            code = "invalid_operation"
        document["plan_id"] = digest({k: v for k, v in document.items() if k != "plan_id"})
        approval["plan_id"] = document["plan_id"]
        plan.write_text(canonical_json(document))
        expected = 3
    elif change in {"receipt", "unknown"}:
        approval["plan_id" if change == "receipt" else "operations"] = (
            "sha256:" + "0" * 64 if change == "receipt" else ["sha256:" + "0" * 64]
        )
        expected, code = 3, "invalid_receipt" if change == "receipt" else "unknown_operation"
    elif change in {"header", "passage", "missing"}:
        rows = [json.loads(line) for line in corpus.read_bytes().splitlines()]
        if change == "header":
            rows[0]["source"]["sha256"] = "sha256:" + "0" * 64
        else:
            passage = next(row for row in rows if row["record"] == "passage")
            if change == "missing":
                rows.remove(passage)
                # Keep a structurally valid corpus so this is binding drift, not invalid input.
                for row in rows:
                    if row["record"] == "passage":
                        row["order"] -= 1
            else:
                passage["text"] = "Mutatum."
                passage["norm"] = "mutatum."
                passage["passage_hash"] = digest_bytes(b"Mutatum.")
        corpus.write_text("".join(canonical_json(row) + "\n" for row in rows))
    elif change == "policy":
        policy = tmp_path / "other.policy.json"
        policy.write_text((FIXTURES / "policy/mixed.policy.json").read_text() + "\n")
    elif change == "foreign":
        sidecar.write_bytes(sidecar.read_bytes() + sidecar.read_bytes().splitlines(keepends=True)[-1])
    elif change == "actor":
        expected, code = 5, "actor_refused"
    elif change == "deny":
        approval["actor"] = "human:reviewer"
        approval["operations"] = [row["operation_id"] for row in document["operations"] if row["route"] == "deny"]
        expected, code = 5, "denied"
    receipt.write_text(canonical_json(approval))
    before = sidecar.read_bytes()
    result = invoke(
        "apply",
        str(plan),
        "--corpus",
        str(corpus),
        "--sidecar",
        str(sidecar),
        "--policy",
        str(policy),
        "--receipt",
        str(receipt),
    )
    assert result["exit_code"] == expected, result
    assert result["diagnostics"][0]["code"] == code, result
    assert sidecar.read_bytes() == before


def test_atomic_failure_and_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    (tmp_path / "unrelated.csv").write_text("Do not touch\n")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def fail_replace(source: object, target: object) -> None:
        raise OSError("injected replacement failure")

    with monkeypatch.context() as patch:
        patch.setattr("latinitas_cards.atomic_io.os.replace", fail_replace)
        result = invoke(
            "apply",
            str(plan),
            "--corpus",
            str(corpus),
            "--sidecar",
            str(sidecar),
            "--policy",
            str(policy),
            "--timestamp",
            TIMESTAMP,
        )
        assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "unreadable", result
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert {key: after[key] for key in before} == before
    assert set(after) - set(before) == {
        f"{path.name}.lock" for path in (corpus, sidecar, Path(f"{corpus}.claims.json"))
    }
    result = invoke(
        "apply",
        str(plan),
        "--corpus",
        str(corpus),
        "--sidecar",
        str(sidecar),
        "--policy",
        str(policy),
        "--timestamp",
        TIMESTAMP,
    )
    assert result["exit_code"] == 0, result
    final = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert {key: final[key] for key in before if key != sidecar.name} == {
        key: value for key, value in before.items() if key != sidecar.name
    }
    assert set(final) == set(after)


@pytest.mark.parametrize("kind", ["direct", "symlink", "hardlink"])
@pytest.mark.parametrize("target", ["corpus", "claims", "manifest", "plan", "policy", "corpus_lock", "claims_lock"])
def test_alias_no_write(tmp_path: Path, kind: str, target: str) -> None:
    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    local_policy = tmp_path / "policy.json"
    local_policy.write_bytes(policy.read_bytes())
    paths = {
        "corpus": corpus,
        "claims": Path(f"{corpus}.claims.json"),
        "manifest": corpus.with_suffix(".manifest.json"),
        "plan": plan,
        "policy": local_policy,
        "corpus_lock": Path(f"{corpus}.lock"),
        "claims_lock": Path(f"{corpus}.claims.json.lock"),
    }
    protected = paths[target]
    if not protected.exists():
        protected.write_bytes(b"protected\n")
    before = protected.read_bytes()
    sidecar.unlink()
    if kind == "direct":
        sidecar = protected
    elif kind == "symlink":
        sidecar.symlink_to(protected)
    else:
        os.link(protected, sidecar)
    result = invoke(
        "apply",
        str(plan),
        "--corpus",
        str(corpus),
        "--sidecar",
        str(sidecar),
        "--policy",
        str(local_policy),
        "--timestamp",
        TIMESTAMP,
    )
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "alias", result
    assert protected.read_bytes() == before


def test_subprocess_path_environment_determinism(tmp_path: Path) -> None:
    outputs = []
    for index, (tz, locale, seed) in enumerate((("UTC", "C", "12"), ("Europe/Zagreb", "C.UTF-8", "98"))):
        directory = tmp_path / str(index)
        directory.mkdir()
        corpus, sidecar, policy, _ = setup_plan(directory)
        output = directory / "subprocess.plan.json"
        env = {**os.environ, "TZ": tz, "LC_ALL": locale, "PYTHONHASHSEED": seed}

        def path(p: Path, index: int = index, directory: Path = directory) -> str:
            return str(p) if index else os.path.relpath(p, directory)

        for command in (
            ["plan", path(FIXTURES / "annotations/mixed-valid.patch.json"), "--output", path(output)],
            ["apply", path(output), "--timestamp", TIMESTAMP],
        ):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "latinitas_cards",
                    "annotations",
                    *command,
                    "--corpus",
                    path(corpus),
                    "--sidecar",
                    path(sidecar),
                    "--policy",
                    path(policy),
                    "--json",
                ],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stdout + result.stderr
        outputs.append((output.read_bytes(), sidecar.read_bytes()))
    assert outputs[0] == outputs[1]
    assert str(tmp_path).encode() not in outputs[0][0]


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_sidecar_lock_alias(tmp_path: Path, kind: str) -> None:
    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    lock = Path(f"{sidecar}.lock")
    if kind == "symlink":
        lock.symlink_to(sidecar)
    else:
        os.link(sidecar, lock)
    before = sidecar.read_bytes()
    result = invoke("apply", str(plan), "--corpus", str(corpus), "--sidecar", str(sidecar), "--policy", str(policy))
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "alias", result
    assert sidecar.read_bytes() == before


def test_record_integrity_precedes_binding_drift(tmp_path: Path) -> None:
    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    document = json.loads(plan.read_bytes())
    document["operations"][0]["records"][0]["input_hash"] = "sha256:" + "0" * 64
    document["plan_id"] = digest({k: v for k, v in document.items() if k != "plan_id"})
    plan.write_text(canonical_json(document))
    before = sidecar.read_bytes()
    result = invoke("apply", str(plan), "--corpus", str(corpus), "--sidecar", str(sidecar))
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "invalid_plan", result
    assert sidecar.read_bytes() == before


def test_competing_normalizer_and_claim_writer_before_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("Mutatum.\n")
    listed = CliRunner().invoke(app, ["claims", "list", str(corpus), "--annotations", str(sidecar), "--json"])
    assert listed.exit_code == 0, listed.output
    fingerprint = json.loads(listed.stdout)["claims"][0]["fingerprint"]
    command = [sys.executable, "-m", "latinitas_cards"]
    previous_corpus = corpus.read_bytes()
    previous_sidecar = sidecar.read_bytes()
    claims = Path(f"{corpus}.claims.json")
    real_replace = os.replace
    writers: list[subprocess.Popen[str]] = []

    def competing_replace(source_path: Any, target: Any) -> None:
        normalizer = subprocess.run(
            [
                *command,
                "corpus",
                "normalize",
                str(source),
                "--adapter",
                "text",
                "--corpus-id",
                "wsl",
                "--output",
                str(corpus),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        assert normalizer.returncode == 4, normalizer.stdout + normalizer.stderr
        assert json.loads(normalizer.stdout)["diagnostics"][0]["code"] == "busy"
        writer = subprocess.Popen(
            [
                *command,
                "claims",
                "record",
                str(corpus),
                "--annotations",
                str(sidecar),
                "--fingerprint",
                fingerprint,
                "--status",
                "accepted",
                "--reviewer",
                "Reviewer",
                "--reason",
                "Checked",
                "--json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        writers.append(writer)
        with pytest.raises(subprocess.TimeoutExpired):
            writer.communicate(timeout=1)
        assert corpus.read_bytes() == previous_corpus
        assert sidecar.read_bytes() == previous_sidecar
        assert not claims.exists()
        real_replace(source_path, target)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", competing_replace)
            result = invoke(
                "apply",
                str(plan),
                "--corpus",
                str(corpus),
                "--sidecar",
                str(sidecar),
                "--policy",
                str(policy),
                "--timestamp",
                TIMESTAMP,
            )
        assert result["exit_code"] == 0, result
        for writer in writers:
            stdout, stderr = writer.communicate(timeout=20)
            assert writer.returncode == 0, stdout + stderr
            assert json.loads(stdout)["recorded"] == 1
        assert corpus.read_bytes() == previous_corpus
        assert claims.exists()
    finally:
        for writer in writers:
            if writer.poll() is None:
                writer.terminate()
                writer.communicate(timeout=20)


@pytest.mark.parametrize("created", [None, TIMESTAMP])
def test_absent_sidecar_timestamp_dedup_and_changed_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, created: str | None
) -> None:
    corpus, sidecar, _, _ = setup_plan(tmp_path)
    sidecar.unlink()
    # A simple auto-only patch independent of the fixture's supersede target.
    operation = json.loads((FIXTURES / "annotations/mixed-valid.patch.json").read_bytes())["operations"][0]
    if created is not None:
        operation["created"] = created
    patch = tmp_path / "auto.patch.json"
    patch.write_text(canonical_json({"format": "latinitas-annotation-patch", "version": 1, "operations": [operation]}))
    plan = tmp_path / "auto.plan.json"
    args = ("--corpus", str(corpus), "--sidecar", str(sidecar))
    assert invoke("plan", str(patch), *args, "--output", str(plan))["exit_code"] == 0
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    result = invoke("apply", str(plan), *args)
    if created is None:
        assert result["exit_code"] == 2 and result["diagnostics"][0]["code"] == "timestamp_required", result
        assert not sidecar.exists()
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "1767323045")
        result = invoke("apply", str(plan), *args)
    assert result["exit_code"] == 0 and result["result"]["appended"] == 1, result
    before = sidecar.read_bytes()
    rows = [json.loads(line) for line in before.splitlines()]
    assert rows[-1]["created"] == TIMESTAMP
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "invalid")
    assert invoke("apply", str(plan), *args)["result"]["already_applied"] is True
    assert sidecar.read_bytes() == before
    operation["value"]["lemma"] = "mutatum"
    operation["producer"]["version"] = "2"
    patch.write_text(canonical_json({"format": "latinitas-annotation-patch", "version": 1, "operations": [operation]}))
    plan = tmp_path / "changed.plan.json"
    assert invoke("plan", str(patch), *args, "--output", str(plan))["exit_code"] == 0
    assert invoke("apply", str(plan), *args, "--timestamp", TIMESTAMP)["result"]["appended"] == 1
    assert len(sidecar.read_bytes().splitlines()) == 3


@pytest.mark.parametrize("change", ["invalid_corpus", "missing_plan", "bad_receipt", "nonboundary", "segment"])
def test_input_and_foreign_suffix_boundaries(tmp_path: Path, change: str) -> None:
    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    extra: list[str] = []
    expected, code = 3, "invalid_corpus"
    if change == "invalid_corpus":
        rows = [json.loads(line) for line in corpus.read_bytes().splitlines()]
        next(row for row in rows if row["record"] == "passage")["passage_hash"] = "sha256:" + "0" * 64
        corpus.write_text("".join(canonical_json(row) + "\n" for row in rows))
        # Also drift policy; rejected corpus must take precedence.
        extra = []
    elif change == "missing_plan":
        plan.unlink()
        code = "invalid_plan"
    elif change == "bad_receipt":
        receipt = tmp_path / "receipt.json"
        receipt.write_text('{"version":1,"version":1}')
        extra = ["--receipt", str(receipt)]
        code = "invalid_receipt"
    elif change == "nonboundary":
        document = json.loads(plan.read_bytes())
        document["base"]["sidecar"] = digest_bytes(sidecar.read_bytes().rstrip(b"\n"))
        document["plan_id"] = digest({k: v for k, v in document.items() if k != "plan_id"})
        plan.write_text(canonical_json(document))
        expected, code = 4, "binding_drift"
    elif change == "segment":
        passage = json.loads(corpus.read_bytes().splitlines()[-1])
        passage.update(passage_id="test-corpus:p000004", native_ref="p000004", order=4)
        passage["citation"] = {
            "scheme": "ordinal",
            "parts": [{"unit": "passage", "value": "p000004"}],
            "display": "p000004",
        }
        corpus.write_bytes(corpus.read_bytes() + (canonical_json(passage) + "\n").encode())
        old = sidecar.read_bytes()
        assert invoke("segment", str(corpus), "--output", str(sidecar), "--timestamp", TIMESTAMP)["exit_code"] == 0
        assert sidecar.read_bytes() != old
        expected, code = 4, "binding_drift"
    before = sidecar.read_bytes()
    policy_args = [] if change == "invalid_corpus" else ["--policy", str(policy)]
    result = invoke("apply", str(plan), "--corpus", str(corpus), "--sidecar", str(sidecar), *policy_args, *extra)
    assert result["exit_code"] == expected and result["diagnostics"][0]["code"] == code, result
    if change == "invalid_corpus":
        assert "hash_mismatch" in result["diagnostics"][0]["message"]
        assert any(d["code"] == "hash_mismatch" for d in result["diagnostics"][1:])
    assert sidecar.read_bytes() == before


@pytest.mark.parametrize("name", ["reserved.lock", "other.manifest.json"])
def test_reserved_output_names(tmp_path: Path, name: str) -> None:
    corpus, sidecar, policy, _ = setup_plan(tmp_path)
    reserved = tmp_path / name
    sidecar.rename(reserved)
    plan = tmp_path / "reserved.plan.json"
    assert (
        invoke(
            "plan",
            str(FIXTURES / "annotations/mixed-valid.patch.json"),
            "--corpus",
            str(corpus),
            "--sidecar",
            str(reserved),
            "--policy",
            str(policy),
            "--output",
            str(plan),
        )["exit_code"]
        == 0
    )
    before = reserved.read_bytes()
    result = invoke(
        "apply",
        str(plan),
        "--corpus",
        str(corpus),
        "--sidecar",
        str(reserved),
        "--policy",
        str(policy),
        "--timestamp",
        TIMESTAMP,
    )
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "alias", result
    assert reserved.read_bytes() == before


def test_replan_applied_retraction(tmp_path: Path) -> None:
    corpus, sidecar, _, _ = setup_plan(tmp_path)
    target = json.loads(sidecar.read_bytes().splitlines()[1])
    patch = tmp_path / "retract.patch.json"
    patch.write_text(
        canonical_json(
            {
                "format": "latinitas-annotation-patch",
                "version": 1,
                "operations": [
                    {
                        "operation": "retract",
                        "target": target["annotation_id"],
                        "producer": {"kind": "human", "name": "reviewer", "version": "1"},
                    },
                ],
            }
        )
    )
    args = ("--corpus", str(corpus), "--sidecar", str(sidecar))
    plan = tmp_path / "retract.plan.json"
    first = invoke("plan", str(patch), *args, "--output", str(plan))
    assert first["exit_code"] == 0, first
    receipt = tmp_path / "retract.receipt.json"
    assert (
        invoke(
            "approve",
            str(plan),
            "--operation",
            first["result"]["operations"][0]["operation_id"],
            "--actor",
            "human:reviewer",
            "--review",
            "Checked",
            "--output",
            str(receipt),
        )["exit_code"]
        == 0
    )
    assert (
        invoke("apply", str(plan), *args, "--receipt", str(receipt), "--timestamp", TIMESTAMP)["result"]["appended"]
        == 1
    )
    before = sidecar.read_bytes()
    replanned = tmp_path / "retract.replanned.json"
    second = invoke("plan", str(patch), *args, "--output", str(replanned))
    assert second["exit_code"] == 0, second
    assert invoke("apply", str(replanned), *args)["result"]["already_applied"] is True
    assert sidecar.read_bytes() == before


@pytest.mark.parametrize("change", ["value", "producer", "missing_record"])
def test_rehashed_plan_records_must_match_patch(tmp_path: Path, change: str) -> None:
    from latinitas_cards.annotation_sidecar import annotation_id

    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    document = json.loads(plan.read_bytes())
    record = document["operations"][0]["records"][0]
    if change == "value":
        record["value"]["lemma"] = "attacker-chosen"
    elif change == "producer":
        record["producer"]["name"] = "unrelated"
    else:
        document["operations"][0]["records"] = []
    record["annotation_id"] = annotation_id(record)
    document["plan_id"] = digest({k: v for k, v in document.items() if k != "plan_id"})
    plan.write_text(canonical_json(document))
    before = sidecar.read_bytes()
    result = invoke(
        "apply",
        str(plan),
        "--corpus",
        str(corpus),
        "--sidecar",
        str(sidecar),
        "--policy",
        str(policy),
        "--timestamp",
        TIMESTAMP,
    )
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "invalid_plan", result
    assert sidecar.read_bytes() == before


def test_invalid_operation_before_receipt_plan_mismatch(tmp_path: Path) -> None:
    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    document = json.loads(plan.read_bytes())
    receipt = tmp_path / "mismatch.receipt.json"
    receipt.write_text(
        canonical_json(
            {
                "format": "latinitas-annotation-receipt",
                "version": 1,
                "plan_id": document["plan_id"],
                "operations": [],
                "actor": "human:reviewer",
                "review": "Checked",
            }
        )
    )
    document["operations"][0]["route"] = "invalid"
    document["plan_id"] = digest({k: v for k, v in document.items() if k != "plan_id"})
    plan.write_text(canonical_json(document))
    before = sidecar.read_bytes()
    result = invoke(
        "apply",
        str(plan),
        "--corpus",
        str(corpus),
        "--sidecar",
        str(sidecar),
        "--policy",
        str(policy),
        "--receipt",
        str(receipt),
    )
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "invalid_operation", result
    assert sidecar.read_bytes() == before


def test_renormalized_source_header_drift(tmp_path: Path) -> None:
    source, corpus, sidecar = tmp_path / "source.txt", tmp_path / "corpus.jsonl", tmp_path / "annotations.jsonl"
    source.write_text("A.1\tArma virumque.\n")
    normalize = [
        "corpus",
        "normalize",
        str(source),
        "--adapter",
        "text",
        "--corpus-id",
        "test-corpus",
        "--output",
        str(corpus),
        "--citation-template",
        "sl:section.line",
        "--json",
    ]
    assert CliRunner().invoke(app, normalize).exit_code == 0
    assert invoke("segment", str(corpus), "--output", str(sidecar), "--timestamp", TIMESTAMP)["exit_code"] == 0
    passage = next(
        json.loads(line) for line in corpus.read_bytes().splitlines() if json.loads(line)["record"] == "passage"
    )
    patch = tmp_path / "patch.json"
    patch.write_text(
        canonical_json(
            {
                "format": "latinitas-annotation-patch",
                "version": 1,
                "operations": [
                    {
                        "operation": "add",
                        "layer": "morphology",
                        "anchor": {"passage_id": passage["passage_id"], "passage_hash": passage["passage_hash"]},
                        "value": {"tagset": "fixture-morph/1", "lemma": "arma"},
                        "producer": {"kind": "rule", "name": "fixture-morph", "version": "1"},
                    }
                ],
            }
        )
    )
    plan = tmp_path / "plan.json"
    args = ("--corpus", str(corpus), "--sidecar", str(sidecar))
    assert invoke("plan", str(patch), *args, "--output", str(plan))["exit_code"] == 0
    old_source = json.loads(corpus.read_bytes().splitlines()[0])["source"]["sha256"]
    source.write_text("A.1\tMutatum.\n")
    assert CliRunner().invoke(app, normalize).exit_code == 0
    assert json.loads(corpus.read_bytes().splitlines()[0])["source"]["sha256"] != old_source
    before = sidecar.read_bytes()
    result = invoke("apply", str(plan), *args, "--timestamp", TIMESTAMP)
    assert result["exit_code"] == 4 and result["diagnostics"][0]["code"] == "binding_drift", result
    assert sidecar.read_bytes() == before


def test_exact_claims_lock_when_decisions_are_symlinked(tmp_path: Path) -> None:
    import fcntl

    corpus, sidecar, policy, plan = setup_plan(tmp_path)
    decisions = tmp_path / "external-decisions.json"
    decisions.write_text('{"version":2,"decisions":[]}')
    claims = Path(f"{corpus}.claims.json")
    claims.symlink_to(decisions)
    before = sidecar.read_bytes()
    # Match record_corpus_batch's exact lock path, not the resolved data target.
    with Path(f"{claims}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = invoke(
            "apply",
            str(plan),
            "--corpus",
            str(corpus),
            "--sidecar",
            str(sidecar),
            "--policy",
            str(policy),
            "--timestamp",
            TIMESTAMP,
        )
        assert result["exit_code"] == 4 and result["diagnostics"][0]["code"] == "busy", result
    assert sidecar.read_bytes() == before
    assert not Path(f"{decisions}.lock").exists()
