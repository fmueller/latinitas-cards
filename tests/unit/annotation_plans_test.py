"""Planning and explicit approval through the real contract CLI."""

import fcntl
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from latinitas_cards.annotation_sidecar import Annotation, read_sidecar
from latinitas_cards.atomic_io import writer_lock
from latinitas_cards.canonical import canonical_json, digest, digest_bytes
from latinitas_cards.cli import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CORPUS = FIXTURES / "corpus/valid/ordinal.jsonl"
SIDECAR = FIXTURES / "annotations/ordinal.annotations.jsonl"


def invoke(*args: str) -> dict[str, Any]:
    result = CliRunner().invoke(app, ["annotations", *args, "--json"])
    assert result.stdout.count("\n") == 1, result.output
    envelope: dict[str, Any] = json.loads(result.stdout)
    assert result.exit_code == envelope["exit_code"], result.output
    return envelope


def proposal(**extra: Any) -> dict[str, Any]:
    row = next(row for row in read_sidecar(SIDECAR) if isinstance(row, Annotation))
    return {
        "operation": "add",
        "layer": "morphology",
        "anchor": row.anchor.model_dump(mode="json"),
        "value": {"tagset": "test", "lemma": "arma"},
        "producer": {"kind": "rule", "name": "test", "version": "1"},
        **extra,
    }


def plan(
    tmp: Path, operations: list[dict[str, Any]], *extra: str, corpus: Path = CORPUS, sidecar: Path = SIDECAR
) -> tuple[dict[str, Any], Path]:
    patch = tmp / "patch.json"
    patch.write_text(json.dumps({"format": "latinitas-annotation-patch", "version": 1, "operations": operations}))
    output = tmp / "plan.json"
    return invoke(
        "plan", str(patch), "--corpus", str(corpus), "--sidecar", str(sidecar), "--output", str(output), *extra
    ), output


def test_plan_identity_and_explicit_approval(tmp_path: Path) -> None:
    result, output = plan(tmp_path, [proposal()])
    assert result["exit_code"] == 0
    document = json.loads(output.read_bytes())
    assert document["plan_id"] == digest({k: v for k, v in document.items() if k != "plan_id"})
    assert document["base"] == {
        "corpus_header": digest(json.loads(CORPUS.read_text().splitlines()[0])),
        "sidecar": digest_bytes(SIDECAR.read_bytes()),
        "policy": None,
        "patch": digest(document["patch"]),
    }
    operation = document["operations"][0]
    assert operation["route"] == "auto"
    assert operation["operation_id"] == digest(proposal())
    assert operation["safety"] == {"decision_status": None, "dependents": [], "conflicts": []}
    assert "created" not in operation["records"][0] and "plan" not in operation["records"][0]
    receipt = tmp_path / "receipt.json"
    args = (
        "approve",
        str(output),
        "--operation",
        operation["operation_id"],
        "--actor",
        "human:tester",
        "--review",
        "Inspected exact proposal",
        "--output",
        str(receipt),
    )
    assert invoke(*args)["exit_code"] == 0
    inode = receipt.stat().st_ino
    assert invoke(*args)["exit_code"] == 0 and receipt.stat().st_ino == inode
    assert json.loads(receipt.read_bytes())["operations"] == [operation["operation_id"]]
    document["operations"][0]["route"] = "review"
    output.write_text(canonical_json(document))
    receipt.unlink()
    assert invoke(*args)["diagnostics"][0]["code"] == "invalid_plan"
    assert not receipt.exists()


def test_entire_patch_conflicts_dependents_and_present_suffix(tmp_path: Path) -> None:
    target = next(row for row in read_sidecar(SIDECAR) if isinstance(row, Annotation))
    dependent = proposal(anchor={**target.anchor.model_dump(mode="json"), "segmentation_id": target.annotation_id})
    operations = [
        proposal(operation="retract", target=target.annotation_id),
        dependent,
        proposal(**{"value": {"tagset": "test", "lemma": "vir"}, "anchor": dependent["anchor"]}),
    ]
    operations[0] = {k: v for k, v in operations[0].items() if k in {"operation", "target", "producer"}}
    result, output = plan(tmp_path, operations)
    assert result["exit_code"] == 0
    rows = json.loads(output.read_bytes())["operations"]
    dependent_ids = sorted(row["records"][0]["annotation_id"] for row in rows[1:])
    assert rows[0]["safety"]["dependents"] == dependent_ids
    assert "destructive" in rows[0]["reasons"]
    assert rows[1]["safety"]["conflicts"] == [
        dependent_ids[0] if rows[1]["records"][0]["annotation_id"] != dependent_ids[0] else dependent_ids[1]
    ]
    assert rows[1]["route"] == rows[2]["route"] == "review"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"value": {}}, "invalid_value"),
        ({"anchor": {"passage_id": "test-corpus:absent", "passage_hash": "sha256:" + "a" * 64}}, "unmatched"),
    ],
)
def test_invalid_operation_still_writes_plan(tmp_path: Path, change: dict[str, Any], reason: str) -> None:
    result, output = plan(tmp_path, [proposal(**change)])
    assert result["exit_code"] == 1
    row = json.loads(output.read_bytes())["operations"][0]
    assert row["route"] == "invalid" and reason in row["reasons"]
    receipt = tmp_path / "receipt.json"
    refused = invoke(
        "approve",
        str(output),
        "--operation",
        row["operation_id"],
        "--actor",
        "agent:test",
        "--review",
        "reviewed",
        "--output",
        str(receipt),
    )
    assert refused["exit_code"] == 3 and not receipt.exists()


def test_output_lock_and_alias(tmp_path: Path) -> None:
    with writer_lock(tmp_path / "plan.json"):
        result, output = plan(tmp_path, [proposal()])
    assert result["exit_code"] == 4 and not output.exists()
    result, _ = plan(tmp_path, [proposal()], "--output", str(tmp_path / "patch.json"))
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "alias"


@pytest.mark.parametrize("status", ["accepted", "withheld"])
@pytest.mark.parametrize("stale", [False, True])
def test_decision_binding_even_when_stale(tmp_path: Path, status: str, stale: bool) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(CORPUS.read_bytes())
    target = next(row for row in read_sidecar(SIDECAR) if isinstance(row, Annotation))
    entry = {
        "base_fingerprint": "a" * 64,
        "decision": {"claim_fingerprint": "a" * 64, "status": status, "reviewer": "R", "reason": "checked"},
        "binding": {
            "kind": "corpus",
            "corpus_id": "test-corpus",
            "passage_id": target.anchor.passage_id,
            "passage_hash": "sha256:" + "b" * 64 if stale else target.input_hash,
            "annotation_id": target.annotation_id,
            "layer": target.layer,
        },
    }
    Path(f"{corpus}.claims.json").write_text(json.dumps({"version": 2, "decisions": [entry]}))
    result, output = plan(
        tmp_path,
        [{"operation": "retract", "target": target.annotation_id, "producer": proposal()["producer"]}],
        corpus=corpus,
    )
    assert result["exit_code"] == 0, result
    row = json.loads(output.read_bytes())["operations"][0]
    assert row["safety"]["decision_status"] == status
    assert row["reasons"] == ["no_policy", "destructive"]


@pytest.mark.parametrize(
    "content,code",
    [("{", "invalid_sidecar"), ('{"version":1,"decisions":[]}', "unsupported_version"), (None, "unreadable")],
)
def test_decisions_input_rejected_without_output(tmp_path: Path, content: str | None, code: str) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(CORPUS.read_bytes())
    claims = Path(f"{corpus}.claims.json")
    if content is None:
        claims.mkdir()
    else:
        claims.write_text(content)
    result, output = plan(tmp_path, [proposal()], corpus=corpus)
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == code
    assert not output.exists()


def test_effective_sidecar_and_whole_patch_dedup(tmp_path: Path) -> None:
    from latinitas_cards.annotation_plans import AnnotationPatch, evaluate_operations
    from latinitas_cards.claim_sidecar import SidecarV2
    from latinitas_cards.corpus_io import read_corpus
    from latinitas_cards.corpus_model import CorpusPassage

    rows = [row for row in read_sidecar(SIDECAR) if isinstance(row, Annotation)]
    target = rows[0]
    operations = [
        {"operation": "retract", "target": target.annotation_id, "producer": proposal()["producer"]},
        proposal(anchor={**target.anchor.model_dump(mode="json"), "segmentation_id": target.annotation_id}),
        proposal(value={"tagset": "test", "lemma": "other"}),
    ]
    operations.append({**operations[1], "created": "2028-01-01T00:00:00Z"})
    patch = AnnotationPatch.model_validate(
        {"format": "latinitas-annotation-patch", "version": 1, "operations": operations}
    )
    passages = {row.passage_id: row for row in read_corpus(CORPUS) if isinstance(row, CorpusPassage)}
    initial = evaluate_operations(patch, passages, rows, SidecarV2(), None)
    suffix = Annotation.model_validate({**initial[1].records[0], "created": "2027-01-01T00:00:00Z", "plan": None})
    later = evaluate_operations(patch, passages, [*rows, suffix], SidecarV2(), None)
    assert initial[0].safety.dependents == [suffix.annotation_id]
    assert [(row.safety, row.route) for row in later] == [(row.safety, row.route) for row in initial]
    assert later[1].route == "auto"  # own already-present add still routes normally


@pytest.mark.parametrize("symlink", [False, True])
def test_protected_aliases_and_immutable_output(tmp_path: Path, symlink: bool) -> None:
    result, output = plan(tmp_path, [proposal()])
    assert result["exit_code"] == 0
    before = output.read_bytes()
    result, _ = plan(tmp_path, [proposal(value={"tagset": "new"})])
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "exists"
    assert output.read_bytes() == before
    output.unlink()
    patch = tmp_path / "patch.json"
    if symlink:
        output.symlink_to(patch)
    else:
        output.hardlink_to(patch)
    result, _ = plan(tmp_path, [proposal()])
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "alias"


@pytest.mark.parametrize(
    "action,actor,unknown,expected",
    [
        ("review", "human:reviewer", False, 0),
        ("deny", "human:reviewer", False, 5),
        ("review", "agent:reviewer", False, 5),
        ("deny", "agent:reviewer", True, 3),
    ],
)
def test_approval_gate_and_precedence(tmp_path: Path, action: str, actor: str, unknown: bool, expected: int) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "format": "latinitas-policy",
                "version": 1,
                "receipt_actor": "human",
                "rules": [{"operation": "add", "layer": "*", "producer_kind": "*", "producer": "*", "action": action}],
            }
        )
    )
    result, output = plan(tmp_path, [proposal()], "--policy", str(policy))
    assert result["exit_code"] == 0
    row = json.loads(output.read_bytes())["operations"][0]
    receipt = tmp_path / "receipt.json"
    args = [
        "approve",
        str(output),
        "--operation",
        row["operation_id"],
        "--actor",
        actor,
        "--review",
        "checked",
        "--output",
        str(receipt),
    ]
    if unknown:
        args.extend(["--operation", "sha256:" + "f" * 64])
    response = invoke(*args)
    assert response["exit_code"] == expected
    assert receipt.exists() == (expected == 0)


@pytest.mark.parametrize("protected_name", ["corpus.manifest.json", "arbitrary.manifest.json", "anything.lock"])
@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink"])
def test_reserved_artifact_aliases(tmp_path: Path, protected_name: str, alias_kind: str) -> None:
    protected = tmp_path / protected_name
    protected.write_text("preserve")
    output = protected if alias_kind == "direct" else tmp_path / "alias.json"
    if alias_kind == "symlink":
        output.symlink_to(protected)
    elif alias_kind == "hardlink":
        output.hardlink_to(protected)
    result, _ = plan(tmp_path, [proposal()], "--output", str(output))
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "alias"
    assert protected.read_text() == "preserve"


@pytest.mark.parametrize("change", [{"version": 1.0}, {"version": 2}, {"format": "other"}, {"unknown": 1}])
def test_patch_envelope_rejection(tmp_path: Path, change: dict[str, Any]) -> None:
    patch = tmp_path / "patch.json"
    patch.write_text(
        json.dumps({"format": "latinitas-annotation-patch", "version": 1, "operations": [proposal()], **change})
    )
    output = tmp_path / "plan.json"
    result = invoke("plan", str(patch), "--corpus", str(CORPUS), "--sidecar", str(SIDECAR), "--output", str(output))
    assert result["exit_code"] == 3 and not output.exists()


@pytest.mark.parametrize("target", ["output", "lock"])
@pytest.mark.parametrize("shape", ["hardlink", "symlink", "chain", "dangling"])
def test_plan_nested_manifest_lock_reservation(tmp_path: Path, target: str, shape: str) -> None:
    def run(*args: str | Path) -> dict[str, Any]:
        result = subprocess.run(
            [sys.executable, "-m", "latinitas_cards", *map(str, args), "--json"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        envelope: dict[str, Any] = json.loads(result.stdout)
        assert result.returncode == envelope["exit_code"], result.stderr
        return envelope

    source, corpus, sidecar = (tmp_path / name for name in ("source.txt", "corpus.jsonl", "side.jsonl"))
    source.write_text("Mater filium amat.\nPuer librum legit.\n")
    normalize = ("corpus", "normalize", source, "--adapter", "text", "--corpus-id", "probe", "--output", corpus)
    assert run(*normalize)["exit_code"] == 0
    storage = tmp_path / "storage"
    storage.mkdir()
    manifest, data = corpus.with_suffix(".manifest.json"), storage / "data.json"
    manifest.rename(data)
    manifest.symlink_to(data)
    assert run(*normalize)["exit_code"] == 0
    reserved = Path(f"{data}.lock")
    passage = next(row for row in map(json.loads, corpus.read_text().splitlines()) if row["record"] == "passage")
    patch = tmp_path / "patch.json"
    patch.write_text(
        json.dumps(
            {
                "format": "latinitas-annotation-patch",
                "version": 1,
                "operations": [
                    {
                        "operation": "add",
                        "layer": "translation",
                        "anchor": {key: passage[key] for key in ("passage_id", "passage_hash")},
                        "value": {"language": "en", "text": "mother"},
                        "producer": {"kind": "agent", "name": "fixture", "version": "1"},
                        "created": "2026-01-02T03:04:05Z",
                    }
                ],
            }
        )
    )
    output = tmp_path / "selected-plan.json"
    alias = output if target == "output" else Path(f"{output}.lock")
    if shape == "hardlink":
        alias.hardlink_to(reserved)
    elif shape == "chain":
        intermediate = tmp_path / "intermediate"
        intermediate.symlink_to(reserved)
        alias.symlink_to(intermediate)
    else:
        if shape == "dangling":
            reserved.unlink()
        alias.symlink_to(reserved)
    args = ("annotations", "plan", patch, "--corpus", corpus, "--sidecar", sidecar, "--output", output)

    def reject() -> None:
        result = run(*args)
        assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "alias", result
        assert not output.exists() if target == "lock" else output.samefile(reserved)

    if shape == "dangling":
        result = run(*args)
        assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "alias", result
        assert not reserved.exists() and not output.exists()
    with reserved.open("ab") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = reserved.stat().st_ino, reserved.read_bytes(), data.read_bytes(), corpus.read_bytes()
        assert run(*normalize)["exit_code"] == 4
        reject()
        assert run(*normalize)["exit_code"] == 4
    reject()
    assert (reserved.stat().st_ino, reserved.read_bytes(), data.read_bytes(), corpus.read_bytes()) == before
    assert run(*normalize)["exit_code"] == 0
    independent = tmp_path / "independent.json"
    assert run(*args, "--output", independent)["exit_code"] == 0
    plan_bytes = independent.read_bytes()
    assert run(*args, "--output", independent)["exit_code"] == 0
    assert independent.read_bytes() == plan_bytes
    apply = ("annotations", "apply", independent, "--corpus", corpus, "--sidecar", sidecar)
    assert run(*apply)["exit_code"] == 0
    sidecar_bytes = sidecar.read_bytes()
    assert run(*apply)["exit_code"] == 0
    assert sidecar.read_bytes() == sidecar_bytes


def test_target_states_duplicate_and_supersede(tmp_path: Path) -> None:
    rows = [row for row in read_sidecar(SIDECAR) if isinstance(row, Annotation)]
    supersede = proposal(operation="supersede", target=rows[0].annotation_id, layer="segmentation", value=rows[0].value)
    response, output = plan(
        tmp_path,
        [
            supersede,
            supersede,
            {"operation": "retract", "target": "sha256:" + "f" * 64, "producer": proposal()["producer"]},
        ],
    )
    assert response["exit_code"] == 1
    planned = json.loads(output.read_bytes())["operations"]
    assert planned[0]["route"] == "review" and planned[0]["records"][0]["supersedes"] == rows[0].annotation_id
    assert planned[1]["reasons"] == ["duplicate_operation"]
    assert planned[2]["reasons"] == ["unknown_target"]
    output.unlink()
    sidecar = tmp_path / "inactive.jsonl"
    sidecar.write_bytes(
        SIDECAR.read_bytes()
        + (
            canonical_json({**planned[0]["records"][0], "created": "2026-01-01T00:00:00Z", "plan": None}) + "\n"
        ).encode()
    )
    response, output = plan(tmp_path, [supersede], sidecar=sidecar)
    assert response["exit_code"] == 1
    assert "target_inactive" in json.loads(output.read_bytes())["operations"][0]["reasons"]


def test_approval_busy_and_required_selection(tmp_path: Path) -> None:
    _, output = plan(tmp_path, [proposal()])
    operation_id = json.loads(output.read_bytes())["operations"][0]["operation_id"]
    receipt = tmp_path / "receipt.json"
    args = ["approve", str(output), "--actor", "human:tester", "--review", "checked", "--output", str(receipt)]
    result = CliRunner().invoke(app, ["annotations", *args, "--json"])
    assert result.exit_code == 2 and not receipt.exists()
    with writer_lock(receipt):
        response = invoke(*args, "--operation", operation_id)
    assert response["exit_code"] == 4 and not receipt.exists()


@pytest.mark.parametrize("actor,review", [("human:ok", ""), ("human:ok", " \n"), ("robot:bad", "checked")])
def test_approval_declared_input_validation(tmp_path: Path, actor: str, review: str) -> None:
    _, output = plan(tmp_path, [proposal()])
    operation_id = json.loads(output.read_bytes())["operations"][0]["operation_id"]
    receipt = tmp_path / "receipt.json"
    result = invoke(
        "approve",
        str(output),
        "--operation",
        operation_id,
        "--actor",
        actor,
        "--review",
        review,
        "--output",
        str(receipt),
    )
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "invalid_input"
    assert not receipt.exists()


def test_identical_invalid_plan_is_noop(tmp_path: Path) -> None:
    response, output = plan(tmp_path, [proposal(value={})])
    assert response["exit_code"] == 1
    before, inode = output.read_bytes(), output.stat().st_ino
    response, _ = plan(tmp_path, [proposal(value={})])
    assert response["exit_code"] == 0
    assert output.read_bytes() == before and output.stat().st_ino == inode


def test_required_mixed_fixtures() -> None:
    from tempfile import TemporaryDirectory

    from latinitas_cards.annotation_plans import AnnotationPatch, create_plan

    fixtures = FIXTURES / "annotations"
    with TemporaryDirectory() as temporary:
        for name, expected in [
            ("mixed", ["auto", "review", "deny", "invalid"]),
            ("mixed-valid", ["auto", "review", "deny"]),
        ]:
            patch = fixtures / f"{name}.patch.json"
            AnnotationPatch.model_validate_json(patch.read_bytes())
            result, diagnostics = create_plan(
                patch,
                CORPUS,
                fixtures / "mixed.base.annotations.jsonl",
                FIXTURES / "policy/mixed.policy.json",
                Path(temporary) / f"{name}.json",
            )
            assert isinstance(result, dict)
            assert isinstance(result["operations"], list)
            assert [row["route"] for row in result["operations"] if isinstance(row, dict)] == expected
            assert bool(diagnostics) == (name == "mixed")
    assert AnnotationPatch.model_validate_json((fixtures / "fixture-morph.patch.json").read_bytes()).operations


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"anchor": {**proposal()["anchor"], "passage_hash": "sha256:" + "a" * 64}}, "stale"),
        ({"anchor": {**proposal()["anchor"], "span": [0, 2000]}}, "span_out_of_range"),
        ({"anchor": {**proposal()["anchor"], "segmentation_id": "sha256:" + "f" * 64}}, "unknown_segmentation"),
    ],
)
def test_anchor_classification(tmp_path: Path, change: dict[str, Any], reason: str) -> None:
    result, output = plan(tmp_path, [proposal(**change)])
    assert result["exit_code"] == 1
    assert reason in json.loads(output.read_bytes())["operations"][0]["reasons"]


@pytest.mark.parametrize(
    "change",
    [
        {"confidence": 0.5},
        {"value": {"tagset": "test", "number": 1.0}},
        {"mystery": True},
        {"producer": {"kind": "untrusted", "name": "x", "version": "1"}},
    ],
)
def test_operation_structure_rejected_without_any_artifact(tmp_path: Path, change: dict[str, Any]) -> None:
    before = SIDECAR.read_bytes()
    result, output = plan(tmp_path, [proposal(**change)])
    assert result["exit_code"] == 3 and not output.exists()
    assert SIDECAR.read_bytes() == before
    assert not Path(f"{output}.lock").exists()


def test_paths_not_identity_and_absent_sidecar(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    copied_corpus = second / "renamed.jsonl"
    copied_corpus.write_bytes(CORPUS.read_bytes())
    result, one = plan(first, [proposal()], sidecar=first / "absent.jsonl")
    other, two = plan(second, [proposal()], corpus=copied_corpus, sidecar=second / "another-absent.jsonl")
    assert result["exit_code"] == other["exit_code"] == 0
    assert one.read_bytes() == two.read_bytes()
    assert json.loads(one.read_bytes())["base"]["sidecar"] is None


def test_allow_agent_and_receipt_immutable(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text('{"format":"latinitas-policy","version":1,"rules":[],"receipt_actor":"any"}')
    _, output = plan(tmp_path, [proposal()], "--policy", str(policy))
    document = json.loads(output.read_bytes())
    operation_id = document["operations"][0]["operation_id"]
    receipt = tmp_path / "receipt.json"
    args = [
        "approve",
        str(output),
        "--operation",
        operation_id,
        "--actor",
        "agent:test",
        "--review",
        "checked",
        "--output",
        str(receipt),
    ]
    assert invoke(*args)["exit_code"] == 0
    before = receipt.read_bytes()
    args[args.index("checked")] = "different review"
    assert invoke(*args)["diagnostics"][0]["code"] == "exists"
    assert receipt.read_bytes() == before
    args[-1] = str(output)
    assert invoke(*args)["diagnostics"][0]["code"] == "alias"


@pytest.mark.parametrize("protected", ["patch", "corpus", "sidecar", "policy", "claims", "manifest"])
def test_receipt_cannot_alias_versioned_inputs(tmp_path: Path, protected: str) -> None:
    _, output = plan(tmp_path, [proposal()])
    operation_id = json.loads(output.read_bytes())["operations"][0]["operation_id"]
    inputs = {
        "patch": (tmp_path / "patch.json").read_bytes(),
        "corpus": CORPUS.read_bytes(),
        "sidecar": SIDECAR.read_bytes(),
        "policy": b'{"format":"latinitas-policy","version":1}',
        "claims": b'{"version":2,"decisions":[]}',
        "manifest": b'{"format":"latinitas-corpus-manifest","version":1}',
    }
    target = tmp_path / "arbitrary-name"
    target.write_bytes(inputs[protected])
    response = invoke(
        "approve",
        str(output),
        "--operation",
        operation_id,
        "--actor",
        "human:tester",
        "--review",
        "checked",
        "--output",
        str(target),
    )
    assert response["exit_code"] == 3 and response["diagnostics"][0]["code"] == "alias"
    assert target.read_bytes() == inputs[protected]
