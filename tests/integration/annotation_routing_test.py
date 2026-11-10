"""Policy safety across immutable plans and locked application snapshots."""

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from annotation_apply_test import FIXTURES, TIMESTAMP, invoke, setup_plan
from typer.testing import CliRunner

from latinitas_cards.annotation_plans import AnnotationPatch, checked_record, proposed_record
from latinitas_cards.canonical import canonical_json, digest_bytes
from latinitas_cards.cli import app


def make_plan(tmp: Path, operations: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    corpus, sidecar, _, _ = setup_plan(tmp)
    patch = tmp / "patch.json"
    patch.write_text(json.dumps({"format": "latinitas-annotation-patch", "version": 1, "operations": operations}))
    plan = tmp / "routing-plan.json"
    result = invoke("plan", str(patch), "--corpus", str(corpus), "--sidecar", str(sidecar), "--output", str(plan))
    assert result["exit_code"] == 0, result
    return corpus, sidecar, plan


def operations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads((FIXTURES / "annotations/mixed-valid.patch.json").read_text())["operations"]
    return rows


def retract() -> dict[str, Any]:
    original = operations()[1]
    return {key: original[key] for key in ("producer", "target")} | {"operation": "retract"}


def apply(
    corpus: Path, sidecar: Path, plan: Path, receipt: Path | None = None, policy: Path | None = None
) -> dict[str, Any]:
    extra: tuple[str, ...] = ("--receipt", str(receipt)) if receipt else ()
    if policy:
        extra += ("--policy", str(policy))
    before = digest_bytes(sidecar.read_bytes())
    result = invoke(
        "apply", str(plan), "--corpus", str(corpus), "--sidecar", str(sidecar), "--timestamp", TIMESTAMP, *extra
    )
    print(
        json.dumps(
            {
                "case": corpus.parent.name,
                "routes": [row["route"] for row in json.loads(plan.read_text())["operations"]],
                "before": before,
                "after": digest_bytes(sidecar.read_bytes()),
                "exit_code": result["exit_code"],
                "diagnostics": [row["code"] for row in result["diagnostics"]],
            }
        )
    )
    return result


def approve(plan: Path) -> Path:
    document = json.loads(plan.read_text())
    receipt = plan.with_name("receipt.json")
    result = invoke(
        "approve",
        str(plan),
        "--operation",
        document["operations"][-1]["operation_id"],
        "--actor",
        "human:ana",
        "--review",
        "Checked",
        "--output",
        str(receipt),
    )
    assert result["exit_code"] == 0, result
    return receipt


def decision(corpus: Path, status: str | None, *, unrelated: bool = False, stale: bool = True) -> None:
    target = retract()["target"] if not unrelated else "sha256:" + "a" * 64
    entries = (
        []
        if status is None
        else [
            {
                "base_fingerprint": "b" * 64,
                "decision": {"claim_fingerprint": "b" * 64, "status": status, "reviewer": "Ana", "reason": "Checked"},
                "binding": {
                    "kind": "corpus",
                    "corpus_id": "test-corpus",
                    "passage_id": "test-corpus:p000001",
                    "passage_hash": "sha256:" + "c" * 64 if stale else operations()[0]["anchor"]["passage_hash"],
                    "annotation_id": target,
                    "layer": "segmentation",
                },
            }
        ]
    )
    Path(f"{corpus}.claims.json").write_text(json.dumps({"version": 2, "decisions": entries}))


@pytest.mark.parametrize("status", ["accepted", "withheld"])
@pytest.mark.parametrize("stale", [False, True])
def test_post_plan_stale_decision_rejects_atomic_batch(tmp_path: Path, status: str, stale: bool) -> None:
    corpus, sidecar, plan = make_plan(tmp_path, [operations()[0], retract()])
    receipt = approve(plan)
    assert json.loads(plan.read_text())["operations"][-1]["safety"]["decision_status"] is None
    decision(corpus, status, stale=stale)
    before = digest_bytes(sidecar.read_bytes())
    result = apply(corpus, sidecar, plan, receipt)
    assert result["exit_code"] == 4, result
    assert result["diagnostics"][0]["code"] == "binding_drift"
    assert digest_bytes(sidecar.read_bytes()) == before


@pytest.mark.parametrize("kind", ["dependent", "conflict"])
def test_same_plan_unplanned_suffix_rejected(tmp_path: Path, kind: str) -> None:
    corpus, sidecar, plan = make_plan(tmp_path, [operations()[0], retract()])
    receipt = approve(plan)
    extra = operations()[0]
    if kind == "dependent":
        extra["anchor"]["segmentation_id"] = retract()["target"]
        extra["anchor"]["span"] = [0, 1]
    else:
        extra["value"]["lemma"] = "alius"
    patch = AnnotationPatch.model_validate(
        {"format": "latinitas-annotation-patch", "version": 1, "operations": [extra]}
    )
    raw = proposed_record(patch.operations[0], {})
    assert raw is not None
    record = checked_record(raw)
    record.plan = json.loads(plan.read_text())["plan_id"]
    with sidecar.open("a") as stream:
        stream.write(canonical_json(record.model_dump(mode="json")) + "\n")
    before = digest_bytes(sidecar.read_bytes())
    result = apply(corpus, sidecar, plan, receipt)
    assert result["exit_code"] == 4, result
    assert result["diagnostics"][0]["code"] == "binding_drift"
    assert digest_bytes(sidecar.read_bytes()) == before


def replan(corpus: Path, sidecar: Path, plan: Path, policy: Path | None = None) -> None:
    plan.unlink()
    extra = ("--policy", str(policy)) if policy else ()
    result = invoke(
        "plan",
        str(plan.with_name("patch.json")),
        "--corpus",
        str(corpus),
        "--sidecar",
        str(sidecar),
        "--output",
        str(plan),
        *extra,
    )
    assert result["exit_code"] == 0, result


def suffix(sidecar: Path, plan: Path, operation: dict[str, Any]) -> str:
    from latinitas_cards.annotation_sidecar import Annotation, read_sidecar

    patch = AnnotationPatch.model_validate(
        {"format": "latinitas-annotation-patch", "version": 1, "operations": [operation]}
    )
    targets = {row.annotation_id: row for row in read_sidecar(sidecar) if isinstance(row, Annotation)}
    raw = proposed_record(patch.operations[0], targets)
    assert raw is not None
    record = checked_record(raw)
    record.plan = json.loads(plan.read_text())["plan_id"]
    with sidecar.open("a") as stream:
        stream.write(canonical_json(record.model_dump(mode="json")) + "\n")
    return record.annotation_id


@pytest.mark.parametrize("status", ["withheld", None])
def test_changed_or_removed_decision_is_drift(tmp_path: Path, status: str | None) -> None:
    corpus, sidecar, plan = make_plan(tmp_path, [operations()[0], retract()])
    decision(corpus, "accepted")
    replan(corpus, sidecar, plan)
    receipt = approve(plan)
    assert json.loads(plan.read_text())["operations"][-1]["reasons"] == ["no_policy", "destructive"]
    decision(corpus, status)
    before = digest_bytes(sidecar.read_bytes())
    result = apply(corpus, sidecar, plan, receipt)
    assert result["exit_code"] == 4 and result["diagnostics"][0]["code"] == "binding_drift", result
    assert digest_bytes(sidecar.read_bytes()) == before


def test_unrelated_decisions_and_completed_noop_are_not_drift(tmp_path: Path) -> None:
    corpus, sidecar, plan = make_plan(tmp_path, [operations()[0], retract()])
    receipt = approve(plan)
    decision(corpus, "accepted", unrelated=True)
    claims = Path(f"{corpus}.claims.json")
    before = claims.read_bytes()
    result = apply(corpus, sidecar, plan, receipt)
    assert result["exit_code"] == 0 and result["result"]["appended"] == 2, result
    assert claims.read_bytes() == before
    decision(corpus, "withheld")
    after = digest_bytes(sidecar.read_bytes())
    result = apply(corpus, sidecar, plan, receipt)
    assert result["exit_code"] == 0 and result["result"]["already_applied"], result
    assert digest_bytes(sidecar.read_bytes()) == after


@pytest.mark.parametrize("kind", ["dependent", "conflict"])
def test_expected_partial_suffix_is_deduplicated(tmp_path: Path, kind: str) -> None:
    first = operations()[0]
    second = retract()
    if kind == "dependent":
        first["anchor"]["segmentation_id"] = retract()["target"]
        first["anchor"]["span"] = [0, 2]
    else:
        second = operations()[0]
        second["value"]["lemma"] = "alius"
    corpus, sidecar, plan = make_plan(tmp_path, [first, second])
    rows = json.loads(plan.read_text())["operations"]
    fact = "dependents" if kind == "dependent" else "conflicts"
    assert rows[1]["safety"][fact] == [rows[0]["records"][0]["annotation_id"]]
    if kind == "dependent":
        assert apply(corpus, sidecar, plan)["result"]["appended"] == 1
    else:
        suffix(sidecar, plan, first)
    receipt = approve(plan)
    result = apply(corpus, sidecar, plan, receipt)
    assert result["exit_code"] == 0 and result["result"]["appended"] == 1, result
    assert apply(corpus, sidecar, plan, receipt)["result"]["appended"] == 0


@pytest.mark.parametrize("kind", ["dependent", "conflict"])
def test_removed_fact_in_same_plan_suffix_is_drift(tmp_path: Path, kind: str) -> None:
    corpus, sidecar, plan = make_plan(tmp_path, [operations()[0], retract()])
    extra = operations()[0]
    if kind == "dependent":
        extra["anchor"]["segmentation_id"] = retract()["target"]
    else:
        extra["value"]["lemma"] = "alius"
    extra_id = suffix(sidecar, plan, extra)
    replan(corpus, sidecar, plan)
    suffix(sidecar, plan, {"operation": "retract", "target": extra_id, "producer": retract()["producer"]})
    before = digest_bytes(sidecar.read_bytes())
    result = apply(corpus, sidecar, plan)
    assert result["exit_code"] == 4 and result["diagnostics"][0]["code"] == "binding_drift", result
    assert digest_bytes(sidecar.read_bytes()) == before


@pytest.mark.parametrize("kind", ["segmentation", "target"])
def test_newly_invalid_pending_operation_rejects_valid_add_too(tmp_path: Path, kind: str) -> None:
    pending = retract()
    changed = retract()
    changed["producer"]["name"] = "unplanned"
    if kind == "segmentation":
        pending = operations()[0]
        pending["anchor"]["segmentation_id"] = retract()["target"]
        pending["anchor"]["span"] = [0, 2]
        changed = operations()[1]
    corpus, sidecar, plan = make_plan(tmp_path, [operations()[0], pending])
    receipt = approve(plan)
    suffix(sidecar, plan, changed)
    before = digest_bytes(sidecar.read_bytes())
    result = apply(corpus, sidecar, plan, receipt)
    assert result["exit_code"] == 4 and result["diagnostics"][0]["code"] == "binding_drift", result
    assert digest_bytes(sidecar.read_bytes()) == before


@pytest.mark.parametrize(
    "content,code",
    [("{", "invalid_sidecar"), ('{"version":1,"decisions":[]}', "unsupported_version"), (None, "unreadable")],
)
def test_current_claims_fail_closed(tmp_path: Path, content: str | None, code: str) -> None:
    corpus, sidecar, plan = make_plan(tmp_path, [operations()[0]])
    claims = Path(f"{corpus}.claims.json")
    if content is None:
        claims.mkdir()
    else:
        claims.write_text(content)
    before = digest_bytes(sidecar.read_bytes())
    result = apply(corpus, sidecar, plan)
    assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == code, result
    assert digest_bytes(sidecar.read_bytes()) == before


def write_policy(tmp: Path, action: str, actor: str = "human") -> Path:
    policy = tmp / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "format": "latinitas-policy",
                "version": 1,
                "receipt_actor": actor,
                "rules": [
                    {
                        "operation": "add" if action == "auto" else "*",
                        "layer": "*",
                        "producer_kind": "*",
                        "producer": "*",
                        "action": action,
                    }
                ],
            }
        )
    )
    return policy


@pytest.mark.parametrize("action", [None, "auto", "deny"])
def test_patch_conflicts_override_auto_but_not_deny(tmp_path: Path, action: str | None) -> None:
    first, second = operations()[0], operations()[0]
    second["value"]["lemma"] = "alius"
    corpus, sidecar, plan = make_plan(tmp_path, [first, second])
    policy = write_policy(tmp_path, action) if action else None
    if policy:
        replan(corpus, sidecar, plan, policy)
    rows = json.loads(plan.read_text())["operations"]
    assert [row["route"] for row in rows] == (["deny"] * 2 if action == "deny" else ["review"] * 2)
    assert all("conflicting" in row["reasons"] for row in rows)
    before = digest_bytes(sidecar.read_bytes())
    result = apply(corpus, sidecar, plan, policy=policy)
    assert result["exit_code"] == 0 and result["result"]["appended"] == 0, result
    assert digest_bytes(sidecar.read_bytes()) == before
    if action == "deny":
        result = invoke(
            "approve",
            str(plan),
            "--operation",
            rows[0]["operation_id"],
            "--actor",
            "human:ana",
            "--review",
            "Checked",
            "--output",
            str(tmp_path / "denied-receipt.json"),
        )
        assert result["exit_code"] == 5 and result["diagnostics"][0]["code"] == "denied", result
        assert not (tmp_path / "denied-receipt.json").exists()


def test_current_conflict_and_created_only_duplicates(tmp_path: Path) -> None:
    corpus, sidecar, plan = make_plan(tmp_path, [operations()[0]])
    extra = operations()[0]
    extra["value"]["lemma"] = "alius"
    suffix(sidecar, plan, extra)
    replan(corpus, sidecar, plan)
    assert json.loads(plan.read_text())["operations"][0]["route"] == "review"
    # Different operation identity, identical annotation identity: no self-conflict.
    extra = operations()[0]
    extra["created"] = TIMESTAMP
    patch = plan.with_name("patch.json")
    patch.write_text(
        json.dumps({"format": "latinitas-annotation-patch", "version": 1, "operations": [operations()[0], extra]})
    )
    sidecar.write_bytes((FIXTURES / "annotations/mixed.base.annotations.jsonl").read_bytes())
    replan(corpus, sidecar, plan)
    rows = json.loads(plan.read_text())["operations"]
    assert rows[0]["operation_id"] != rows[1]["operation_id"]
    assert [row["route"] for row in rows] == ["auto", "auto"]
    assert all(row["safety"]["conflicts"] == [] for row in rows)
    result = apply(corpus, sidecar, plan)
    assert result["exit_code"] == 0 and result["result"]["appended"] == 1, result


@pytest.mark.parametrize("actor_policy", ["human", "any"])
def test_agent_receipt_is_declaration_not_authentication(tmp_path: Path, actor_policy: str) -> None:
    corpus, sidecar, plan = make_plan(tmp_path, [retract()])
    policy = write_policy(tmp_path, "review", actor_policy)
    replan(corpus, sidecar, plan, policy)
    row = json.loads(plan.read_text())["operations"][0]
    receipt = tmp_path / "agent-receipt.json"
    result = invoke(
        "approve",
        str(plan),
        "--operation",
        row["operation_id"],
        "--actor",
        "agent:bot",
        "--review",
        "Checked",
        "--output",
        str(receipt),
    )
    assert result["exit_code"] == (5 if actor_policy == "human" else 0), result
    assert receipt.exists() == (actor_policy == "any")
    receipt.write_text(
        json.dumps(
            {
                "format": "latinitas-annotation-receipt",
                "version": 1,
                "plan_id": json.loads(plan.read_text())["plan_id"],
                "operations": [row["operation_id"]],
                "actor": "agent:bot",
                "review": "Checked",
            }
        )
    )
    before = digest_bytes(sidecar.read_bytes())
    result = apply(corpus, sidecar, plan, receipt, policy)
    assert result["exit_code"] == (5 if actor_policy == "human" else 0), result
    if actor_policy == "human":
        assert result["diagnostics"][0]["code"] == "actor_refused"
        assert digest_bytes(sidecar.read_bytes()) == before
    else:
        assert result["result"]["appended"] == 1


def test_invalid_actor_independence_and_atomic_batch(tmp_path: Path) -> None:
    corpus, sidecar, plan = make_plan(tmp_path, [operations()[0]])
    patch = plan.with_name("patch.json")
    valid = [operations()[0] for _ in range(3)]
    for index, row in enumerate(valid):
        row["producer"]["name"] = f"producer-{index}"
    reasons = []
    before = digest_bytes(sidecar.read_bytes())
    for kind in ("human", "agent"):
        invalid = operations()[0]
        invalid["producer"]["kind"] = kind
        invalid["anchor"]["passage_hash"] = "sha256:" + "d" * 64
        invalid["value"].pop("tagset")
        patch.write_text(
            json.dumps({"format": "latinitas-annotation-patch", "version": 1, "operations": [*valid, invalid]})
        )
        plan.unlink()
        result = invoke("plan", str(patch), "--corpus", str(corpus), "--sidecar", str(sidecar), "--output", str(plan))
        assert result["exit_code"] == 1, result
        rows = json.loads(plan.read_text())["operations"]
        assert [row["route"] for row in rows] == ["auto", "auto", "auto", "invalid"]
        reasons.append(rows[-1]["reasons"])
        for actor in ("human:ana", "agent:bot"):
            result = invoke(
                "approve",
                str(plan),
                "--operation",
                rows[-1]["operation_id"],
                "--actor",
                actor,
                "--review",
                "Checked",
                "--output",
                str(tmp_path / "invalid-receipt.json"),
            )
            assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "invalid_operation", result
            assert not (tmp_path / "invalid-receipt.json").exists()
        result = apply(corpus, sidecar, plan)
        assert result["exit_code"] == 3 and result["diagnostics"][0]["code"] == "invalid_operation", result
        assert digest_bytes(sidecar.read_bytes()) == before
    assert reasons[0] == reasons[1]
    patch.write_text(json.dumps({"format": "latinitas-annotation-patch", "version": 1, "operations": valid}))
    replan(corpus, sidecar, plan)
    result = apply(corpus, sidecar, plan)
    assert result["exit_code"] == 0 and result["result"]["appended"] == 3, result


def test_provenance_and_no_linguistic_acceptance(tmp_path: Path) -> None:
    fixture = json.loads((FIXTURES / "annotations/fixture-morph.patch.json").read_text())["operations"][0]
    corpus, sidecar, plan = make_plan(tmp_path, [fixture, retract()])
    receipt = approve(plan)
    claims = Path(f"{corpus}.claims.json")
    assert not claims.exists()
    assert apply(corpus, sidecar, plan, receipt)["exit_code"] == 0
    listed = CliRunner().invoke(app, ["claims", "list", str(corpus), "--annotations", str(sidecar), "--json"])
    assert listed.exit_code == 0, listed.output
    assert all(row["review_status"] == "open" for row in json.loads(listed.stdout)["claims"])
    assert not claims.exists()
    result = invoke("inspect", str(sidecar), "--corpus", str(corpus))
    applied = [row for row in result["result"]["annotations"] if row["plan"] is not None]
    assert len(applied) == 2
    header = json.loads(corpus.read_text().splitlines()[0])
    for row in applied:
        assert row["plan"] == json.loads(plan.read_text())["plan_id"]
        assert row["passage_id"] == "test-corpus:p000001"
        assert row["source_sha256"] == header["source"]["sha256"]
        assert row["input_hash"] == operations()[0]["anchor"]["passage_hash"]
        assert row["producer"] == (fixture["producer"] if row["layer"] == "morphology" else retract()["producer"])
    assert not {"cltk", "stanza", "torch"} & sys.modules.keys()


def imported_names(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}


def test_no_optional_provider_imports_or_claim_recorders() -> None:
    forbidden = {"cltk", "stanza", "torch", "openai", "anthropic", "ollama"}
    assert imported_names("import torch.nn\nfrom anthropic import Client") & forbidden == {"torch", "anthropic"}
    for path in FIXTURES.parent.rglob("*.py"):
        assert not imported_names(path.read_text()) & forbidden, path
    root = FIXTURES.parents[1] / "src/latinitas_cards"
    for name in ("annotation_plans.py", "annotation_apply.py", "policy.py"):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not {alias.name for alias in node.names} & {
                    "record_batch",
                    "record_corpus_batch",
                    "review_claim",
                }


def test_completed_patch_segmentation_cannot_revive_invalid_anchor(tmp_path: Path) -> None:
    segmentation = operations()[1]
    segmentation.pop("target")
    segmentation["operation"] = "add"
    patch = AnnotationPatch.model_validate(
        {"format": "latinitas-annotation-patch", "version": 1, "operations": [segmentation]}
    )
    raw = proposed_record(patch.operations[0], {})
    assert raw is not None
    pending = operations()[0]
    pending["anchor"].update(segmentation_id=raw["annotation_id"], span=[0, 2])
    corpus, sidecar, plan = make_plan(tmp_path, [segmentation, operations()[0], pending])
    suffix(sidecar, plan, segmentation)
    replacement = operations()[1]
    replacement["target"] = raw["annotation_id"]
    replacement["producer"]["name"] = "unplanned"
    suffix(sidecar, plan, replacement)
    before = digest_bytes(sidecar.read_bytes())
    result = apply(corpus, sidecar, plan)
    assert result["exit_code"] == 4 and result["diagnostics"][0]["code"] == "binding_drift", result
    assert digest_bytes(sidecar.read_bytes()) == before


@pytest.mark.parametrize("operation", ["supersede", "retract"])
@pytest.mark.parametrize("status", [None, "accepted", "withheld"])
@pytest.mark.parametrize("denied", [False, True])
def test_target_decision_destructiveness_and_deny_precedence(
    tmp_path: Path, operation: str, status: str | None, denied: bool
) -> None:
    target_operation = operations()[1] if operation == "supersede" else retract()
    corpus, sidecar, plan = make_plan(tmp_path, [target_operation])
    decision(corpus, status)
    policy = write_policy(tmp_path, "deny") if denied else None
    replan(corpus, sidecar, plan, policy)
    row = json.loads(plan.read_text())["operations"][0]
    assert row["route"] == ("deny" if denied else "review")
    assert ("destructive" in row["reasons"]) == (status is not None)
    assert row["safety"] == {"decision_status": status, "dependents": [], "conflicts": []}
