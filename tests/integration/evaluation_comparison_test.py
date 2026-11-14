"""Answer-free comparison records exercise frozen references, never study solves."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from evaluation_capture_test import SCRIPT, capture, fixture


def comparison(
    tmp_path: Path, *, four: bool = False, audit_role: str | None = None, ref_a_citations: list[str] | None = None
) -> tuple[Path, dict[str, Any]]:
    tasks = ["task-b", "task-c", "task-d", "task-e"]
    initials: dict[str, Any] = {}
    reference: list[dict[str, Any]] = []
    run_observations: dict[str, Any] = {}
    for role in ["REF-A", "REF-B", "ADJ"]:
        root = tmp_path / role
        root.mkdir()
        initial_binding, initial = fixture(root)
        bound = json.loads(initial_binding.read_bytes())
        bound["role"] = role
        bound["run_id"] = f"initial-{role}"
        if role == "REF-A" and ref_a_citations is not None:
            bound["citation_ids"] = ref_a_citations
        requested = bound["tasks"]["task-b"]["requested"]
        if four:
            for pid in ("p3", "p9"):
                requested[pid] = deepcopy(requested["p7"])
                prop = deepcopy(initial["propositions"][1])
                prop["id"] = pid
                prop["assertion_ids"] = [f"local-{pid}"]
                initial["propositions"].append(prop)
        if audit_role:
            requested["p7"]["eligibility_by_role"][audit_role] = "audit"
            if role == audit_role:
                initial["propositions"][1]["status"] = "withheld"
                initial["propositions"][1]["abstention_reason"] = "synthetic role-only audit"
        bound["tasks"] = dict.fromkeys(tasks, bound["tasks"]["task-b"])
        initial_binding.write_text(json.dumps(bound))
        initial["binding_sha256"] = hashlib.sha256(initial_binding.read_bytes()).hexdigest()
        # A's distinct initial position ensures comparison is not a vote/count test.
        if role == "REF-A":
            initial["propositions"][1]["answer"] = "synthetic alternate position"
        code, receipt = capture(root, initial_binding, json.dumps(initial).encode())
        assert code == 0 and receipt["valid"]
        initials[role] = {
            "thread_id": f"synthetic-thread-{role}",
            "run_id": bound["run_id"],
            "frozen": True,
            "initial_protocol_sha256": bound["protocol_sha256"],
            **{
                f"{kind}_{key}": value
                for kind, path in [
                    ("original", root / "capture/original.bin"),
                    ("receipt", root / "capture/receipt.json"),
                    ("binding", initial_binding),
                ]
                for key, value in [("path", str(path)), ("sha256", hashlib.sha256(path.read_bytes()).hexdigest())]
            },
        }
        if role == "ADJ":
            reference = initial["propositions"]
            run_observations = initial["run_observations"]
    bound = {
        "schema_version": "t185.comparison-binding.v1",
        "attempt_id": "synthetic-attempt",
        "role": "ADJ",
        "thread_id": "synthetic-thread-ADJ",
        "run_id": "synthetic-comparison",
        "packet_sha256": "1" * 64,
        "source_dictionary_sha256": "5" * 64,
        "protocol_sha256": "6" * 64,
        "schema_sha256": "7" * 64,
        "harness_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "prompt_sha256": "8" * 64,
        "envelope_sha256": "9" * 64,
        "initials_before_peer_exposure": True,
        "safe_report_limit": 65536,
        "safe_report_path": "answer-free-report.json",
        "tasks": {
            task: {"output_directory": str(tmp_path / ("capture" if task == "task-b" else task)), "initials": initials}
            for task in tasks
        },
    }
    binding = tmp_path / "comparison-binding.json"
    binding.write_text(json.dumps(bound))
    record = {
        "schema_version": "t185.comparison.v1",
        "phase": "comparison",
        "binding_sha256": hashlib.sha256(binding.read_bytes()).hexdigest(),
        "attempt_id": bound["attempt_id"],
        "task_id": "task-b",
        "role": "ADJ",
        "thread_id": bound["thread_id"],
        "decision_basis": "source_based",
        "answerability": "partial",
        "limitations": ["synthetic shared-model limitation; runtime/cost unavailable"],
        "run_observations": run_observations,
        "results": [
            {
                "id": prop["id"],
                "reference": prop,
                "positions": {role: [f"local-{prop['id']}"] for role in initials},
                "disputed_assertion_ids": {
                    role: ([f"local-{prop['id']}"] if prop["id"] == "p7" else []) for role in initials
                },
                "agreement_audit": "synthetic audit of agreement and shared errors",
                "source_checks": {
                    role: {
                        "correspondence": "pass",
                        "entailment": "unavailable" if role == "REF-A" else "pass",
                        "applicability": "pass",
                        "reason": "synthetic independent source assessment",
                    }
                    for role in initials
                },
                "changes": {
                    role: {
                        "disposition": "changed" if role == "REF-A" and prop["id"] == "p7" else "unchanged",
                        "reason": "synthetic cited comparison",
                        "citation_ids": ["cite-3"],
                    }
                    for role in initials
                },
                "unresolved": [] if prop["status"] == "supported" else ["synthetic audit-only gap"],
            }
            for prop in reference
        ],
        "counts": {
            "unit": "proposition",
            "role": "ADJ",
            "phase": "comparison",
            "language": "de",
            "requested": len(reference),
            "attempted": len(reference),
            "adjudicable": len(reference),
            "withheld": 1,
            "failed": 0,
        },
    }
    return binding, record


def rebind(binding: Path, record: dict[str, Any], bound: dict[str, Any]) -> None:
    binding.write_text(json.dumps(bound))
    record["binding_sha256"] = hashlib.sha256(binding.read_bytes()).hexdigest()


def test_asymmetric_comparison_preserves_initials_and_derives_reference(tmp_path: Path) -> None:
    binding, record = comparison(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.glob("*/capture/*")}
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert code == 0 and receipt["valid"]
    assert receipt["schema_version"] == "t185.comparison-capture.v1"
    projection = {
        "schema_version": "t185.reference.v1",
        "binding_sha256": record["binding_sha256"],
        "task_id": "task-b",
        "results": record["results"],
    }
    canonical = json.dumps(projection, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    assert receipt["final_reference_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert receipt["final_scope_eligible_ids"] == ["p7"]
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize(
    "defect",
    [
        "role",
        "attempt",
        "task",
        "hash",
        "unfrozen",
        "exposure",
        "reference",
        "agreement",
        "disposition",
        "unresolved",
        "count",
        "support",
        "open_change",
        "duplicate",
        "initial_receipt",
    ],
)
def test_invalid_comparison_is_retained_without_repair(tmp_path: Path, defect: str) -> None:
    binding, record = comparison(tmp_path)
    bound = json.loads(binding.read_bytes())
    result = record["results"][1]
    if defect == "role":
        record["role"] = "REF-A"
    elif defect == "attempt":
        record["attempt_id"] = "different-attempt"
    elif defect == "task":
        record["task_id"] = "different-task"
    elif defect == "hash":
        bound["tasks"]["task-b"]["initials"]["REF-B"]["original_sha256"] = "0" * 64
    elif defect == "unfrozen":
        bound["tasks"]["task-b"]["initials"]["ADJ"]["frozen"] = False
    elif defect == "exposure":
        bound["initials_before_peer_exposure"] = False
    elif defect == "reference":
        result["positions"]["REF-A"] = ["unknown-local-id"]
    elif defect == "agreement":
        del result["agreement_audit"]
    elif defect == "disposition":
        del result["changes"]["ADJ"]
    elif defect == "unresolved":
        del result["unresolved"]
    elif defect == "count":
        record["counts"]["withheld"] = 0
    elif defect == "support":
        result["reference"]["audit"]["source_support_disposition"]["value"]["entailment"] = "fail"
    elif defect == "open_change":
        result["changes"]["ADJ"]["disposition"] = "unresolved"
    elif defect == "duplicate":
        record["results"][0] = deepcopy(result)
    else:
        ref = bound["tasks"]["task-b"]["initials"]["REF-B"]
        path = Path(ref["receipt_path"])
        receipt = json.loads(path.read_bytes())
        receipt["capture_complete"] = False
        path.write_text(json.dumps(receipt))
        ref["receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rebind(binding, record, bound)
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert code != 0 and not receipt["valid"]
    assert receipt.get("final_reference_sha256") is None


@pytest.mark.parametrize("size,valid", [(32768, True), (32769, False)])
def test_comparison_byte_boundary(tmp_path: Path, size: int, valid: bool) -> None:
    binding, record = comparison(tmp_path)
    raw = json.dumps(record).encode()
    assert len(raw) < 32768
    raw += b" " * (size - len(raw))
    code, receipt = capture(tmp_path, binding, raw)
    assert (code == 0) is valid


@pytest.mark.parametrize("raw", [b"{bad", b'{"a":1,"a":2}', b'{"n":1e999}'])
def test_comparison_malformed_original_is_retained(tmp_path: Path, raw: bytes) -> None:
    binding, _ = comparison(tmp_path)
    code, receipt = capture(tmp_path, binding, raw)
    assert code != 0 and not receipt["valid"]


def test_final_support_requires_all_role_scope_intersection(tmp_path: Path) -> None:
    binding, record = comparison(tmp_path, audit_role="REF-A")
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert code != 0 and not receipt["valid"]


def test_comparison_cannot_overwrite_a_failed_original(tmp_path: Path) -> None:
    binding, record = comparison(tmp_path)
    code, receipt = capture(tmp_path, binding, b"malformed")
    assert code != 0 and not receipt["valid"]
    before = {p: p.read_bytes() for p in (tmp_path / "capture").iterdir()}
    # Use the public subprocess invocation without the helper's equality check.
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--binding",
            str(binding),
            "--task-id",
            "task-b",
            "--output",
            str(tmp_path / "capture"),
        ],
        input=json.dumps(record).encode(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize("citations,valid", [(["cite-3", "cite-9"], True), (["cite-3", "other"], False)])
def test_comparison_citation_allowlists_use_membership(tmp_path: Path, citations: list[str], valid: bool) -> None:
    binding, record = comparison(tmp_path, ref_a_citations=citations)
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert receipt["valid"] is valid
    assert (code == 0) is valid


def test_full_four_request_comparison_capacity(tmp_path: Path) -> None:
    binding, record = comparison(tmp_path, four=True)
    for result in record["results"]:
        prop = result["reference"]
        for key in ("answer", "support", "applicability", "segmentation_convention", "wrong_answer", "rejection"):
            prop[key] = "x" * 128
        for key in ("alternatives", "caveats", "missing_evidence"):
            prop[key] = ["x" * 128] * 4
    raw = json.dumps(record).encode()
    assert len(raw) == 23998 < 32768
    code, receipt = capture(tmp_path, binding, raw)
    assert code == 0 and receipt["valid"]


@pytest.mark.parametrize(
    "defect",
    [
        "binding_role",
        "thread",
        "initial_binding",
        "old_protocol",
        "dictionary",
        "initial_invalid",
        "same_slot",
        "output_slot",
        "vote",
        "disputed",
        "citation",
    ],
)
def test_comparison_binding_and_evidence_integrity(tmp_path: Path, defect: str) -> None:
    binding, record = comparison(tmp_path)
    bound = json.loads(binding.read_bytes())
    ref = bound["tasks"]["task-b"]["initials"]["REF-A"]
    if defect == "binding_role":
        bound["role"] = "REF-A"
    elif defect == "thread":
        bound["thread_id"] = record["thread_id"] = "other-ADJ"
    elif defect == "initial_binding":
        ref["binding_sha256"] = "0" * 64
    elif defect == "old_protocol":
        ref["initial_protocol_sha256"] = "0" * 64
    elif defect == "dictionary":
        bound["source_dictionary_sha256"] = "0" * 64
    elif defect == "initial_invalid":
        path = Path(ref["receipt_path"])
        invalid = json.loads(path.read_bytes())
        invalid["valid"] = False
        path.write_text(json.dumps(invalid))
        ref["receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif defect == "same_slot":
        bound["tasks"]["task-c"]["output_directory"] = bound["tasks"]["task-b"]["output_directory"]
    elif defect == "output_slot":
        bound["tasks"]["task-b"]["output_directory"] = "different"
    elif defect == "vote":
        record["decision_basis"] = "majority"
    elif defect == "disputed":
        record["results"][1]["disputed_assertion_ids"]["ADJ"] = ["unknown"]
    else:
        record["results"][1]["changes"]["REF-A"]["citation_ids"] = ["unknown"]
    rebind(binding, record, bound)
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert code != 0 and not receipt["valid"]


@pytest.mark.parametrize("defect", ["nonfinite", "duplicate"])
def test_otherwise_valid_comparison_rejects_nonstandard_json(tmp_path: Path, defect: str) -> None:
    binding, record = comparison(tmp_path)
    record["run_observations"]["timestamps"]["value"] = "numeric-marker"
    raw = json.dumps(record).encode()
    raw = raw.replace(b'"numeric-marker"', b"1e999") if defect == "nonfinite" else raw[:-1] + b', "role": "ADJ"}'
    code, receipt = capture(tmp_path, binding, raw)
    assert code != 0 and not receipt["valid"]
