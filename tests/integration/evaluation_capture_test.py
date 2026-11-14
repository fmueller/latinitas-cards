"""Synthetic research records exercise custody, not Latin reference judgments."""

import hashlib
import json
import os
import runpy
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/capture-evaluation.py"


def observation(value: Any = None) -> dict[str, Any]:
    return {
        "state": "unavailable" if value is None else "observed",
        "scope": "synthetic requested scope",
        "reason": "synthetic observation or explicit evidence gap",
        "value": value,
    }


def fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    binding = {
        "schema_version": "t185.binding.v1",
        "attempt_id": "synthetic-attempt",
        "role": "REF-B",
        "run_id": "synthetic-run-b",
        "packet_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "prompt_sha256": "3" * 64,
        "envelope_sha256": "4" * 64,
        "source_dictionary_sha256": "5" * 64,
        "tasks": {
            "task-b": {
                "language": "de",
                "requested": {
                    pid: {
                        "scope": "synthetic-scope",
                        "eligibility_by_role": dict.fromkeys(["REF-A", "REF-B", "ADJ"], scope),
                    }
                    for pid, scope in [("p7", "eligible"), ("p2", "audit")]
                },
            }
        },
        "citation_ids": ["cite-9", "cite-3"],
    }
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding), encoding="utf-8")
    record = {
        "schema_version": "t185.initial.v1",
        "phase": "initial",
        "binding_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "task_id": "task-b",
        "answerability": "partial",
        "limitations": ["synthetic limitation"],
        "run_observations": {
            key: observation("synthetic actual observation")
            for key in [
                "tools_network_scope",
                "prior_exposure",
                "conflicts_shared_dependencies",
                "timestamps",
                "hash_scope",
                "attempt_budget",
                "exposure_log",
            ]
        },
        "propositions": [
            {
                "id": pid,
                "assertion_ids": [f"local-{pid}"],
                "status": status,
                "answer": "synthetic statement" if status == "supported" else "",
                "citation_ids": ["cite-3"] if status == "supported" else [],
                "support": "synthetic support" if status == "supported" else "",
                "alternatives": ["synthetic alternative"],
                "caveats": ["synthetic caveat"],
                "applicability": "synthetic applicability",
                "segmentation_convention": "not applicable: synthetic label",
                "wrong_answer": "synthetic competing statement",
                "rejection": "synthetic rejection",
                "missing_evidence": [] if status == "supported" else ["synthetic gap"],
                "abstention_reason": "" if status == "supported" else "audit-only",
                "audit": {
                    **{
                        key: observation()
                        for key in [
                            "corroboration",
                            "answerability",
                            "segmentation",
                            "target_language_validation",
                            "counterexample_handling",
                            "native_abstention",
                            "normalized_counting",
                            "correction_links",
                            "reference_status",
                        ]
                    },
                    "processing_outcome": observation("attempted_complete"),
                    "source_support_disposition": observation(
                        {"correspondence": "pass", "entailment": "pass", "applicability": "pass"}
                    )
                    if status == "supported"
                    else observation(),
                },
            }
            for pid, status in [("p2", "withheld"), ("p7", "supported")]
        ],
    }
    return path, record


def capture(tmp_path: Path, binding: Path, raw: bytes, name: str = "capture") -> tuple[int, dict[str, Any]]:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--binding",
            str(binding),
            "--task-id",
            "task-b",
            "--output",
            str(tmp_path / name),
        ],
        input=raw,
        capture_output=True,
        check=False,
    )
    receipt = json.loads((tmp_path / name / "receipt.json").read_bytes())
    stored = (tmp_path / name / "original.bin").read_bytes()
    assert stored == raw[: 1024 * 1024]
    assert receipt["raw_sha256"] == hashlib.sha256(stored).hexdigest()
    assert receipt["retained_bytes"] == len(stored)
    return result.returncode, receipt


@pytest.mark.parametrize("size,valid", [(32768, True), (32769, False)])
def test_exact_byte_boundary_preserves_original(tmp_path: Path, size: int, valid: bool) -> None:
    binding, record = fixture(tmp_path)
    raw = json.dumps(record, ensure_ascii=False).encode("utf-8")
    raw += b" " * (size - len(raw))
    code, receipt = capture(tmp_path, binding, raw)
    assert (code == 0) is valid
    assert receipt["valid"] is valid
    assert receipt["capture_complete"] is True


@pytest.mark.parametrize("defect", ["missing", "unknown", "citation", "duplicate", "audit", "binding", "extra"])
def test_invalid_records_are_retained_without_repair(tmp_path: Path, defect: str) -> None:
    binding, record = fixture(tmp_path)
    if defect == "missing":
        del record["propositions"][1]["alternatives"]
    elif defect == "unknown":
        record["propositions"][1]["id"] = "unrequested"
    elif defect == "citation":
        record["propositions"][1]["citation_ids"] = ["unknown-citation"]
    elif defect == "duplicate":
        record["propositions"][1]["id"] = "p2"
    elif defect == "audit":
        record["propositions"][0]["status"] = "supported"
    elif defect == "binding":
        record["binding_sha256"] = "0" * 64
    else:
        record["discard_me"] = "must not silently drop extra fields"
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert code != 0 and receipt["valid"] is False and receipt["errors"]


@pytest.mark.parametrize("raw", [b'{"broken":', b"\xffnot-utf8", b'{"task_id":"a","task_id":"b"}', b"NaN"])
def test_malformed_original_is_hashable_and_never_overwritten(tmp_path: Path, raw: bytes) -> None:
    binding, _ = fixture(tmp_path)
    code, receipt = capture(tmp_path, binding, raw)
    assert code != 0 and receipt["capture_complete"] and not receipt["valid"]
    before = {p.name: p.read_bytes() for p in (tmp_path / "capture").iterdir()}
    duplicate = subprocess.run(
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
        input=b"replacement",
        capture_output=True,
        check=False,
    )
    assert duplicate.returncode != 0
    assert {p.name: p.read_bytes() for p in (tmp_path / "capture").iterdir()} == before


def test_transport_ceiling_reports_incomplete_capture(tmp_path: Path) -> None:
    binding, _ = fixture(tmp_path)
    code, receipt = capture(tmp_path, binding, b"x" * (1024 * 1024 + 17))
    assert code != 0 and receipt["valid"] is False
    assert receipt["capture_complete"] is False
    assert receipt["observed_bytes_at_least"] == 1024 * 1024 + 1


def test_documented_capacity_uses_all_four_propositions_without_losing_fields(tmp_path: Path) -> None:
    binding, record = fixture(tmp_path)
    bound = json.loads(binding.read_bytes())
    bound["tasks"]["task-b"]["requested"] = {
        f"p{i}": {
            "scope": "synthetic-scope",
            "eligibility_by_role": dict.fromkeys(["REF-A", "REF-B", "ADJ"], "eligible"),
        }
        for i in range(4)
    }
    binding.write_text(json.dumps(bound))
    record["binding_sha256"] = hashlib.sha256(binding.read_bytes()).hexdigest()
    prototype = record["propositions"][1]
    record["propositions"] = []
    for i in range(4):
        prop = deepcopy(prototype)
        prop["id"] = f"p{i}"
        prop["assertion_ids"] = [f"local-{i}"]
        for field in ["answer", "support", "applicability", "segmentation_convention", "wrong_answer", "rejection"]:
            prop[field] = "x" * 224
        for field in ["alternatives", "caveats", "missing_evidence"]:
            prop[field] = ["x" * 256] * 4
        prop["citation_ids"] = ["cite-9", "cite-3"]
        for field in prop["audit"]:
            if field != "source_support_disposition":
                prop["audit"][field] = observation("synthetic actual assessment")
        prop["audit"]["segmentation"] = observation(
            dict.fromkeys(["convention_ref", "units", "recombination", "component_functions", "support"], "x" * 64)
        )
        prop["audit"]["target_language_validation"] = observation(
            dict.fromkeys(["wording", "Latin_feature_or_sense_support", "meaning_check", "limitations"], "x" * 64)
        )
        prop["audit"]["corroboration"] = observation(
            {"citation_ids": ["cite-9"], "claim_scope": "synthetic scope", "disposition": "synthetic gap"}
        )
        prop["audit"]["processing_outcome"] = observation("attempted_complete")
        prop["audit"]["normalized_counting"] = observation(
            {
                "attempted": {"unit": "proposition", "scope": f"p{i}:initial:REF-B:de:eligible", "value": 1},
                "withheld": {"unit": "proposition", "scope": f"p{i}:initial:REF-B:de:eligible", "value": False},
            }
        )
        record["propositions"].append(prop)
    raw = json.dumps(record).encode()
    assert len(raw) == 32563 < 32768
    code, receipt = capture(tmp_path, binding, raw)
    assert code == 0 and receipt["valid"] is True


def test_missing_binding_still_preserves_received_raw_first(tmp_path: Path) -> None:
    code, receipt = capture(tmp_path, tmp_path / "absent.json", b'{"synthetic":"raw"}')
    assert code != 0 and receipt["capture_complete"] and not receipt["valid"]


@pytest.mark.parametrize("defect", ["missing_check", "empty_reason", "unsupported", "missing_peer", "phase"])
def test_output_specific_checks_and_all_role_scope_cannot_disappear(tmp_path: Path, defect: str) -> None:
    binding, record = fixture(tmp_path)
    prop = record["propositions"][1]
    if defect == "missing_check":
        del prop["audit"]["target_language_validation"]
    elif defect == "empty_reason":
        prop["audit"]["segmentation"]["reason"] = ""
    elif defect == "unsupported":
        prop["audit"]["source_support_disposition"]["value"]["entailment"] = "unavailable"
    elif defect == "missing_peer":
        bound = json.loads(binding.read_bytes())
        del bound["tasks"]["task-b"]["requested"]["p7"]["eligibility_by_role"]["ADJ"]
        binding.write_text(json.dumps(bound))
        record["binding_sha256"] = hashlib.sha256(binding.read_bytes()).hexdigest()
    else:
        record["phase"] = "adjudication"
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert code != 0 and not receipt["valid"]


def test_initial_role_support_does_not_confer_final_intersection(tmp_path: Path) -> None:
    binding, record = fixture(tmp_path)
    bound = json.loads(binding.read_bytes())
    bound["tasks"]["task-b"]["requested"]["p7"]["eligibility_by_role"]["ADJ"] = "audit"
    binding.write_text(json.dumps(bound))
    record["binding_sha256"] = hashlib.sha256(binding.read_bytes()).hexdigest()
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert code == 0 and receipt["valid"]
    assert receipt["final_scope_eligible_ids"] == []


def test_overflowing_json_number_is_not_a_finite_observation(tmp_path: Path) -> None:
    binding, record = fixture(tmp_path)
    record["run_observations"]["timestamps"] = observation("overflow-marker")
    raw = json.dumps(record).encode().replace(b'"overflow-marker"', b"1e999")
    code, receipt = capture(tmp_path, binding, raw)
    assert code != 0 and not receipt["valid"]


@pytest.mark.parametrize("field", ["segmentation", "target_language_validation", "corroboration"])
def test_observed_structured_checks_cannot_omit_their_details(tmp_path: Path, field: str) -> None:
    binding, record = fixture(tmp_path)
    record["propositions"][1]["audit"][field] = observation({"unrelated": "not enough"})
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert code != 0 and not receipt["valid"]


def test_assertion_ids_are_preserved_separately_from_requested_ids(tmp_path: Path) -> None:
    binding, record = fixture(tmp_path)
    record["propositions"][0]["assertion_ids"] = []
    record["propositions"][1]["assertion_ids"] = ["local-judgment-19"]
    raw = json.dumps(record).encode()
    code, receipt = capture(tmp_path, binding, raw)
    assert code == 0 and receipt["valid"]
    assert json.loads((tmp_path / "capture/original.bin").read_bytes())["propositions"][1]["id"] == "p7"


@pytest.mark.parametrize("ids", [[], ["local-p2"], ["local-p7", "local-p7"]])
def test_supported_assertion_ids_are_nonempty_and_task_unique(tmp_path: Path, ids: list[str]) -> None:
    binding, record = fixture(tmp_path)
    record["propositions"][1]["assertion_ids"] = ids
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert code != 0 and not receipt["valid"]


@pytest.mark.parametrize("value", ["unknown", None, 2])
def test_invalid_processing_outcome_is_rejected(tmp_path: Path, value: Any) -> None:
    binding, record = fixture(tmp_path)
    record["propositions"][1]["audit"]["processing_outcome"] = observation(value)
    record["propositions"][1]["audit"]["processing_outcome"]["state"] = "observed"
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert code != 0 and not receipt["valid"]


@pytest.mark.parametrize(
    "value",
    [
        False,
        "count",
        {},
        {"attempted": {"unit": "proposition", "scope": "p7", "value": -1}},
        {"n": {"unit": "task", "scope": "", "value": 1}},
        {"n": {"unit": "task", "scope": "p7", "value": 0.5}},
    ],
)
def test_normalized_counts_require_typed_scoped_values(tmp_path: Path, value: Any) -> None:
    binding, record = fixture(tmp_path)
    record["propositions"][1]["audit"]["normalized_counting"] = observation(value)
    code, receipt = capture(tmp_path, binding, json.dumps(record).encode())
    assert code != 0 and not receipt["valid"]


def test_stream_failure_retains_early_spooled_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding, _ = fixture(tmp_path)
    output = tmp_path / "capture"
    reads = 0

    def interrupted_read(fd: int, size: int) -> bytes:
        nonlocal reads
        reads += 1
        if reads == 1:
            return b"synthetic partial raw"
        assert (output / "original.bin").read_bytes() == b"synthetic partial raw"
        raise OSError("synthetic interrupted transport")

    monkeypatch.setattr(os, "read", interrupted_read)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--binding", str(binding), "--task-id", "task-b", "--output", str(output)]
    )
    harness = runpy.run_path(str(SCRIPT))
    with binding.open() as stdin:
        monkeypatch.setattr(sys, "stdin", stdin)
        assert harness["main"]() == 1
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["capture_complete"] is False and receipt["valid"] is False
    assert receipt["retained_bytes"] == len(b"synthetic partial raw")
    assert receipt["raw_sha256"] == hashlib.sha256(b"synthetic partial raw").hexdigest()
