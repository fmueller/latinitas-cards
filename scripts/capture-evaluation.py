"""Capture original T-185 task bytes before validation; standard library only.

See docs/evaluation-harness-v1.md for the versioned binding and record contract.
This validates research transport/structure, never linguistic truth or policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

TASK_LIMIT = 32768
TRANSPORT_LIMIT = 1024 * 1024
TEXT_FIELDS = {
    "answer",
    "support",
    "applicability",
    "segmentation_convention",
    "wrong_answer",
    "rejection",
    "abstention_reason",
}
LIST_FIELDS = {"citation_ids", "alternatives", "caveats", "missing_evidence", "assertion_ids"}
PROCESSING_OUTCOMES = {
    "not_attempted",
    "attempted_complete",
    "failed_partial",
    "failed_malformed",
    "failed_over_limit",
    "failed_processing",
    "failed_custody_gate",
}
AUDIT_FIELDS = {
    "corroboration",
    "answerability",
    "segmentation",
    "target_language_validation",
    "counterexample_handling",
    "native_abstention",
    "normalized_counting",
    "correction_links",
    "reference_status",
    "processing_outcome",
    "source_support_disposition",
}
RUN_OBSERVATIONS = {
    "tools_network_scope",
    "prior_exposure",
    "conflicts_shared_dependencies",
    "timestamps",
    "hash_scope",
    "attempt_budget",
    "exposure_log",
}
ROLES = {"REF-A", "REF-B", "ADJ"}
STRUCTURED_OBSERVATIONS = {
    "segmentation": {"convention_ref", "units", "recombination", "component_functions", "support"},
    "target_language_validation": {"wording", "Latin_feature_or_sense_support", "meaning_check", "limitations"},
    "corroboration": {"citation_ids", "claim_scope", "disposition"},
}


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError("non-JSON numeric constant")


def finite_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def decode(raw: bytes) -> Any:
    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant, parse_float=finite_number
    )


def strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def observations(value: Any, keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("missing/extra output-specific observation fields")
    for key, item in value.items():
        if not isinstance(item, dict) or set(item) != {"state", "scope", "reason", "value"}:
            raise ValueError("invalid scoped observation")
        if item["state"] not in {"observed", "unavailable", "not_applicable"}:
            raise ValueError("invalid observation state")
        if any(not isinstance(item[key], str) or not item[key].strip() for key in ("scope", "reason")):
            raise ValueError("observation requires explicit scope and reason")
        if (item["state"] == "observed") != (item["value"] is not None):
            raise ValueError("observation state/value mismatch")
        if item["state"] == "observed" and key in STRUCTURED_OBSERVATIONS:
            details = item["value"]
            if not isinstance(details, dict) or set(details) != STRUCTURED_OBSERVATIONS[key]:
                raise ValueError("structured observation details missing/extra")
            if any(value is None for value in details.values()):
                raise ValueError("structured observation details cannot be null")
        if item["state"] == "observed" and key == "processing_outcome" and item["value"] not in PROCESSING_OUTCOMES:
            raise ValueError("invalid processing outcome")
        if item["state"] == "observed" and key == "normalized_counting":
            measures = item["value"]
            if not isinstance(measures, dict) or not measures:
                raise ValueError("normalized counting requires named measurements")
            for name, measure in measures.items():
                if not name.strip() or not isinstance(measure, dict) or set(measure) != {"unit", "scope", "value"}:
                    raise ValueError("invalid normalized measurement")
                if measure["unit"] not in {"task", "proposition"}:
                    raise ValueError("invalid normalized measurement unit")
                if not isinstance(measure["scope"], str) or not measure["scope"].strip():
                    raise ValueError("normalized measurement scope missing")
                if not isinstance(measure["value"], int) or measure["value"] < 0:
                    raise ValueError("normalized measurement requires nonnegative integer or boolean")


def validate(record: Any, binding: Any, binding_hash: str, task_id: str) -> list[str]:
    """Reject structural errors; do not mutate or normalize either input."""
    if not isinstance(binding, dict) or binding.get("schema_version") != "t185.binding.v1":
        raise ValueError("invalid binding version/object")
    for key in ("attempt_id", "run_id", "role"):
        if not isinstance(binding.get(key), str) or not binding[key].strip():
            raise ValueError(f"binding missing {key}")
    if binding["role"] not in ROLES:
        raise ValueError("invalid binding role")
    for key in ("packet_sha256", "protocol_sha256", "prompt_sha256", "envelope_sha256", "source_dictionary_sha256"):
        if not digest(binding.get(key)):
            raise ValueError(f"invalid binding {key}")
    citations = binding.get("citation_ids")
    if not isinstance(citations, list) or not strings(citations):
        raise ValueError("invalid binding citation IDs")
    known_citations = set(citations)
    if len(citations) != len(known_citations):
        raise ValueError("invalid binding citation IDs")
    tasks = binding.get("tasks")
    if not isinstance(tasks, dict) or task_id not in tasks:
        raise ValueError("unknown requested task")
    task = tasks[task_id]
    if not isinstance(task, dict) or task.get("language") not in {"de", "en"}:
        raise ValueError("invalid bound task language")
    requested = task.get("requested")
    if not isinstance(requested, dict) or not 1 <= len(requested) <= 4:
        raise ValueError("invalid bound proposition requests")
    for pid, request in requested.items():
        if not pid.strip() or not isinstance(request, dict) or set(request) != {"scope", "eligibility_by_role"}:
            raise ValueError("invalid scoped request")
        if not isinstance(request["scope"], str) or not request["scope"].strip():
            raise ValueError("request scope missing")
        roles = request["eligibility_by_role"]
        if (
            not isinstance(roles, dict)
            or set(roles) != ROLES
            or any(v not in {"eligible", "audit"} for v in roles.values())
        ):
            raise ValueError("all three role eligibility bindings required")
    keys = {
        "schema_version",
        "phase",
        "binding_sha256",
        "task_id",
        "answerability",
        "limitations",
        "propositions",
        "run_observations",
    }
    if not isinstance(record, dict) or set(record) != keys:
        raise ValueError("record fields do not match v1")
    if (
        record["schema_version"] != "t185.initial.v1"
        or record["phase"] != "initial"
        or record["binding_sha256"] != binding_hash
    ):
        raise ValueError("record version/binding mismatch")
    if record["task_id"] != task_id or record["answerability"] not in {"answerable", "partial", "insufficient"}:
        raise ValueError("invalid task identity/answerability")
    if not strings(record["limitations"]):
        raise ValueError("invalid limitations")
    observations(record["run_observations"], RUN_OBSERVATIONS)
    props = record["propositions"]
    if not isinstance(props, list) or len(props) != len(requested):
        raise ValueError("requested proposition count mismatch")
    seen: set[str] = set()
    assertion_ids: set[str] = set()
    for prop in props:
        if not isinstance(prop, dict) or set(prop) != TEXT_FIELDS | LIST_FIELDS | {"id", "status", "audit"}:
            raise ValueError("proposition fields do not match v1")
        pid = prop["id"]
        if not isinstance(pid, str) or pid not in requested or pid in seen:
            raise ValueError("unknown or repeated proposition ID")
        seen.add(pid)
        if prop["status"] not in {"supported", "withheld", "insufficient"}:
            raise ValueError("invalid proposition status")
        if any(not isinstance(prop[key], str) for key in TEXT_FIELDS):
            raise ValueError("invalid proposition text field")
        if any(not strings(prop[key]) for key in LIST_FIELDS):
            raise ValueError("invalid proposition list field")
        local_ids = prop["assertion_ids"]
        if len(local_ids) != len(set(local_ids)) or assertion_ids.intersection(local_ids):
            raise ValueError("repeated judge-local assertion ID")
        assertion_ids.update(local_ids)
        if not set(prop["citation_ids"]) <= known_citations:
            raise ValueError("unknown citation reference")
        observations(prop["audit"], AUDIT_FIELDS)
        corroboration = prop["audit"]["corroboration"]
        if corroboration["state"] == "observed":
            refs = corroboration["value"]["citation_ids"]
            if not strings(refs) or not set(refs) <= known_citations:
                raise ValueError("unknown corroboration citation reference")
        disposition = prop["audit"]["source_support_disposition"]
        if disposition["state"] == "observed":
            checks = disposition["value"]
            if (
                not isinstance(checks, dict)
                or set(checks) != {"correspondence", "entailment", "applicability"}
                or any(v not in {"pass", "fail", "unavailable"} for v in checks.values())
            ):
                raise ValueError("separate correspondence/entailment/applicability checks required")
        if prop["status"] == "supported":
            if not local_ids:
                raise ValueError("supported proposition lacks judge-local assertion identity")
            if requested[pid]["eligibility_by_role"][binding["role"]] != "eligible" or not prop["citation_ids"]:
                raise ValueError("unsupported eligibility/citation assertion")
            if not all(prop[key].strip() for key in ("answer", "support", "applicability")):
                raise ValueError("supported assertion lacks answer/support/applicability")
            if disposition["state"] != "observed" or any(v != "pass" for v in disposition["value"].values()):
                raise ValueError("supported assertion lacks complete support checks")
        elif not prop["abstention_reason"].strip():
            raise ValueError("withheld/insufficient proposition lacks reason")
    return sorted(
        pid
        for pid, request in requested.items()
        if all(v == "eligible" for v in request["eligibility_by_role"].values())
    )


def checked_artifact(ref: dict[str, Any], kind: str, root: Path, limit: int) -> bytes:
    """Read bounded, hash-bound custody bytes without rewriting the source."""
    if not isinstance(ref.get(f"{kind}_path"), str) or not digest(ref.get(f"{kind}_sha256")):
        raise ValueError("invalid initial artifact reference")
    with (root / Path(ref[f"{kind}_path"])).open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != ref[f"{kind}_sha256"]:
        raise ValueError("initial artifact size/hash mismatch")
    return raw


def counts(props: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [p["audit"]["processing_outcome"] for p in props]
    known = all(o["state"] == "observed" for o in outcomes)
    return {
        "requested": len(props),
        "attempted": sum(o["value"] != "not_attempted" for o in outcomes) if known else None,
        "withheld": sum(p["status"] != "supported" for p in props),
        "failed": sum(o["value"].startswith("failed_") for o in outcomes) if known else None,
    }


def validate_comparison(
    record: Any, binding: Any, binding_hash: str, task_id: str, root: Path, output: Path
) -> tuple[list[str], str, dict[str, Any]]:
    """Verify three unchanged initial captures before validating a comparison."""
    if binding.get("role") != "ADJ" or binding.get("initials_before_peer_exposure") is not True:
        raise ValueError("comparison requires ADJ and independent frozen initials")
    for key in ("attempt_id", "thread_id", "run_id", "safe_report_path"):
        if not isinstance(binding.get(key), str) or not binding[key].strip():
            raise ValueError("comparison identity missing")
    for key in (
        "packet_sha256",
        "source_dictionary_sha256",
        "protocol_sha256",
        "schema_sha256",
        "harness_sha256",
        "prompt_sha256",
        "envelope_sha256",
    ):
        if not digest(binding.get(key)):
            raise ValueError("comparison commitment missing")
    if binding["harness_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("comparison harness mismatch")
    tasks = binding.get("tasks")
    if not isinstance(tasks, dict) or len(tasks) != 4 or task_id not in tasks:
        raise ValueError("comparison allocation must contain exactly four tasks")
    directories = set()
    for key, slot in tasks.items():
        if (
            not key.strip()
            or not isinstance(slot, dict)
            or not isinstance(slot.get("output_directory"), str)
            or not slot["output_directory"].strip()
        ):
            raise ValueError("invalid comparison task slot")
        directories.add((root / slot["output_directory"]).resolve())
    if len(directories) != 4 or binding.get("safe_report_limit") != 65536:
        raise ValueError("distinct task slots and 65536-byte safe report limit required")
    task = tasks[task_id]
    if (root / task["output_directory"]).resolve() != output.resolve():
        raise ValueError("comparison output differs from frozen task slot")
    initials = task["initials"]
    if not isinstance(initials, dict) or set(initials) != ROLES:
        raise ValueError("three initial roles required")
    positions: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    initial_counts: dict[str, Any] = {}
    eligible: list[str] = []
    threads: set[str] = set()
    for role, ref in initials.items():
        if not isinstance(ref, dict) or ref.get("frozen") is not True:
            raise ValueError("unfrozen initial")
        thread = ref.get("thread_id")
        if not isinstance(thread, str) or not thread.strip() or thread in threads:
            raise ValueError("distinct initial identities required")
        threads.add(thread)
        original_raw = checked_artifact(ref, "original", root, TASK_LIMIT)
        receipt = decode(checked_artifact(ref, "receipt", root, TRANSPORT_LIMIT))
        initial_binding = decode(checked_artifact(ref, "binding", root, TRANSPORT_LIMIT))
        original = decode(original_raw)
        eligible = validate(original, initial_binding, ref["binding_sha256"], task_id)
        if (
            initial_binding["role"] != role
            or initial_binding["attempt_id"] != binding["attempt_id"]
            or initial_binding["run_id"] != ref.get("run_id")
            or initial_binding["protocol_sha256"] != ref.get("initial_protocol_sha256")
            or any(initial_binding[key] != binding[key] for key in ("packet_sha256", "source_dictionary_sha256"))
        ):
            raise ValueError("initial identity/protocol/dictionary mismatch")
        if (
            receipt.get("schema_version") != "t185.capture.v1"
            or receipt.get("valid") is not True
            or receipt.get("capture_complete") is not True
            or receipt.get("errors") != []
            or receipt.get("raw_sha256") != ref["original_sha256"]
            or receipt.get("binding_sha256") != ref["binding_sha256"]
            or receipt.get("task_id") != task_id
            or receipt.get("retained_bytes") != len(original_raw)
            or receipt.get("observed_bytes_at_least") != len(original_raw)
            or receipt.get("task_limit") != TASK_LIMIT
            or receipt.get("transport_limit") != TRANSPORT_LIMIT
            or receipt.get("final_scope_eligible_ids") != eligible
        ):
            raise ValueError("initial receipt is incomplete/invalid/inconsistent")
        positions[role] = {p["id"]: p for p in original["propositions"]}
        bindings[role] = initial_binding
        initial_counts[role] = {
            "unit": "proposition",
            "role": role,
            "phase": "initial",
            "language": initial_binding["tasks"][task_id]["language"],
            **counts(original["propositions"]),
        }
    adj = bindings["ADJ"]
    if initials["ADJ"]["thread_id"] != binding["thread_id"]:
        raise ValueError("comparison ADJ identity differs from initial")
    if any(
        b["tasks"][task_id] != adj["tasks"][task_id] or set(b["citation_ids"]) != set(adj["citation_ids"])
        for b in bindings.values()
    ):
        raise ValueError("initial request/eligibility/citation definitions disagree")
    known_citations = set(adj["citation_ids"])
    keys = {
        "schema_version",
        "phase",
        "binding_sha256",
        "attempt_id",
        "task_id",
        "role",
        "thread_id",
        "decision_basis",
        "answerability",
        "limitations",
        "run_observations",
        "results",
        "counts",
    }
    if not isinstance(record, dict) or set(record) != keys:
        raise ValueError("comparison fields missing/extra")
    if (
        record["schema_version"] != "t185.comparison.v1"
        or record["phase"] != "comparison"
        or record["binding_sha256"] != binding_hash
    ):
        raise ValueError("comparison version/binding mismatch")
    if (
        record["task_id"] != task_id
        or record["decision_basis"] != "source_based"
        or any(record[key] != binding[key] for key in ("attempt_id", "role", "thread_id"))
    ):
        raise ValueError("comparison identity/basis mismatch")
    results = record["results"]
    if not isinstance(results, list) or len(results) != len(positions["ADJ"]):
        raise ValueError("comparison requested count mismatch")
    refs = []
    for result in results:
        if not isinstance(result, dict) or set(result) != {
            "id",
            "reference",
            "positions",
            "disputed_assertion_ids",
            "agreement_audit",
            "source_checks",
            "changes",
            "unresolved",
        }:
            raise ValueError("comparison result fields missing/extra")
        pid = result["id"]
        if not isinstance(pid, str) or pid not in positions["ADJ"] or result["reference"]["id"] != pid:
            raise ValueError("comparison request/reference mismatch")
        if not isinstance(result["agreement_audit"], str) or not result["agreement_audit"].strip():
            raise ValueError("agreement/shared-error audit missing")
        for field in ("positions", "disputed_assertion_ids", "source_checks", "changes"):
            if not isinstance(result[field], dict) or set(result[field]) != ROLES:
                raise ValueError("comparison requires all role positions/audits/dispositions")
        for role in ROLES:
            original = positions[role][pid]
            if result["positions"][role] != original["assertion_ids"]:
                raise ValueError("initial assertion reference mismatch")
            disputed = result["disputed_assertion_ids"][role]
            if (
                not strings(disputed)
                or len(set(disputed)) != len(disputed)
                or not set(disputed) <= set(original["assertion_ids"])
            ):
                raise ValueError("unknown/duplicate disputed assertion")
            checks = result["source_checks"][role]
            if not isinstance(checks, dict) or set(checks) != {
                "correspondence",
                "entailment",
                "applicability",
                "reason",
            }:
                raise ValueError("initial source checks missing")
            if (
                any(
                    checks[k] not in {"pass", "fail", "unavailable"}
                    for k in ("correspondence", "entailment", "applicability")
                )
                or not isinstance(checks["reason"], str)
                or not checks["reason"].strip()
            ):
                raise ValueError("invalid initial source checks")
            change = result["changes"][role]
            if not isinstance(change, dict) or set(change) != {"disposition", "reason", "citation_ids"}:
                raise ValueError("change disposition missing")
            if (
                change["disposition"] not in {"changed", "unchanged", "unresolved"}
                or not isinstance(change["reason"], str)
                or not change["reason"].strip()
                or not strings(change["citation_ids"])
                or not set(change["citation_ids"]) <= known_citations
            ):
                raise ValueError("invalid change reason/citations")
            if change["disposition"] == "changed" and not change["citation_ids"]:
                raise ValueError("changed reference requires cited reason")
        reference = result["reference"]
        if not strings(result["unresolved"]):
            raise ValueError("reference eligibility/unresolved mismatch")
        if reference["status"] == "supported":
            if (
                pid not in eligible
                or result["unresolved"]
                or any(c["disposition"] == "unresolved" for c in result["changes"].values())
            ):
                raise ValueError("reference eligibility/unresolved mismatch")
        elif not result["unresolved"]:
            raise ValueError("reference eligibility/unresolved mismatch")
        refs.append(reference)
    # Reuse v1's reference-field contract on a derived view, never rewrite an initial.
    validate(
        {
            "schema_version": "t185.initial.v1",
            "phase": "initial",
            "binding_sha256": initials["ADJ"]["binding_sha256"],
            "task_id": task_id,
            "answerability": record["answerability"],
            "limitations": record["limitations"],
            "run_observations": record["run_observations"],
            "propositions": refs,
        },
        adj,
        initials["ADJ"]["binding_sha256"],
        task_id,
    )
    expected_counts = {
        "unit": "proposition",
        "role": "ADJ",
        "phase": "comparison",
        "language": adj["tasks"][task_id]["language"],
        "adjudicable": len(refs),
        **counts(refs),
    }
    if record["counts"] != expected_counts or any(
        type(record["counts"][k]) is not type(v) for k, v in expected_counts.items()
    ):
        raise ValueError("comparison counts mismatch")
    projection = {
        "schema_version": "t185.reference.v1",
        "binding_sha256": binding_hash,
        "task_id": task_id,
        "results": results,
    }
    canonical = json.dumps(projection, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return eligible, hashlib.sha256(canonical).hexdigest(), initial_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new private directory; never reused")
    args = parser.parse_args()
    try:
        args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError:
        print("capture directory unavailable/already exists; nothing overwritten", file=sys.stderr)
        return 2
    errors: list[str] = []
    count = 0
    observed = 0
    complete = False
    try:
        with (args.output / "original.bin").open("xb") as output:
            os.chmod(output.name, 0o600)
            while count < TRANSPORT_LIMIT:
                chunk = os.read(sys.stdin.fileno(), min(8192, TRANSPORT_LIMIT - count))
                if not chunk:
                    complete = True
                    break
                observed += len(chunk)
                output.write(chunk)
                output.flush()
                count += len(chunk)
            if not complete:
                extra = os.read(sys.stdin.fileno(), 1)
                observed += len(extra)
                complete = not extra
            os.fsync(output.fileno())
    except OSError:
        errors.append("capture I/O failure; original may be incomplete")
        complete = False
    retained: bytes | None = None
    try:
        retained = (args.output / "original.bin").read_bytes()
        count = len(retained)
    except OSError:
        errors.append("retained bytes unavailable; hash/count cannot be established")
        complete = False
    if not complete:
        errors.append("incomplete capture; retained prefix is not a complete original")
    if count > TASK_LIMIT:
        errors.append("original exceeds 32768-byte task limit")
    binding_hash: str | None = None
    final_scope_eligible_ids: list[str] = []
    comparison_receipt: dict[str, Any] = {}
    receipt_version = "t185.capture.v1"
    try:
        with args.binding.open("rb") as source:
            bound_raw = source.read(TRANSPORT_LIMIT + 1)
        if len(bound_raw) > TRANSPORT_LIMIT:
            raise ValueError("binding exceeds transport limit")
        binding_hash = hashlib.sha256(bound_raw).hexdigest()
        if complete and retained is not None and count <= TASK_LIMIT:
            binding = decode(bound_raw)
            if isinstance(binding, dict) and binding.get("schema_version") == "t185.comparison-binding.v1":
                receipt_version = "t185.comparison-capture.v1"
                final_scope_eligible_ids, reference_hash, initial_counts = validate_comparison(
                    decode(retained), binding, binding_hash, args.task_id, args.binding.parent, args.output
                )
                comparison_receipt = {
                    "final_reference_sha256": reference_hash,
                    "final_reference_schema": "t185.reference.v1",
                    "initial_counts": initial_counts,
                }
            else:
                final_scope_eligible_ids = validate(decode(retained), binding, binding_hash, args.task_id)
    except (OSError, ValueError, TypeError, RecursionError, KeyError, AttributeError) as error:
        errors.append(f"validation failed: {type(error).__name__}")
    receipt = {
        "schema_version": receipt_version,
        "task_id": args.task_id,
        "binding_sha256": binding_hash,
        "final_scope_eligible_ids": final_scope_eligible_ids,
        "raw_sha256": hashlib.sha256(retained).hexdigest() if retained is not None else None,
        "retained_bytes": count if retained is not None else None,
        "observed_bytes_at_least": observed,
        "capture_complete": complete,
        "task_limit": TASK_LIMIT,
        "transport_limit": TRANSPORT_LIMIT,
        "valid": not errors,
        "errors": errors,
        **comparison_receipt,
    }
    with (args.output / "receipt.json").open("x", encoding="utf-8") as target:
        os.chmod(target.name, 0o600)
        json.dump(receipt, target, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    return 0 if receipt["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
