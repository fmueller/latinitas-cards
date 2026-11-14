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
    try:
        with args.binding.open("rb") as source:
            bound_raw = source.read(TRANSPORT_LIMIT + 1)
        if len(bound_raw) > TRANSPORT_LIMIT:
            raise ValueError("binding exceeds transport limit")
        binding_hash = hashlib.sha256(bound_raw).hexdigest()
        if complete and retained is not None and count <= TASK_LIMIT:
            final_scope_eligible_ids = validate(decode(retained), decode(bound_raw), binding_hash, args.task_id)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        errors.append(f"validation failed: {type(error).__name__}")
    receipt = {
        "schema_version": "t185.capture.v1",
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
