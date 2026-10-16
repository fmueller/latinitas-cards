"""Locked snapshot and atomic append transaction for planned annotations."""

import hashlib
import json
import os
from pathlib import Path

from pydantic import JsonValue

from .annotation_plans import AnnotationPlan, AnnotationReceipt, Retract, checked_record, load_document, proposed_record
from .annotation_sidecar import Annotation, Retraction, append_records, read_sidecar, resolve_created
from .atomic_io import reject_aliases, writer_locks
from .canonical import digest, digest_bytes
from .cli_contract import Code, ContractError, Diagnostic
from .corpus_io import read_corpus
from .corpus_model import CorpusHeader, CorpusPassage
from .corpus_validation import validate_records
from .policy import policy_digest


def check_sidecar_base(data: bytes | None, plan: AnnotationPlan) -> None:
    """Accept the exact base, or a line-boundary base followed only by this plan."""
    if (digest_bytes(data) if data is not None else None) == plan.base.sidecar:
        return
    if data is not None:
        lines = data.splitlines(keepends=True)
        prefix = hashlib.sha256()
        matched = plan.base.sidecar is None
        for line in lines:
            if matched:
                try:
                    row = json.loads(line)
                    # A formerly absent sidecar starts with its versioned header.
                    if plan.base.sidecar is None and prefix.digest() == hashlib.sha256().digest():
                        if row != {"format": "latinitas-annotations", "record": "header", "version": 1}:
                            break
                    elif not isinstance(row, dict) or row.get("plan") != plan.plan_id:
                        break
                except (ValueError, UnicodeError):
                    break
            prefix.update(line)
            if line.endswith(b"\n") and f"sha256:{prefix.hexdigest()}" == plan.base.sidecar:
                matched = True
        else:
            if matched:
                return
    raise ContractError(Code.BINDING_DRIFT, "Sidecar differs from the plan base; re-plan")


def apply_plan(
    plan_path: Path,
    corpus: Path,
    sidecar: Path,
    policy_path: Path | None,
    receipt_path: Path | None,
    timestamp: str | None,
) -> tuple[JsonValue, list[Diagnostic]]:
    """Validate inputs first, then bind and append under all three writer locks."""
    try:
        plan = load_document(plan_path, AnnotationPlan)
    except ContractError as error:
        raise ContractError(Code.INVALID_PLAN, str(error), path=str(plan_path)) from error
    assert isinstance(plan, AnnotationPlan)
    receipt: AnnotationReceipt | None = None
    if receipt_path is not None:
        try:
            loaded_receipt = load_document(receipt_path, AnnotationReceipt)
        except ContractError as error:
            raise ContractError(Code.INVALID_RECEIPT, str(error), path=str(receipt_path)) from error
        assert isinstance(loaded_receipt, AnnotationReceipt)
        receipt = loaded_receipt
    if any(row.route == "invalid" for row in plan.operations):
        raise ContractError(Code.INVALID_OPERATION, "Plan contains an invalid operation")
    if receipt is not None and receipt.plan_id != plan.plan_id:
        raise ContractError(Code.INVALID_RECEIPT, "Receipt names a different plan")
    try:
        for patch_operation, operation in zip(plan.patch.operations, plan.operations, strict=True):
            if len(operation.records) != 1:
                raise ValueError("Expected one record per operation")
            for raw in operation.records:
                record = checked_record(raw)
                if isinstance(patch_operation, Retract):
                    if (
                        not isinstance(record, Retraction)
                        or record.retracts != patch_operation.target
                        or record.producer != patch_operation.producer
                        or raw.get("created") != patch_operation.created
                    ):
                        raise ValueError("Retraction differs from the embedded patch")
                elif operation.records != [proposed_record(patch_operation, {})]:
                    raise ValueError("Planned record differs from the embedded patch")
    except (ValueError, KeyError) as error:
        raise ContractError(Code.INVALID_PLAN, str(error)) from error
    by_id = {row.operation_id: row for row in plan.operations}
    approved = set(receipt.operations) if receipt is not None else set()
    if approved - by_id.keys():
        raise ContractError(Code.UNKNOWN_OPERATION, "Receipt names an unknown operation")
    if any(path.suffix == ".lock" or path.name.endswith(".manifest.json") for path in (sidecar, sidecar.resolve())):
        raise ContractError(Code.ALIAS, "Sidecar must not be a lock or manifest", path=str(sidecar))
    claims = Path(f"{corpus}.claims.json")
    targets = [corpus, sidecar, claims]
    locks = [Path(f"{path.resolve()}.lock") for path in (corpus, sidecar)] + [Path(f"{claims}.lock")]
    protected = [plan_path, corpus, claims, corpus.with_suffix(".manifest.json")]
    protected.extend(path for path in (policy_path, receipt_path) if path is not None)
    reject_aliases(sidecar, [*protected, *locks])
    for index, lock in enumerate(locks):
        reject_aliases(lock, [*protected, sidecar, *locks[:index]])
    with writer_locks(targets, lock_paths=locks):
        reject_aliases(sidecar, [*protected, *locks])
        rows = list(read_corpus(corpus))
        _, findings = validate_records(iter(rows), path=str(corpus))
        if findings:
            return None, [
                Diagnostic(code=Code.INVALID_CORPUS, message=f"Invalid corpus: {findings[0].code}"),
                *findings,
            ]
        header = next(row for row in rows if isinstance(row, CorpusHeader))
        passages = {row.passage_id: row for row in rows if isinstance(row, CorpusPassage)}
        if (
            digest(header.model_dump(mode="json")) != plan.base.corpus_header
            or policy_digest(policy_path) != plan.base.policy
        ):
            raise ContractError(Code.BINDING_DRIFT, "Corpus header or policy differs from the plan base; re-plan")
        try:
            data = sidecar.read_bytes() if sidecar.exists() else None
        except OSError as error:
            raise ContractError(Code.UNREADABLE, str(error), path=str(sidecar)) from error
        check_sidecar_base(data, plan)
        current = list(read_sidecar(sidecar)) if data is not None else []
        targets_by_id = {row.annotation_id: row for row in current if isinstance(row, Annotation)}
        for patch_operation, operation in zip(plan.patch.operations, plan.operations, strict=True):
            if isinstance(patch_operation, Retract):
                expected = proposed_record(patch_operation, targets_by_id)
                if expected is None:
                    raise ContractError(Code.BINDING_DRIFT, "Retraction target is missing; re-plan")
                if operation.records != [expected]:
                    raise ContractError(Code.INVALID_PLAN, "Retraction binding differs from its target")
        known = {row.annotation_id for row in current if isinstance(row, Annotation | Retraction)}
        selected = [
            row
            for row in plan.operations
            if row.route == "auto" or (row.route == "review" and row.operation_id in approved)
        ]
        raw_records: dict[str, dict[str, JsonValue]] = {}
        for operation in selected:
            for raw in operation.records:
                if "annotation_id" in raw:
                    raw_records.setdefault(str(raw["annotation_id"]), raw)
        additions: list[Annotation | Retraction] = []
        for operation in selected:
            for raw in operation.records:
                try:
                    record = checked_record(raw)
                except (ValueError, KeyError) as error:
                    raise ContractError(Code.INVALID_PLAN, str(error)) from error
                if record.annotation_id in known:
                    continue
                passage = passages.get(record.anchor.passage_id)
                if passage is None or passage.passage_hash != record.input_hash:
                    raise ContractError(Code.BINDING_DRIFT, "Planned passage is missing or edited; re-plan")
                additions.append(record)
                known.add(record.annotation_id)
        # T-119 plugs current pending-operation route and complete safety comparison
        # into this locked snapshot, before actor checks, timestamps or replacement.
        if receipt is not None and receipt.actor.startswith("agent:") and plan.receipt_actor == "human":
            raise ContractError(Code.ACTOR_REFUSED, "A declared human actor is required")
        if any(by_id[operation_id].route == "deny" for operation_id in approved):
            raise ContractError(Code.DENIED, "Receipt selects a denied operation")
        for record in additions:
            raw_created = raw_records[record.annotation_id].get("created")
            created = str(raw_created) if raw_created is not None else resolve_created(timestamp, os.environ)
            record.created = created
            record.plan = plan.plan_id
        appended = append_records(sidecar, additions, protected, locked=True) if additions else 0
        pending = [
            row.operation_id for row in plan.operations if row.route == "review" and row.operation_id not in approved
        ]
        denied = [row.operation_id for row in plan.operations if row.route == "deny"]
        diagnostics = [
            Diagnostic(code=Code.DENIED, severity="warning", message="Operation denied", ref=key) for key in denied
        ]
        return {
            "appended": appended,
            "pending": list(pending),
            "denied": list(denied),
            "already_applied": not additions,
        }, diagnostics
