"""Immutable annotation plans and receipts; no sidecar writes or claim decisions."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, JsonValue, ValidationError, model_validator

from .annotation_sidecar import (
    Anchor,
    Annotation,
    Created,
    OptionalFields,
    Producer,
    Retraction,
    annotation_id,
    read_sidecar,
    validate_anchor,
)
from .annotation_state import annotation_states, dependents
from .atomic_io import write_artifact
from .canonical import canonical_json, digest, digest_bytes
from .claim_sidecar import SidecarV2, UnsupportedVersion
from .cli_contract import Code, ContractError, Diagnostic
from .corpus_claims import load_corpus_decisions
from .corpus_io import read_corpus
from .corpus_model import CorpusHeader, CorpusModel, CorpusPassage, Sha256, VersionOne
from .corpus_validation import validate_records
from .policy import Layer, Policy, load_policy, policy_digest, route


class Replacement(OptionalFields):
    layer: Layer
    anchor: Anchor
    value: JsonValue
    alternatives: list[JsonValue] | None = None
    producer: Producer
    confidence: Annotated[int, Field(ge=0, le=1000)] | None = None
    created: Created | None = None


class Add(Replacement):
    operation: Literal["add"]


class Supersede(Replacement):
    operation: Literal["supersede"]
    target: Sha256


class Retract(OptionalFields):
    operation: Literal["retract"]
    target: Sha256
    producer: Producer
    created: Created | None = None


type PatchOperation = Annotated[Add | Supersede | Retract, Field(discriminator="operation")]


class AnnotationPatch(VersionOne):
    format: Literal["latinitas-annotation-patch"]
    version: Literal[1]
    operations: list[PatchOperation]

    @model_validator(mode="before")
    @classmethod
    def canonical_values(cls, value: Any) -> Any:
        canonical_json(value)
        return value


type InvalidReason = Literal[
    "unmatched",
    "stale",
    "orphaned",
    "input_hash",
    "invalid_value",
    "span_out_of_range",
    "unknown_segmentation",
    "unknown_target",
    "target_inactive",
    "duplicate_operation",
]


class Safety(CorpusModel):
    decision_status: Literal["accepted", "withheld"] | None
    dependents: list[Sha256]
    conflicts: list[Sha256]


class PlannedOperation(CorpusModel):
    operation_id: Sha256
    index: Annotated[int, Field(ge=0)]
    route: Literal["auto", "review", "deny", "invalid"]
    reasons: list[str]
    safety: Safety
    records: list[dict[str, JsonValue]]


class PlanBase(CorpusModel):
    corpus_header: Sha256
    sidecar: Sha256 | None
    policy: Sha256 | None
    patch: Sha256


class AnnotationPlan(VersionOne):
    format: Literal["latinitas-annotation-plan"]
    version: Literal[1]
    plan_id: Sha256
    base: PlanBase
    receipt_actor: Literal["human", "any"]
    patch: AnnotationPatch
    operations: list[PlannedOperation]

    @model_validator(mode="after")
    def identity(self) -> Self:
        raw = self.model_dump(mode="json")
        if self.plan_id != digest({k: v for k, v in raw.items() if k != "plan_id"}):
            raise ValueError("plan_id does not recompute")
        if self.base.patch != digest(raw["patch"]) or len(self.operations) != len(self.patch.operations):
            raise ValueError("patch binding does not recompute")
        for index, (operation, planned) in enumerate(zip(self.patch.operations, self.operations, strict=True)):
            if planned.index != index or planned.operation_id != digest(operation.model_dump(mode="json")):
                raise ValueError("operation binding does not recompute")
        return self


class AnnotationReceipt(VersionOne):
    format: Literal["latinitas-annotation-receipt"]
    version: Literal[1]
    plan_id: Sha256
    operations: list[Sha256]
    actor: str = Field(pattern=r"^(human|agent):[^\s:]+$")
    review: str = Field(min_length=1, pattern=r"\S")


def load_document(path: Path, model: type[AnnotationPatch] | type[AnnotationPlan]) -> AnnotationPatch | AnnotationPlan:
    """Gate the envelope before strict model validation, without normalizing input."""
    invalid = Code.INVALID_PLAN if model is AnnotationPlan else Code.INVALID_PATCH
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        raw = json.loads(data, object_pairs_hook=unique)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ContractError(invalid if model is AnnotationPlan else Code.INVALID_JSON, str(error)) from error
    if model is AnnotationPatch and isinstance(raw, dict):
        if raw.get("format") != "latinitas-annotation-patch":
            raise ContractError(Code.UNSUPPORTED_FORMAT, "Expected latinitas-annotation-patch")
        if type(raw.get("version")) is int and raw["version"] != 1:
            raise ContractError(Code.UNSUPPORTED_VERSION, "Expected patch version 1")
    try:
        canonical_json(raw)
        return model.model_validate(raw)
    except (ValueError, RecursionError) as error:
        raise ContractError(invalid, str(error), path=str(path)) from error


def read_decisions(corpus: Path) -> SidecarV2:
    """Keep legacy claims exits unchanged; translate only at the contract boundary."""
    path = Path(f"{corpus}.claims.json")
    try:
        return load_corpus_decisions(path)
    except UnsupportedVersion as error:
        raise ContractError(Code.UNSUPPORTED_VERSION, str(error), path=str(path)) from error
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error
    except (ValueError, UnicodeError) as error:
        raise ContractError(Code.INVALID_SIDECAR, str(error), path=str(path)) from error


def proposed_record(operation: PatchOperation, targets: Mapping[str, Annotation]) -> dict[str, Any] | None:
    raw = operation.model_dump(mode="json")
    raw.pop("operation")
    target = raw.pop("target", None)
    if isinstance(operation, Retract):
        if target not in targets:
            return None
        original = targets[target]
        raw.update(
            record="retraction",
            retracts=target,
            layer=original.layer,
            anchor=original.anchor.model_dump(mode="json"),
            input_hash=original.input_hash,
        )
    else:
        raw.update(record="annotation", input_hash=operation.anchor.passage_hash)
        if isinstance(operation, Supersede):
            raw["supersedes"] = target
    raw["annotation_id"] = annotation_id(raw)
    return raw


def checked_record(raw: dict[str, Any]) -> Annotation | Retraction:
    model = Retraction if raw["record"] == "retraction" else Annotation
    return model.model_validate({**raw, "plan": None, "created": raw.get("created", "1970-01-01T00:00:00Z")})


def evaluate_operations(
    patch: AnnotationPatch,
    passages: Mapping[str, CorpusPassage],
    sidecar: Sequence[Annotation | Retraction],
    decisions: SidecarV2,
    policy: Policy | None,
) -> list[PlannedOperation]:
    """Reusable plan/apply evaluator: current state plus the whole proposed patch.

    Proposed records contribute facts before any append. ID deduplication makes
    those facts stable when an expected proposal is later present in a retry suffix.
    """
    hashes = {key: row.passage_hash for key, row in passages.items()}
    states = annotation_states(hashes, sidecar)
    targets = {row.annotation_id: row for row in sidecar if isinstance(row, Annotation)}
    raws = [proposed_record(operation, targets) for operation in patch.operations]
    proposals: list[Annotation | Retraction | None] = []
    for raw in raws:
        try:
            proposals.append(checked_record(raw) if raw is not None else None)
        except ValueError:
            proposals.append(None)
    current = [
        row
        for row in sidecar
        if states[row.annotation_id].state == "current"
        and not states[row.annotation_id].superseded
        and not states[row.annotation_id].retracted
    ]
    effective = {row.annotation_id: row for row in [*current, *(row for row in proposals if row is not None)]}
    segmentations = {
        key: row for key, row in effective.items() if isinstance(row, Annotation) and row.layer == "segmentation"
    }
    seen: set[str] = set()
    result: list[PlannedOperation] = []
    for index, (operation, raw, record) in enumerate(zip(patch.operations, raws, proposals, strict=True)):
        operation_id = digest(operation.model_dump(mode="json"))
        reasons: list[str] = []
        if operation_id in seen:
            reasons.append("duplicate_operation")
        seen.add(operation_id)
        target = targets.get(operation.target) if isinstance(operation, Supersede | Retract) else None
        if isinstance(operation, Supersede | Retract):
            if target is None:
                reasons.append("unknown_target")
            else:
                state = states[target.annotation_id]
                if state.state != "current":
                    reasons.append(state.state)
                if state.superseded or state.retracted:
                    reasons.append("target_inactive")
        if raw is not None and record is None:
            reasons.append("invalid_value")
        if record is not None:
            passage = passages.get(record.anchor.passage_id)
            if passage is None:
                reasons.append("unmatched")
            elif passage.passage_hash != record.anchor.passage_hash:
                reasons.append("stale")
            else:
                segmentation = segmentations.get(record.anchor.segmentation_id or "")
                if record.anchor.segmentation_id is not None and segmentation is None:
                    old = states.get(record.anchor.segmentation_id)
                    reasons.append(
                        "orphaned" if old is not None and (old.superseded or old.retracted) else "unknown_segmentation"
                    )
                try:
                    validate_anchor(record, passage, segmentation)
                except ValueError:
                    reasons.append("span_out_of_range")
        safety = Safety(decision_status=None, dependents=[], conflicts=[])
        if target is not None:
            statuses = {
                entry.decision.status
                for entry in decisions.decisions
                if entry.binding.annotation_id == target.annotation_id
            }
            if "withheld" in statuses:
                safety.decision_status = "withheld"
            elif statuses:
                safety.decision_status = "accepted"
            safety.dependents = sorted(
                {row.annotation_id for row in dependents(list(effective.values()), target.annotation_id)}
            )
        if isinstance(record, Annotation):
            safety.conflicts = sorted(
                {
                    row.annotation_id
                    for row in effective.values()
                    if isinstance(row, Annotation)
                    and row.annotation_id != record.annotation_id
                    and row.layer == record.layer
                    and row.anchor == record.anchor
                    and row.producer == record.producer
                    and row.value != record.value
                }
            )
        if record is not None:
            layer = record.layer
        elif isinstance(operation, Replacement):
            layer = operation.layer
        else:
            layer = "segmentation"
        routed = route(
            operation.operation,
            layer,
            operation.producer.kind,
            operation.producer.name,
            bool(safety.decision_status or safety.dependents),
            bool(safety.conflicts),
            policy,
        )
        result.append(
            PlannedOperation(
                operation_id=operation_id,
                index=index,
                route="invalid" if reasons else routed.action,
                reasons=list(dict.fromkeys(reasons)) if reasons else list(routed.reasons),
                safety=safety,
                records=[raw] if raw is not None else [],
            )
        )
    return result


def create_plan(
    patch_path: Path, corpus: Path, sidecar: Path, policy_path: Path | None, output: Path
) -> tuple[JsonValue, list[Diagnostic]]:
    patch = load_document(patch_path, AnnotationPatch)
    assert isinstance(patch, AnnotationPatch)
    policy = load_policy(policy_path) if policy_path is not None else None
    rows = list(read_corpus(corpus))
    _, findings = validate_records(iter(rows), path=str(corpus))
    if findings:
        return None, [Diagnostic(code=Code.INVALID_CORPUS, message=f"Invalid corpus: {findings[0].code}"), *findings]
    header = next(row for row in rows if isinstance(row, CorpusHeader))
    passages = {row.passage_id: row for row in rows if isinstance(row, CorpusPassage)}
    existing: list[Annotation | Retraction] = []
    sidecar_digest = None
    if sidecar.exists():
        try:
            existing = [row for row in read_sidecar(sidecar) if isinstance(row, Annotation | Retraction)]
            sidecar_digest = digest_bytes(sidecar.read_bytes())
        except ContractError as error:
            if error.diagnostic.code in {Code.INVALID_JSON, Code.MISSING_HEADER}:
                raise ContractError(Code.INVALID_SIDECAR, str(error), path=str(sidecar)) from error
            raise
    operations = evaluate_operations(patch, passages, existing, read_decisions(corpus), policy)
    raw = {
        "format": "latinitas-annotation-plan",
        "version": 1,
        "base": {
            "corpus_header": digest(header.model_dump(mode="json")),
            "sidecar": sidecar_digest,
            "policy": policy_digest(policy_path),
            "patch": digest(patch.model_dump(mode="json")),
        },
        "receipt_actor": policy.receipt_actor if policy is not None else "human",
        "patch": patch.model_dump(mode="json"),
        "operations": [row.model_dump(mode="json") for row in operations],
    }
    document = AnnotationPlan.model_validate({**raw, "plan_id": digest(raw)})
    protected = [patch_path, corpus, sidecar, Path(f"{corpus}.claims.json"), corpus.with_suffix(".manifest.json")]
    if policy_path is not None:
        protected.append(policy_path)
    created = write_document(output, document.model_dump(mode="json"), protected)
    diagnostics = [
        Diagnostic(code=Code(reason), message=f"Operation {row.index}: {reason}", ref=row.operation_id)
        for row in operations
        if row.route == "invalid"
        for reason in row.reasons
    ]
    return document.model_dump(mode="json"), diagnostics if created else []


def write_document(output: Path, document: dict[str, Any], protected: Sequence[Path]) -> bool:
    reserved = [output, output.resolve()]
    if any(path.suffix == ".lock" or path.name.endswith(".manifest.json") for path in reserved):
        raise ContractError(Code.ALIAS, "Output must not be a lock or manifest", path=str(output))
    # Plans deliberately contain no source paths. Recognize existing protected
    # documents here as well, so approval cannot target renamed/hardlinked inputs.
    try:
        previous = output.read_bytes()
    except FileNotFoundError:
        previous = b""
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(output)) from error
    try:
        try:
            header = json.loads(previous)
        except ValueError:
            header = json.loads(previous.split(b"\n", 1)[0])
    except (ValueError, UnicodeError, RecursionError):
        header = None
    if isinstance(header, dict) and (
        header.get("format")
        in {
            "latinitas-corpus",
            "latinitas-annotations",
            "latinitas-corpus-manifest",
            "latinitas-policy",
            "latinitas-annotation-patch",
        }
        or "decisions" in header
    ):
        raise ContractError(Code.ALIAS, "Output is a protected document", path=str(output))
    own_lock = Path(f"{output.resolve()}.lock")
    aliases = list(protected)
    for directory in {path.parent for path in [*protected, *reserved]}:
        aliases.extend(directory.glob("*.manifest.json"))
        aliases.extend(path for path in directory.glob("*.lock") if path.resolve() != own_lock)
    return write_artifact(output, (canonical_json(document) + "\n").encode(), aliases)


def approve_plan(
    plan_path: Path, operations: Sequence[str], actor: str, review: str, output: Path
) -> tuple[JsonValue, list[Diagnostic]]:
    plan = load_document(plan_path, AnnotationPlan)
    assert isinstance(plan, AnnotationPlan)
    selected = sorted(set(operations))
    by_id = {row.operation_id: row for row in plan.operations}
    diagnostics: list[Diagnostic] = []
    for operation_id in selected:
        row = by_id.get(operation_id)
        if row is None:
            diagnostics.append(Diagnostic(code=Code.UNKNOWN_OPERATION, message="Unknown operation", ref=operation_id))
        elif row.route == "invalid":
            diagnostics.append(Diagnostic(code=Code.INVALID_OPERATION, message="Invalid operation", ref=operation_id))
        elif row.route == "deny":
            diagnostics.append(Diagnostic(code=Code.DENIED, message="Operation denied", ref=operation_id))
    if actor.startswith("agent:") and plan.receipt_actor == "human":
        diagnostics.append(Diagnostic(code=Code.ACTOR_REFUSED, message="A declared human actor is required"))
    try:
        receipt = AnnotationReceipt(
            format="latinitas-annotation-receipt",
            version=1,
            plan_id=plan.plan_id,
            operations=selected,
            actor=actor,
            review=review,
        )
    except ValidationError as error:
        diagnostics.append(Diagnostic(code=Code.INVALID_INPUT, message=str(error)))
        return None, diagnostics
    if diagnostics:
        return None, diagnostics
    write_document(output, receipt.model_dump(mode="json"), [plan_path])
    return receipt.model_dump(mode="json"), []
