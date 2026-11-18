"""Separate acceptance store and fail-closed readers for new opt-in consumers.

Callers supply current source/task assertions and execution versions, never values
copied from the stored decision. Eligibility authorizes content, not Anki writes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .acceptance_policy import ScopeCell, policy_reason, supported
from .acceptance_records import (
    Acceptance as Acceptance,
)
from .acceptance_records import (
    Assertion as Assertion,
)
from .acceptance_records import (
    Binding as Binding,
)
from .acceptance_records import (
    Check as Check,
)
from .acceptance_records import (
    CheckEnvelope,
    Method,
    Stage,
    StageName,
    canonical,
    decode_json,
)
from .acceptance_records import (
    Human as Human,
)
from .acceptance_records import (
    Versions as Versions,
)
from .acceptance_records import (
    digest as digest,
)
from .claim_review import Evidence


def check_envelope(
    binding: Binding,
    versions: Versions,
    method: Method,
    assertions: tuple[Assertion, ...],
    *,
    prerequisite_output_sha256s: tuple[str, ...] = (),
    reference_constraints: tuple[Evidence, ...] = (),
) -> dict[str, object]:
    return {
        "binding": binding,
        "versions": versions,
        "method": method,
        "checked_assertions": assertions,
        "prerequisite_output_sha256s": prerequisite_output_sha256s,
        "reference_constraints": reference_constraints,
    }


def _inside(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes store")
    return path


def _stage_path(root: Path, identity: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", identity) or identity in (".", ".."):
        raise ValueError("stage ID is not a safe store key")
    return _inside(root, f"stages/{identity}.json")


def _immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError("immutable record conflict") from None


def save_acceptance(path: Path, decision: Acceptance) -> None:
    validated = Acceptance.model_validate_json(decision.model_dump_json())
    _immutable(path, canonical(validated))


def load_acceptance(path: Path) -> Acceptance:
    return Acceptance.read_json(path.read_bytes())


def save_stage_record(root: Path, stage: Stage) -> None:
    """Persist failed/unknown/deferred outcomes too, without making them reusable."""
    validated = Stage.read_json(stage.model_dump_json())
    if validated.state == "succeeded":
        stage_artifact(root, validated)
    _immutable(_stage_path(root, validated.stage_id), canonical(validated))


def save_stage(
    root: Path,
    *,
    run_id: str,
    stage_id: str,
    stage: StageName,
    binding: Binding,
    versions: Versions,
    envelope: dict[str, object],
    prerequisite_ids: tuple[str, ...] = (),
) -> Stage:
    """Publish an immutable succeeded stage; failed/unknown work uses Stage directly."""
    path = _stage_path(root, stage_id)
    artifact = f"artifacts/{stage_id}.json"
    data = canonical(envelope)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    record = Stage(
        run_id=run_id,
        stage_id=stage_id,
        stage=stage,
        binding=binding,
        versions=versions,
        prerequisite_ids=prerequisite_ids,
        state="succeeded",
        input_sha256=digest(envelope),
        output_sha256=hashlib.sha256(data).hexdigest(),
        artifact=artifact,
        call_ids=(),
        reason=None,
        started_at=now,
        finished_at=now,
    )
    outputs = []
    for identity in prerequisite_ids:
        prerequisite = load_stage(root, identity)
        stage_artifact(root, prerequisite)
        outputs.append(prerequisite.output_sha256)
    if envelope.get("prerequisite_output_sha256s", ()) not in (tuple(outputs), outputs):
        raise ValueError("prerequisite commitment mismatch")
    if path.exists():
        previous = load_stage(root, stage_id)
        if (
            previous.stage != record.stage
            or previous.binding != record.binding
            or previous.versions != record.versions
            or previous.input_sha256 != record.input_sha256
            or previous.prerequisite_ids != prerequisite_ids
            or previous.output_sha256 != record.output_sha256
        ):
            raise ValueError("immutable stage conflict")
        stage_artifact(root, previous)
        return previous
    _immutable(_inside(root, artifact), data)
    save_stage_record(root, record)
    return record


def load_stage(root: Path, stage_id: str) -> Stage:
    record = Stage.read_json(_stage_path(root, stage_id).read_bytes())
    if record.stage_id != stage_id:
        raise ValueError("stage identity mismatch")
    return record


def stage_artifact(root: Path, record: Stage) -> object:
    if record.state != "succeeded" or record.artifact is None:
        raise ValueError("stage is not reusable")
    data = _inside(root, record.artifact).read_bytes()
    if hashlib.sha256(data).hexdigest() != record.output_sha256:
        raise ValueError("stale_evidence: artifact digest mismatch")
    return decode_json(data)


def reusable_stage(
    root: Path,
    stage_id: str,
    *,
    binding: Binding,
    versions: Versions,
    input_sha256: str,
    _active: frozenset[str] = frozenset(),
) -> Stage:
    if stage_id in _active:
        raise ValueError("cyclic stage dependencies")
    stage = load_stage(root, stage_id)
    if stage.binding != binding or stage.versions != versions or stage.input_sha256 != input_sha256:
        raise ValueError("binding_mismatch")
    artifact = stage_artifact(root, stage)
    outputs = []
    for identity in stage.prerequisite_ids:
        prerequisite = load_stage(root, identity)
        if prerequisite.run_id != stage.run_id:
            raise ValueError("foreign prerequisite")
        reusable_stage(
            root,
            identity,
            binding=binding,
            versions=versions,
            input_sha256=prerequisite.input_sha256,
            _active=_active | {stage_id},
        )
        outputs.append(prerequisite.output_sha256)
    if not isinstance(artifact, dict) or artifact.get("prerequisite_output_sha256s", []) != outputs:
        raise ValueError("prerequisite commitment mismatch")
    return stage


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    origin: str
    reason: str


def assess(
    decision: Acceptance,
    *,
    root: Path,
    binding: Binding,
    versions: Versions,
    assertions: tuple[Assertion, ...],
    cells: tuple[ScopeCell, ...],
    policy_revision: str = "1",
) -> Eligibility:
    """Use at preview/reuse/export/handoff; never trust a stored accepted flag."""

    def result(reason: str) -> Eligibility:
        return Eligibility(False, decision.origin, reason)

    try:
        decision = Acceptance.model_validate_json(decision.model_dump_json())
        version_exclusions = {"policy", "policy_sha256"} if decision.origin == "human" else set()
        if (
            decision.binding != binding
            or decision.assertions != assertions
            or decision.versions.model_dump(exclude=version_exclusions)
            != versions.model_dump(exclude=version_exclusions)
        ):
            return result("binding_mismatch")
        # A human approval does not acquire or lose authority through automatic
        # policy changes. Its own check commitments remain exact and immutable.
        evidence_versions = decision.versions if decision.origin == "human" else versions
        if decision.origin == "automatic":
            reason = policy_reason(versions, policy_revision, cells)
            if reason:
                return result(reason)
            if {c.claim_class for c in cells} != {a.kind for a in assertions} or any(
                c.language != binding.language for c in cells
            ):
                return result("unsupported_policy_scope")
        if any(c.material == "curated/1" for c in cells):
            # Generic claim attestation is not a rule, rendered-card or
            # language-specific mnemonic fingerprint approval.
            return result("human_approval_required")
        if decision.state != "accepted":
            return result(decision.reason)
        stages = {identity: load_stage(root, identity) for identity in decision.stage_ids}
        for stage in stages.values():
            reusable_stage(
                root, stage.stage_id, binding=binding, versions=evidence_versions, input_sha256=stage.input_sha256
            )
        projections = []
        for stage in stages.values():
            artifact = stage_artifact(root, stage)
            if isinstance(artifact, dict) and artifact.get("format") == "latinitas-human-projection":
                projections.append(stage)
        if projections or decision.reason == "legacy_human_projection":
            from .acceptance_adapters import verify_projection

            if decision.origin != "human":
                return result("origin_mismatch")
            verify_projection(root, decision, assertions)
        elif len(cells) != len(assertions) or any(
            cell.claim_class != assertion.kind or cell.language != binding.language or not supported(cell)
            for cell, assertion in zip(cells, assertions, strict=True)
        ):
            return result("unrepresentable_claim")
        current = {a.id: a for a in assertions}
        covered: set[str] = set()
        for check in decision.checks:
            if check.method not in ("rule", "library", "human"):
                # This library-only route cannot authorize model-method checks.
                return result("unrepresentable_claim")
            if check.method == "human" and decision.origin != "human":
                return result("origin_mismatch")
            if decision.origin == "human" and check.method != "human":
                return result("missing_human_check")
            if check.version != versions.checker or check.result != "pass":
                return result("stale_evidence")
            matches = [s for s in stages.values() if s.input_sha256 == check.input_sha256]
            if len(matches) != 1:
                return result("missing_check_artifact")
            envelope = CheckEnvelope.read_json(canonical(stage_artifact(root, matches[0])))
            expected = check_envelope(
                binding,
                evidence_versions,
                check.method,
                tuple(current[i] for i in check.assertion_ids),
                prerequisite_output_sha256s=envelope.prerequisite_output_sha256s,
                reference_constraints=envelope.reference_constraints,
            )
            if digest(envelope) != check.input_sha256 or digest(expected) != check.input_sha256:
                return result("stale_evidence")
            covered.update(check.assertion_ids)
        if covered != set(current) or not covered:
            return result("incomplete_coverage")
        return Eligibility(True, decision.origin, "accepted")
    except (ValueError, OSError, TypeError, KeyError):
        return result("stale_evidence")


@dataclass(frozen=True)
class AcceptanceReader:
    """Shared preview/export/handoff boundary for explicitly adapted consumers.

    Each operation reloads and revalidates. Neither this reader nor old human
    sidecar readers silently translate one format into another.
    """

    root: Path
    binding: Binding
    versions: Versions
    assertions: tuple[Assertion, ...]
    cells: tuple[ScopeCell, ...]
    policy_revision: str = "1"

    def decide_automatic(self, *, checks: tuple[Check, ...], stage_ids: tuple[str, ...]) -> Acceptance:
        """Evaluate an analysis for this card without inheriting an aggregate decision.

        Callers supply current card assertions/context and the complete analysis
        check set. Retain every check affecting those assertions. A cross-assertion
        check cannot be trimmed; human attestations are never projected here.
        """
        required = {a.id for a in self.assertions}
        relevant = tuple(c for c in checks if required.intersection(c.assertion_ids))
        if any(not set(c.assertion_ids) <= required for c in relevant):
            return self.decide(checks=(), origin="automatic", human=None, stage_ids=())
        inputs = {c.input_sha256 for c in relevant}
        selected = []
        for identity in stage_ids:
            try:
                stage = load_stage(self.root, identity)
            except (ValueError, OSError):
                # Missing required proofs fail coverage/artifact validation below;
                # an unrelated analysis branch is not this card's prerequisite.
                continue
            if stage.input_sha256 in inputs:
                selected.append(identity)
        # assess/reusable_stage validates each selected proof's full recursive
        # prerequisite closure, even when it is absent from this top-level list.
        return self.decide(checks=relevant, origin="automatic", human=None, stage_ids=tuple(selected))

    def decide(
        self,
        *,
        checks: tuple[Check, ...],
        origin: Literal["human", "automatic", "none"],
        human: Human | None,
        stage_ids: tuple[str, ...],
    ) -> Acceptance:
        complete = (
            bool(self.assertions)
            and all(a.evidence for a in self.assertions)
            and {i for c in checks if c.result == "pass" for i in c.assertion_ids} == {a.id for a in self.assertions}
            and all(c.result == "pass" for c in checks)
            and bool(stage_ids)
            and origin != "none"
        )
        record = Acceptance.create(
            binding=self.binding,
            versions=self.versions,
            assertions=self.assertions,
            checks=checks,
            origin=origin,
            human=human,
            stage_ids=stage_ids,
            state="accepted" if complete else "withheld",
            reason="accepted" if complete else "incomplete_checks",
            created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        eligibility = assess(
            record,
            root=self.root,
            binding=self.binding,
            versions=self.versions,
            assertions=self.assertions,
            cells=self.cells,
            policy_revision=self.policy_revision,
        )
        if eligibility.eligible:
            return record
        return Acceptance.create(
            **{**record.model_dump(exclude={"decision_id"}), "state": "withheld", "reason": eligibility.reason}
        )

    def read(self, path: Path) -> tuple[Acceptance, Eligibility]:
        decision = load_acceptance(path)
        eligibility = assess(
            decision,
            root=self.root,
            binding=self.binding,
            versions=self.versions,
            assertions=self.assertions,
            cells=self.cells,
            policy_revision=self.policy_revision,
        )
        return decision, eligibility

    def preview(self, path: Path) -> dict[str, object]:
        decision, result = self.read(path)
        return {
            "decision_id": decision.decision_id,
            "origin": decision.origin,
            "state": "accepted" if result.eligible else "withheld",
            "reason": result.reason,
            "assertions": decision.model_dump(mode="json")["assertions"],
        }

    def export(self, path: Path) -> tuple[Assertion, ...]:
        decision, result = self.read(path)
        if not result.eligible:
            raise ValueError(f"withheld: {result.reason}")
        return decision.assertions

    def handoff(self, path: Path, *, note_id: str, card_key: str) -> dict[str, object] | None:
        decision, result = self.read(path)
        if result.eligible:
            return None
        return {
            "note_id": note_id,
            "card_key": card_key,
            "previous_decision_ids": [decision.decision_id],
            "reason": result.reason,
            "resolution": "manual",
            "required_action": "Reconcile previously installed content; no destination change was applied",
        }
