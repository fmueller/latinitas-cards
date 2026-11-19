"""Versioned projections of execution, acceptance and selection; never authority."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal, Self, cast

from pydantic import Field, JsonValue, model_validator

from .acceptance import load_stage, stage_artifact
from .acceptance_records import Hash, Record, Text, canonical, decode_json, digest
from .pipeline_batches import Batches
from .pipeline_consumers import (
    FormObjective,
    Rendered,
    _analysis_decision,
    _selected,
    _selection_snapshot,
    current_cards,
    load_run,
    load_source,
)
from .pipeline_profile import ProfileSnapshot
from .providers import HostedPermission, Limits

type Count = Annotated[int, Field(ge=0, strict=True)]
type StageState = Literal["pending", "running", "succeeded", "failed", "cancelled", "unknown", "deferred"]
type OutcomeClass = Literal[
    "not_generated", "deferred", "withheld", "processing_failed", "accepted_unselected", "exported"
]


class Outcome(Record):
    source_identity: Text
    candidate_id: Hash | None
    card_key: Text | None
    outcome: OutcomeClass
    reason: Text
    related_ids: tuple[Text, ...]


class OutcomeDetails(Outcome):
    note_id: Text | None
    source_links: tuple[Text, ...]
    origin: Text
    policy_id: Text
    policy_version: Text
    policy_sha256: Hash
    claim_ids: tuple[Hash, ...]
    failed_claim_ids: tuple[Hash, ...]
    alternatives: tuple[Text, ...]
    objections: tuple[Text, ...]
    evidence: tuple[Text, ...]
    limitations: tuple[Text, ...]
    example: str


class SourceUnits(Record):
    selected: Count
    attempted: Count
    deferred: Count


class PipelineReport(Record):
    format: Literal["latinitas-pipeline-report"]
    version: Literal[1]
    run_id: Text
    profile_sha256: Hash
    effective_profile: ProfileSnapshot
    source_units: SourceUnits
    stage_counts: dict[StageState, Count]
    outcomes: tuple[Outcome, ...]
    calls: tuple[Text, ...]
    decision_ids: tuple[Hash, ...]
    receipt_ids: tuple[Hash, ...]
    permission_sha256: Hash | None
    complete: bool


class ReportDetails(Record):
    counts: dict[str, Count]
    limits: dict[str, JsonValue]
    outcomes: tuple[OutcomeDetails, ...]
    processing_outcomes: tuple[dict[str, JsonValue], ...]
    reuse: dict[str, Count]


class Reporting(Record):
    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    record: PipelineReport
    details: ReportDetails

    @model_validator(mode="after")
    def matching_projection(self) -> Self:
        rows = tuple(Outcome(**o.model_dump(include=set(Outcome.model_fields))) for o in self.details.outcomes)
        if rows != self.record.outcomes:
            raise ValueError("report_detail_identity_mismatch")
        return self


def redact(value: object, secrets: tuple[str, ...] = ()) -> object:
    """Omit arbitrary exception prose; redact runtime credentials before serialization."""
    if isinstance(value, dict):
        if any(redact(str(k), secrets) != str(k) for k in value):
            raise ValueError("report_credential_collision")
        result: dict[str, object] = {}
        for key, item in value.items():
            if key in ("detail", "error") and item != "stale_evidence: artifact digest mismatch":
                result[str(key)] = "diagnostic_details_redacted"
            elif key == "analysis_errors" and isinstance(item, (list, tuple)):
                result[str(key)] = [
                    redact({**row, "reason": "analysis_processing_failed"}, secrets)
                    for row in item
                    if isinstance(row, dict)
                ]
            else:
                result[str(key)] = redact(item, secrets)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(r"(?i)\bBearer\s+[^\s\"']+", "Bearer [REDACTED]", value)
    return value


def safe_snapshot(value: object, secrets: tuple[str, ...] = ()) -> dict[str, object]:
    """Validate the final public bytes, including redaction and historical diagnostics."""
    if not isinstance(value, dict):
        raise ValueError("invalid_report")
    profile = value.get("effective_profile", {})
    if not isinstance(profile, dict) or not isinstance(providers := profile.get("providers", {}), dict):
        raise ValueError("invalid_report_profile")
    runtime = tuple(
        secret
        for endpoint in providers.values()
        if isinstance(endpoint, dict)
        if isinstance(ref := endpoint.get("credential_ref"), str) and (secret := os.environ.get(ref))
    )
    result = cast(dict[str, object], redact(value, (*secrets, *runtime)))
    if "outcome_report" in result:
        original = Reporting.read_json(canonical(value["outcome_report"]))
        public = Reporting.read_json(canonical(result["outcome_report"]))
        if (
            original.record != public.record
            or tuple(o.note_id for o in original.details.outcomes) != tuple(o.note_id for o in public.details.outcomes)
            or value.get("stage_ids") != result.get("stage_ids")
        ):
            raise ValueError("report_credential_collision")
    return result


def _reason(value: object) -> str:
    text = str(value)
    return text if re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", text) else "source_or_evidence_unavailable"


def build_report(
    root: Path, execution: dict[str, object], permission_sha256: str | None, permission: HostedPermission | None
) -> dict[str, object]:
    """Snapshot current public assessments, using the existing selection and receipt identities."""
    run = load_run(root)
    path = Path(run.profile_path)
    profile = run.profile
    source = load_source(path, profile)
    cards, skipped = current_cards(path, profile, run.objectives, root)
    selected, suppressed = _selected(cards)
    retained = {(str(s["note_id"]), str(s["card_key"])): s for s in suppressed}
    chosen = {(c.claims.note_id, c.claims.card_key) for c in selected[: profile.selection.output_limit]}
    exported = cast(dict[str, object], execution.get("file_export", {}))
    exports = [exported] if exported else []
    if not exported:
        batches = Batches(root, digest(str(root.resolve())))
        snapshot = _selection_snapshot(selected, skipped, suppressed, profile)
        for receipt_id in sorted(set(batches.index.requests.values())):
            try:
                receipt = batches.read(receipt_id)
                if receipt.selection_sha256 != snapshot:
                    continue
                if cast(dict[str, object], receipt.result["effective_profile"])["layout"] != profile.layout.model_dump(
                    mode="json"
                ):
                    continue
                exports.append(batches.result(receipt))
            except (ValueError, OSError, TypeError, KeyError):
                continue
    exported_refs = {
        (str(c["note_id"]), str(c["card_key"]))
        for export in exports
        for c in cast(list[dict[str, object]], export.get("cards", []))
    }
    exported_refs.update(
        (str(c["note_id"]), str(c["card_key"]))
        for export in exports
        for c in cast(
            list[dict[str, object]], cast(dict[str, object], export.get("reconciliation", {})).get("retained", [])
        )
    )
    stages = dict(cast(dict[str, list[dict[str, object]]], execution.get("stages", {})))
    index = cast(dict[str, str], execution.get("stage_ids", {}))
    for name in ("ingest", "analyze", "propose", "counter_check"):
        if name in stages:
            continue
        for card in cards:
            identity = index.get(name + "/" + digest((card.claims.note_id, card.claims.card_key)))
            if identity is None:
                continue
            try:
                old = load_stage(root, identity)
                if old.binding != card.binding or old.versions != card.versions:
                    continue
                previous_state, reason = old.state, old.reason
                if old.state == "succeeded":
                    stage_artifact(root, old)
            except (OSError, ValueError, TypeError):
                previous_state, reason = "failed", "invalid_stage_artifact"
            if previous_state != "succeeded":
                stages.setdefault(name, []).append(
                    {
                        "note_id": card.claims.note_id,
                        "card_key": card.claims.card_key,
                        "state": previous_state,
                        "reason": reason,
                        "execution_stage_id": identity,
                    }
                )
    if "decide" not in stages and "decide" in index:
        for card in cards:
            if card.analysis_path.exists() and card.claims.decisions[0].path.name != "unanalysed.json":
                try:
                    _analysis_decision(card)
                except (OSError, ValueError, TypeError):
                    stages.setdefault("decide", []).append(
                        {
                            "note_id": card.claims.note_id,
                            "card_key": card.claims.card_key,
                            "state": "failed",
                            "reason": "analysis_processing_failed",
                        }
                    )
    processing: list[dict[str, JsonValue]] = []
    stage_counts: Counter[str] = Counter()
    invalid = {str(r["stage_id"]) for r in cast(list[dict[str, object]], execution.get("invalid_stages", []))}
    for identity in set(index.values()):
        if identity in invalid:
            stage_counts["failed"] += 1
            continue
        try:
            stage_counts[load_stage(root, identity).state] += 1
        except (ValueError, OSError, TypeError):
            stage_counts["failed"] += 1
    for name, rows in stages.items():
        for row in rows:
            for error in cast(list[dict[str, str]], row.get("analysis_errors", [])):
                processing.append(
                    {
                        "stage": name,
                        "note_id": error["note_id"],
                        "card_key": error["card_key"],
                        "outcome": "processing_failed",
                        "reason": "analysis_processing_failed",
                    }
                )
            state = str(row.get("state", "succeeded"))
            if state in ("failed", "unknown", "cancelled", "deferred"):
                processing.append(
                    {
                        "stage": name,
                        "note_id": cast(str | None, row.get("note_id")),
                        "card_key": cast(str | None, row.get("card_key")),
                        "outcome": "processing_failed" if state in ("failed", "unknown") else "deferred",
                        "reason": _reason(row.get("reason", state)),
                        "stage_id": cast(str | None, row.get("execution_stage_id")),
                    }
                )
    for row in cast(list[dict[str, object]], execution.get("invalid_stages", [])):
        processing.append(
            {"stage_id": str(row["stage_id"]), "outcome": "processing_failed", "reason": "invalid_stage_artifact"}
        )
    inputs = next(i for i in profile.inputs if i.kind == "phrases")
    input_url = (path.parent / inputs.path).resolve().as_uri()
    entry_links = {e.identity: f"{input_url}#/entries/{i}" for i, e in enumerate(source.data.entries)}
    pattern_links = {p.key: f"{input_url}#/patterns/{i}" for i, p in enumerate(source.data.patterns)}
    outcomes: list[OutcomeDetails] = []
    decisions: set[str] = set()
    works = {digest(c.objective) for c in cards}
    deferred_work = {s["work_id"] for s in skipped if s["reason"] == "processing_limit"}
    works.update(s["work_id"] for s in skipped if "work_id" in s)
    for card in cards:
        assessment = card.claims.assess()
        ref = card.claims.note_id, card.claims.card_key
        problems = [p for p in processing if (p.get("note_id"), p.get("card_key")) == ref]
        kind: OutcomeClass = "withheld"
        reason = assessment.reason
        if assessment.eligible:
            if ref in exported_refs:
                kind, reason = "exported", "accepted_selected"
            elif ref not in chosen:
                kind, reason = "accepted_unselected", str(retained.get(ref, {}).get("reason", "output_limit"))
            else:
                kind, reason = "deferred", "export_not_run"
        elif problems:
            problem = next((p for p in problems if p["outcome"] == "processing_failed"), problems[0])
            kind, reason = cast(OutcomeClass, problem["outcome"]), str(problem["reason"])
        elif "decide" not in stages and assessment.reason == "stale_evidence":
            kind, reason = "deferred", "assessment_not_run"
        findings = [row for rows in stages.values() for row in rows if (row.get("note_id"), row.get("card_key")) == ref]
        outcomes.append(_card_outcome(card, kind, reason, entry_links, pattern_links, decisions, findings))
    for skip in skipped:
        entries = cast(list[str], decode_json(skip.get("source_entries", "[]")))
        links = [entry_links[e] for e in entries if e in entry_links]
        if skip.get("source_pattern") in pattern_links:
            links.append(pattern_links[skip["source_pattern"]])
        reason = skip["reason"]
        outcomes.append(
            OutcomeDetails(
                source_identity="+".join(entries) or "recipe:" + skip["object_key"],
                candidate_id=skip.get("candidate_key"),
                card_key=None,
                note_id=None,
                outcome="deferred" if reason == "processing_limit" else "not_generated",
                reason=_reason(reason),
                related_ids=(skip["work_id"],) if "work_id" in skip else (),
                source_links=tuple(links),
                origin="none",
                policy_id=profile.policy.id,
                policy_version=profile.policy.version,
                policy_sha256=profile.policy.sha256,
                claim_ids=(),
                failed_claim_ids=(),
                alternatives=(),
                objections=(),
                evidence=(),
                limitations=("No candidate/card count inferred for unattempted or unavailable work.",),
                example=skip["object_key"],
            )
        )
    calls = cast(list[dict[str, object]], execution.get("provider_calls", []))
    configured = Limits(**profile.processing.model_dump(exclude={"max_source_units"}))
    hosted = configured.model_dump(mode="json")
    if permission is not None:
        for key in ("max_calls", "max_tokens", "timeout_seconds"):
            hosted[key] = min(hosted[key], getattr(permission, key))
    details = ReportDetails(
        limits={
            "local": configured.model_dump(mode="json"),
            "hosted": hosted,
            "source": "effective_profile capped by supplied permission for hosted calls only",
            "calls_reserved": len(calls),
            "tokens_reserved": sum(int(cast(int, c["reserved_tokens"])) for c in calls),
        },
        counts={
            "claims": len({i for o in outcomes for i in o.claim_ids}),
            "failed_claims": len({i for o in outcomes for i in o.failed_claim_ids}),
            "candidates": len({o.candidate_id for o in outcomes if o.candidate_id is not None}),
            "cards": len({(o.note_id, o.card_key) for o in outcomes if o.card_key is not None}),
            "cards_affected_by_failed_claims": sum(bool(o.failed_claim_ids) for o in outcomes),
            "selected_cards": len(exported_refs),
        },
        outcomes=tuple(outcomes),
        processing_outcomes=tuple(processing),
        reuse=dict(Counter(str(row["reuse"]) for rows in stages.values() for row in rows if "reuse" in row)),
    )
    report = PipelineReport(
        format="latinitas-pipeline-report",
        version=1,
        run_id=str(execution["run_id"]),
        profile_sha256=digest(profile),
        effective_profile=profile,
        source_units=SourceUnits(
            selected=len(works), attempted=len(works - deferred_work), deferred=len(deferred_work)
        ),
        stage_counts=cast(dict[StageState, int], dict(stage_counts)),
        outcomes=tuple(Outcome(**o.model_dump(include=set(Outcome.model_fields))) for o in outcomes),
        calls=tuple(str(c["call_id"]) for c in calls),
        decision_ids=tuple(sorted(decisions)),
        receipt_ids=tuple(str(e["receipt_id"]) for e in exports),
        permission_sha256=permission_sha256,
        complete=bool(exports)
        and not processing
        and not deferred_work
        and not any(o.outcome == "deferred" for o in outcomes)
        and not any(stage_counts[s] for s in ("pending", "running", "failed", "cancelled", "unknown", "deferred"))
        and execution.get("status") == "completed",
    )
    return {
        **execution,
        "outcome_report": Reporting(schema_version=1, record=report, details=details).model_dump(mode="json"),
    }


def _card_outcome(
    card: Rendered,
    kind: OutcomeClass,
    reason: str,
    entry_links: dict[str, str],
    pattern_links: dict[str, str],
    decisions: set[str],
    findings: list[dict[str, object]],
) -> OutcomeDetails:
    claims = {
        # Count a shared linguistic assertion once even when cards attach
        # different corroborating evidence. Decision commitments stay unchanged.
        a.id: digest((card.binding.source_scope, card.binding.source_identity, a.model_dump(exclude={"evidence"})))
        for a in card.claims.assertions
    }
    failed: set[str] = set()
    alternatives = {s for a in card.claims.assertions for s in a.alternatives}
    objections: set[str] = set()
    evidence = {f"{e.source}: {e.reference} ({e.version})" for a in card.claims.assertions for e in a.evidence}
    related: set[str] = set()
    limitations: set[str] = set()
    for row in findings:
        answer = row.get("answer")
        if isinstance(answer, dict):
            alternatives.update(cast(list[str], answer.get("alternatives", [])))
            objections.update(cast(list[str], answer.get("objections", [])))
            limitations.add("Provider findings are transport evidence, not linguistic acceptance.")
    for destination in card.claims.decisions:
        try:
            decision, _ = destination.reader.read(destination.path)
            decisions.add(decision.decision_id)
            related.add(decision.decision_id)
            related.update(decision.stage_ids)
            for check in decision.checks:
                alternatives.update(check.alternatives)
                objections.update(check.objections)
                if check.result == "fail":
                    failed.update(claims[i] for i in check.assertion_ids if i in claims)
                if check.result != "pass":
                    limitations.add(f"{check.method}: {check.result}")
        except (OSError, ValueError, TypeError):
            limitations.add("No current readable acceptance decision.")
    entries = (
        [card.objective.entry]
        if isinstance(card.objective, FormObjective)
        else sorted({b.entry for b in card.objective.selection.bindings})
    )
    links = [entry_links[e] for e in entries]
    if not isinstance(card.objective, FormObjective):
        links.append(pattern_links[card.objective.selection.pattern])
    return OutcomeDetails(
        source_identity=card.binding.source_identity,
        candidate_id=digest(card.objective),
        card_key=card.claims.card_key,
        note_id=card.claims.note_id,
        outcome=kind,
        reason=_reason(reason),
        related_ids=tuple(sorted(related)),
        source_links=tuple(links),
        origin=card.claims.assess().origin,
        policy_id=card.versions.policy,
        policy_version=card.claims.decisions[0].reader.policy_revision,
        policy_sha256=card.versions.policy_sha256,
        claim_ids=tuple(sorted(claims.values())),
        failed_claim_ids=tuple(sorted(failed)),
        alternatives=tuple(sorted(alternatives)),
        objections=tuple(sorted(objections)),
        evidence=tuple(sorted(evidence)),
        limitations=tuple(sorted(limitations)),
        example=card.prompt,
    )


def readable(report: dict[str, object]) -> str:
    """Reason groups with one representative example; terminal controls stay escaped."""
    if "outcome_report" not in report:
        return "Legacy execution diagnostics only; no versioned outcome report. Use --json."
    view = Reporting.read_json(canonical(report["outcome_report"]))
    model, details = view.record, view.details
    groups: dict[tuple[str, ...], list[OutcomeDetails]] = {}
    for outcome in details.outcomes:
        key = (outcome.outcome, outcome.reason, outcome.origin, outcome.policy_id, outcome.policy_version)
        groups.setdefault(key, []).append(outcome)
    lines = [
        f"Pipeline {model.run_id} (report v{model.version})",
        "Yield: " + ", ".join(f"{n} {k}" for k, n in details.counts.items()),
        "Source work (recipe/objective units): " + json.dumps(model.source_units.model_dump()),
        "Stages: " + json.dumps(model.stage_counts),
        "Limits: " + json.dumps(details.limits),
        "Selection: " + json.dumps(model.effective_profile.selection.model_dump()),
        "Reuse (analysis/provider operations this invocation): " + json.dumps(details.reuse),
        "Validated receipt IDs: " + ", ".join(model.receipt_ids),
        str(report.get("coverage_limit", "")),
    ]
    for (kind, reason, origin, policy, version), examples in sorted(groups.items()):
        example = examples[0]
        lines.extend(
            (
                f"{len(examples)} {kind}: {reason}",
                "  Example: " + json.dumps(example.example, ensure_ascii=True),
                "  Sources: " + ", ".join(example.source_links),
                f"  Origin: {origin}; policy: {policy}/{version}",
                "  Alternatives: " + json.dumps(sorted({v for o in examples for v in o.alternatives})),
                "  Objections: " + json.dumps(sorted({v for o in examples for v in o.objections})),
                "  Evidence: " + json.dumps(sorted({v for o in examples for v in o.evidence})),
                "  Limitations: " + json.dumps(sorted({v for o in examples for v in o.limitations})),
            )
        )
    lines.extend(
        (
            "Processing problems (not linguistic rejections): " + json.dumps(details.processing_outcomes),
            "File export only; not applied to Anki. No mandatory per-card review.",
        )
    )
    controls = {i: f"\\u{i:04x}" for i in (*range(32), *range(127, 160))}
    return "\n".join(line.translate(controls) for line in lines)
