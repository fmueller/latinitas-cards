"""Development-only evidence classification; never an acceptance store."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .acceptance import Acceptance, assess_evidence
from .acceptance_policy import SUPPORT_REGISTRY, ScopeCell, implementation_digest
from .acceptance_records import Assertion, Binding, Check, Hash, Record, Text, Versions, canonical, digest
from .pipeline_consumers import Analysis, Rendered, current_cards, load_run


class Obligation(Record):
    assertion_id: Text
    cell: ScopeCell
    required: tuple[Literal["source_evidence", "blind_model", "semantic_comparison"], ...]
    checks: tuple[Check, ...]
    result: Literal["pass", "withhold", "unscorable"]
    reasons: tuple[Text, ...]


class Classification(Record):
    format: Literal["latinitas-research-classification"] = "latinitas-research-classification"
    version: Literal[1] = 1
    implementation_sha256: Hash
    support_registry_sha256: Hash
    criteria: Literal["development-complete-evidence/1"]
    binding: Binding
    versions: Versions
    cells: tuple[ScopeCell, ...]
    assertions: tuple[Assertion, ...]
    checks: tuple[Check, ...]
    stage_ids: tuple[Text, ...]
    obligations: tuple[Obligation, ...]
    failures: tuple[Text, ...]
    classification: Literal["would_accept", "would_withhold", "unscorable"]
    reason: Text


def _classify(root: Path, card: Rendered) -> Classification:
    reader = card.claims.decisions[0].reader
    checks: tuple[Check, ...] = ()
    stages: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    reason = "missing_model_evidence"
    try:
        analysis = Analysis.read_json(card.analysis_path.read_bytes())
        if analysis.binding != card.binding or analysis.versions != card.versions:
            raise ValueError("stale analysis context")
        inventory = {a.id: a for a in analysis.assertions}
        if any(inventory.get(a.id) != a for a in reader.assertions):
            raise ValueError("analysis assertion mismatch")
        required = {a.id for a in reader.assertions}
        checks = tuple(c for c in analysis.checks if required.intersection(c.assertion_ids))
        if any(not set(c.assertion_ids) <= required for c in checks):
            raise ValueError("cross-card check cannot be projected")
        stages = analysis.stage_ids
        model_checks, model_stages = reader.model_checks()
        checks = (*checks, *model_checks)
        stages = tuple(dict.fromkeys((*stages, *model_stages)))
        # A withheld intermediate carries checks through the same validator;
        # no accepted decision is created, saved or returned by this API.
        bundle = Acceptance.create(
            binding=reader.binding,
            versions=reader.versions,
            assertions=reader.assertions,
            checks=checks,
            origin="automatic",
            human=None,
            stage_ids=stages,
            state="withheld",
            reason="research_only",
            created_at="2000-01-01T00:00:00Z",
        )
        assessment = assess_evidence(
            bundle,
            root=root,
            binding=reader.binding,
            versions=reader.versions,
            assertions=reader.assertions,
            cells=reader.cells,
        )
        reason = assessment.reason
        if reason == "accepted" and not model_checks:
            reason = "missing_model_evidence"
        if reason in ("stale_evidence", "binding_mismatch", "origin_mismatch", "missing_check_artifact"):
            failures = (reason,)
        if any("processing_unavailable" in c.objections for c in checks):
            failures = (*failures, "processing_unavailable")
    except (ValueError, OSError, TypeError, KeyError) as error:
        failures = (f"{type(error).__name__}: {error}",)
        reason = "processing_failure"
    obligations = []
    for assertion, cell in zip(reader.assertions, reader.cells, strict=True):
        own = tuple(c for c in checks if assertion.id in c.assertion_ids)
        reasons = tuple(
            dict.fromkeys(
                (
                    *(f"{c.id}:{c.result}" for c in own),
                    *(r for c in own for r in (*c.alternatives, *c.objections)),
                    *(f"assertion_alternative:{a}" for a in assertion.alternatives),
                    *(("missing_source_evidence",) if not assertion.evidence else ()),
                    *(("missing_blind_model",) if not any(c.method == "blind_model" for c in own) else ()),
                    *(
                        ("missing_semantic_comparison",)
                        if not any(c.method == "semantic_comparison" for c in own)
                        else ()
                    ),
                    *failures,
                )
            )
        )
        passed = (
            bool(assertion.evidence)
            and not assertion.alternatives
            and {"blind_model", "semantic_comparison"} <= {c.method for c in own}
            and all(c.result == "pass" and not c.alternatives and not c.objections for c in own)
        )
        obligations.append(
            Obligation(
                assertion_id=assertion.id,
                cell=cell,
                required=("source_evidence", "blind_model", "semantic_comparison"),
                checks=own,
                result="unscorable" if failures else "pass" if passed else "withhold",
                reasons=reasons,
            )
        )
    return Classification(
        implementation_sha256=implementation_digest(),
        support_registry_sha256=digest(SUPPORT_REGISTRY),
        criteria="development-complete-evidence/1",
        binding=reader.binding,
        versions=reader.versions,
        cells=tuple(dict.fromkeys(reader.cells)),
        assertions=reader.assertions,
        checks=checks,
        stage_ids=stages,
        obligations=tuple(obligations),
        failures=failures,
        classification="unscorable"
        if failures
        else "would_accept"
        if reason == "accepted" and obligations and all(o.result == "pass" for o in obligations)
        else "would_withhold",
        reason=reason,
    )


def classify(root: Path) -> tuple[Classification, ...]:
    """Reconstruct every current card, including every unsupported sibling.

    Invalid run/source inventories raise rather than invent denominator slots.
    Once the inventory is known, failed evidence remains an unscorable record.
    This function does not dispatch models, modify policies or write any output.
    """
    run = load_run(root)
    cards, skipped = current_cards(Path(run.profile_path), run.profile, run.objectives, root, resolve_decisions=False)
    failures = [s for s in skipped if s["reason"] not in ("processing_limit", "variant_limit", "duplicate_card_ref")]
    if failures:
        raise ValueError(f"research inventory reconstruction failed: {canonical(failures).decode()}")
    return tuple(_classify(root, card) for card in cards)


def save(path: Path, record: Classification) -> None:
    """Save a research record, exclusively; integrity is checked again on read."""
    validated = Classification.read_json(record.model_dump_json())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(validated))


def read(root: Path, path: Path) -> Classification:
    """Never trust a stored classification: reconstruct it from current proofs."""
    stored = Classification.read_json(path.read_bytes())
    matches = [record for record in classify(root) if record.binding == stored.binding]
    if len(matches) != 1 or matches[0] != stored or stored.classification == "unscorable":
        raise ValueError("stale or unscorable research classification")
    return stored
