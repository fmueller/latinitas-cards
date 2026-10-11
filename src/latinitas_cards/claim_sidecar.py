"""Versioned offline attestations, independent of identity and destination approval."""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .claim_review import Claim, ClaimAssessment, Evidence, ReviewDecision, assess_claim, review_claim


@dataclass(frozen=True)
class BoundClaim:
    identity: str
    feature: str
    claim: Claim
    decision: ReviewDecision | None = None
    authorable: bool = True


class DecisionItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["accepted", "withheld"]
    reviewer: str
    reason: str
    value: str | None = None
    evidence: tuple[Evidence, ...] | None = None
    selected_sense: str | None = None


class StoredDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_fingerprint: str
    decision: ReviewDecision
    value: str | None = None
    evidence: tuple[Evidence, ...] | None = None
    selected_sense: str | None = None


class Sidecar(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    decisions: tuple[StoredDecision, ...] = ()


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    item: int | None = None


def sidecar_path(source: Path) -> Path:
    return Path(f"{source}.claims.json")


def load_sidecar(path: Path) -> Sidecar:
    return Sidecar.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else Sidecar()


def authored_claim(
    base: Claim, value: str | None, evidence: tuple[Evidence, ...] | None, selected_sense: str | None = None
) -> Claim:
    if value is None:
        if evidence is not None or selected_sense is not None:
            raise ValueError("authored evidence requires a value")
        return base
    if not evidence:
        raise ValueError("authored values require their own evidence")
    if base.kind == "translation" and base.profile.language_tag.lower().split("-")[0] != "de":
        raise ValueError("Reviewed form translations currently support German user language only")
    return replace(
        base,
        value=value,
        evidence=evidence,
        analyzer="reviewer-authored/v1",
        supported=True,
        selected_sense=selected_sense if selected_sense is not None else base.selected_sense,
    )


def resolve_claims(
    claims: Sequence[BoundClaim], sidecar: Sidecar
) -> tuple[tuple[BoundClaim, ...], tuple[Diagnostic, ...]]:
    bases = {b.claim.fingerprint: b for b in claims}
    stored: dict[str, StoredDecision] = {}
    seen: set[str] = set()
    diagnostics: list[Diagnostic] = []
    for index, entry in enumerate(sidecar.decisions):
        base = bases.get(entry.base_fingerprint)
        if base is None:
            diagnostics.append(Diagnostic("stale", "Stale or unmatched decision ignored", index))
            continue
        if entry.base_fingerprint in seen:
            diagnostics.append(Diagnostic("conflict", "Duplicate decisions ignored", index))
            stored.pop(entry.base_fingerprint, None)
            continue
        seen.add(entry.base_fingerprint)
        claim = authored_claim(base.claim, entry.value, entry.evidence, entry.selected_sense)
        if entry.decision.claim_fingerprint != claim.fingerprint or (
            entry.decision.status == "accepted" and not claim.supported
        ):
            diagnostics.append(Diagnostic("stale", "Invalid claim binding ignored", index))
            continue
        stored[entry.base_fingerprint] = entry
    result: list[BoundClaim] = []
    for base in claims:
        saved = stored.get(base.claim.fingerprint)
        if saved is None:
            result.append(base)
        else:
            result.append(
                replace(
                    base,
                    claim=authored_claim(base.claim, saved.value, saved.evidence, saved.selected_sense),
                    decision=saved.decision,
                )
            )
    return tuple(result), tuple(diagnostics)


def assessments(claims: Sequence[BoundClaim]) -> tuple[ClaimAssessment, ...]:
    return tuple(assess_claim(b.claim, b.decision) for b in claims)


def claim_report(bound: BoundClaim) -> dict[str, object]:
    assessment = assess_claim(bound.claim, bound.decision)
    return {
        "identity": bound.identity,
        "candidate_id": bound.claim.candidate_id,
        "feature": bound.feature,
        "kind": bound.claim.kind,
        "selected_sense": bound.claim.selected_sense,
        "language": bound.claim.profile.language_tag if bound.claim.kind == "translation" else None,
        "form": bound.claim.candidate_text,
        "value": bound.claim.value if bound.claim.supported else None,
        "alternatives": list(bound.claim.alternatives),
        "evidence": [asdict(e) for e in bound.claim.evidence],
        "fingerprint": bound.claim.fingerprint,
        "status": assessment.status,
        "review_status": bound.decision.status
        if bound.decision is not None and bound.decision.claim_fingerprint == bound.claim.fingerprint
        else "open",
        "reason": assessment.reason,
    }


def record_batch(path: Path, claims: Sequence[BoundClaim], items: Sequence[object]) -> tuple[Diagnostic, ...]:
    """Validate every item before one atomic replacement; serialize concurrent writers."""
    with Path(f"{path}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sidecar = load_sidecar(path)
        current, diagnostics = resolve_claims(claims, sidecar)
        if diagnostics:
            return diagnostics
        lookup = {b.claim.fingerprint: (base, b) for base, b in zip(claims, current, strict=True)}
        entries = {e.base_fingerprint: e for e in sidecar.decisions}
        item_indexes: dict[str, int] = {}
        errors: list[Diagnostic] = []
        for index, raw in enumerate(items):
            try:
                item = DecisionItem.model_validate(raw)
                binding = lookup.get(item.fingerprint)
                if binding is None:
                    errors.append(Diagnostic("stale", "Fingerprint is not a current claim", index))
                    continue
                base, bound = binding
                if item.value is not None and not base.authorable:
                    raise ValueError("unresolved source forms require separate source correction")
                claim = authored_claim(bound.claim, item.value, item.evidence, item.selected_sense)
                # Role labels are canonical slots, never free-text identity changes.
                if (
                    base.claim.profile_version == "principal-parts/v1"
                    and claim.kind == "label"
                    and claim.value != base.feature
                ):
                    raise ValueError("role label must match the confirmed canonical role")
                decision = review_claim(claim, status=item.status, reviewer=item.reviewer, reason=item.reason)
                previous = entries.get(base.claim.fingerprint)
                entry = StoredDecision(
                    base_fingerprint=base.claim.fingerprint,
                    decision=decision,
                    value=item.value if item.value is not None else (previous.value if previous else None),
                    evidence=item.evidence if item.value is not None else (previous.evidence if previous else None),
                    selected_sense=claim.selected_sense,
                )
                if (
                    previous is not None
                    and previous != entry
                    and (
                        previous.decision.claim_fingerprint == decision.claim_fingerprint
                        or base.claim.fingerprint in item_indexes
                    )
                ):
                    errors.append(Diagnostic("conflict", "Conflicting decision for the same claim", index))
                else:
                    entries[base.claim.fingerprint] = entry
                    item_indexes[base.claim.fingerprint] = index
            except ValueError as error:
                errors.append(Diagnostic("invalid", str(error), index))
        proposed, _ = resolve_claims(claims, Sidecar(decisions=tuple(entries.values())))
        accepted: dict[tuple[str, str], set[str]] = {}
        for bound in proposed:
            if assess_claim(bound.claim, bound.decision).status == "accepted":
                accepted.setdefault((bound.claim.candidate_id, bound.claim.kind), set()).add(bound.claim.value)
        for base, bound in zip(claims, proposed, strict=True):
            if len(accepted.get((bound.claim.candidate_id, bound.claim.kind), set())) > 1:
                errors.append(
                    Diagnostic(
                        "conflict", "Conflicting accepted alternatives", item_indexes.get(base.claim.fingerprint)
                    )
                )
        if errors:
            return tuple(errors)
        payload = Sidecar(decisions=tuple(entries.values())).model_dump_json(indent=2) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return ()
