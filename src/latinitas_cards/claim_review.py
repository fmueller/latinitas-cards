"""Offline, precision-first linguistic claim review; no destination authorization.

There is deliberately no automatic acceptance path until a separately reviewed,
domain-relevant calibration establishes a policy for an analyzer/category.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Literal

from .profile import DeckProfile
from .source_extraction import SourceExtraction

type ClaimKind = Literal["label", "segmentation", "explanation"]
type ReviewStatus = Literal["accepted", "withheld"]


@dataclass(frozen=True, slots=True)
class Evidence:
    source: str
    reference: str
    version: str
    text: str

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.source, self.reference, self.version, self.text)):
            raise ValueError("evidence requires source, reference, version and text")


@dataclass(frozen=True, slots=True)
class Claim:
    candidate_id: str
    candidate_text: str
    kind: ClaimKind
    value: str
    analyzer: str
    evidence: tuple[Evidence, ...]
    profile: DeckProfile
    profile_version: str
    alternatives: tuple[str, ...] = ()
    supported: bool = True
    extraction: SourceExtraction | None = None

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.candidate_id, self.value, self.analyzer, self.profile_version)):
            raise ValueError("claim requires candidate, value, analyzer and profile version")
        if self.kind not in {"label", "segmentation", "explanation"} or not self.evidence:
            raise ValueError("claim requires a known kind and provenance")

    @property
    def fingerprint(self) -> str:
        """Bind claim data, extraction and linguistic profile, not presentation.

        Destination authorization independently binds the rendered payload.
        """
        data = {
            "candidate_id": self.candidate_id,
            "candidate_text": self.candidate_text,
            "kind": self.kind,
            "value": self.value,
            "analyzer": self.analyzer,
            "evidence": [asdict(item) for item in self.evidence],
            "profile": self.profile.model_dump(mode="json", exclude={"morphology"}),
            "profile_version": self.profile_version,
            "alternatives": self.alternatives,
            "supported": self.supported,
            "extraction": asdict(self.extraction) if self.extraction is not None else None,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    claim_fingerprint: str
    status: ReviewStatus
    reviewer: str
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "withheld"}:
            raise ValueError("unknown review status")
        if not all(value.strip() for value in (self.claim_fingerprint, self.reviewer, self.reason)):
            raise ValueError("review requires a claim binding, reviewer and reason")


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    claim: Claim
    status: ReviewStatus
    reason: str
    decision: ReviewDecision | None = None


def review_claim(claim: Claim, *, status: ReviewStatus, reviewer: str, reason: str) -> ReviewDecision:
    """Explicitly adjudicate this proposal, not its alternatives or sibling claims.

    An unsupported proposal needs new supporting evidence and renewed review, not
    an override. Choosing an alternative means constructing/reviewing that claim.
    """
    if status == "accepted" and not claim.supported:
        raise ValueError("cannot accept an unsupported claim")
    return ReviewDecision(claim.fingerprint, status, reviewer, reason)


def assess_claim(claim: Claim, decision: ReviewDecision | None = None) -> ClaimAssessment:
    if decision is not None and decision.claim_fingerprint != claim.fingerprint:
        return ClaimAssessment(claim, "withheld", "Stale or different claim review; renewed review required", decision)
    if not claim.supported:
        return ClaimAssessment(claim, "withheld", "Unsupported claim; supporting evidence required", decision)
    if decision is not None:
        return ClaimAssessment(claim, decision.status, decision.reason, decision)
    return ClaimAssessment(claim, "withheld", "No applicable calibration; explicit claim review required")


def asserted_claims(assessments: Iterable[ClaimAssessment]) -> tuple[Claim, ...]:
    """Only freshly validated individual acceptances may become asserted knowledge."""
    return tuple(
        item.claim
        for item in assessments
        if item.status == "accepted" and assess_claim(item.claim, item.decision).status == "accepted"
    )


type ExpectedJudgment = Literal["correct", "rejected", "ambiguous", "unsupported"]


@dataclass(slots=True)
class EvaluationCounts:
    total: int = 0
    accepted_correct: int = 0
    accepted_errors: int = 0
    withheld: Counter[ExpectedJudgment] = field(default_factory=Counter)

    @property
    def precision(self) -> float | None:
        accepted = self.accepted_correct + self.accepted_errors
        return self.accepted_correct / accepted if accepted else None

    @property
    def coverage(self) -> float:
        return (self.accepted_correct + self.accepted_errors) / self.total if self.total else 0


def summarize_evaluation(
    rows: Iterable[tuple[ClaimAssessment, ExpectedJudgment]],
) -> dict[tuple[str, ClaimKind], EvaluationCounts]:
    """Measure claims against separately authored judgments, never provider scores.

    This reports observations, not threshold authorization. Analyzer identities
    include versions; results must not be pooled across analyzer/claim categories.
    """
    results: dict[tuple[str, ClaimKind], EvaluationCounts] = {}
    for assessment, expected in rows:
        if expected not in {"correct", "rejected", "ambiguous", "unsupported"}:
            raise ValueError("unknown evaluation judgment")
        counts = results.setdefault((assessment.claim.analyzer, assessment.claim.kind), EvaluationCounts())
        counts.total += 1
        accepted = bool(asserted_claims((assessment,)))
        if accepted and expected == "correct":
            counts.accepted_correct += 1
        elif accepted:
            counts.accepted_errors += 1
        else:
            counts.withheld[expected] += 1
    return results
