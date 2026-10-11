"""Existing recipe adapters for the shared per-claim review contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .claim_review import Claim, ClaimAssessment, ClaimKind, Evidence
from .claim_sidecar import BoundClaim, Diagnostic, assessments, load_sidecar, resolve_claims, sidecar_path
from .form_parsing import ParsingCase, ParsingInput, ParsingProposal
from .preview_export import PrincipalPartExportResult
from .principal_parts import PrincipalPartParseFailure, parse_principal_parts
from .sources import read_source_records


@dataclass(frozen=True)
class ReviewState:
    bases: tuple[BoundClaim, ...]
    claims: tuple[BoundClaim, ...]
    diagnostics: tuple[Diagnostic, ...]


def review_state(bases: tuple[BoundClaim, ...], source: Path) -> ReviewState:
    claims, diagnostics = resolve_claims(bases, load_sidecar(sidecar_path(source)))
    inline_stale = tuple(
        Diagnostic("stale", "Stale embedded decision ignored")
        for b in claims
        if b.decision is not None and b.decision.claim_fingerprint != b.claim.fingerprint
    )
    return ReviewState(bases, claims, (*diagnostics, *inline_stale))


def parsing_review(source: Path) -> tuple[ParsingInput, ReviewState]:
    data = ParsingInput.model_validate_json(source.read_text(encoding="utf-8"))
    data = data.model_copy(update={"cases": tuple(_open_features(case) for case in data.cases)})
    bases = tuple(
        BoundClaim(case.latinitas_id, p.feature, p.to_claim(case, data.profile), p.decision)
        for case in data.cases
        for p in case.proposals
    )
    state = review_state(bases, source)
    iterator = iter(state.claims)
    cases = []
    for case in data.cases:
        proposals = []
        for proposal in case.proposals:
            bound = next(iterator)
            proposals.append(
                ParsingProposal(
                    feature=proposal.feature,
                    value=bound.claim.value,
                    analyzer=bound.claim.analyzer,
                    evidence=bound.claim.evidence,
                    alternatives=bound.claim.alternatives,
                    supported=bound.claim.supported,
                    decision=bound.decision,
                )
            )
        cases.append(case.model_copy(update={"proposals": tuple(proposals)}))
    return data.model_copy(update={"cases": tuple(cases)}), state


def _open_features(case: ParsingCase) -> ParsingCase:
    existing = {p.feature for p in case.proposals}
    missing = tuple(
        ParsingProposal(
            feature=feature,
            value="No proposal; reviewer-authored evidence required",
            analyzer="context-placeholder/v1",
            supported=False,
            evidence=(Evidence("confirmed-context", case.object_key, "1", case.context),),
        )
        for feature in case.applicable_features
        if feature not in existing
    )
    return case.model_copy(update={"proposals": (*case.proposals, *missing)})


def principal_review(result: PrincipalPartExportResult) -> ReviewState:
    profile = result.profile
    records = {r.provenance.location: r for r in read_source_records(result.source_path)}
    bases: list[BoundClaim] = []
    for note in result.generation.notes:
        record = records[note.provenance.location]
        parsed = parse_principal_parts(record, profile)
        if isinstance(parsed, PrincipalPartParseFailure):
            continue
        for part in parsed.value.parts:
            if part.is_omitted or part.display is None:
                continue
            for kind in ("label", "segmentation", "explanation"):
                claim_kind: ClaimKind = kind
                claim = Claim(
                    candidate_id=f"{note.latinitas_id}:{part.role}",
                    candidate_text=part.display,
                    kind=claim_kind,
                    value=part.role if kind == "label" else "No proposal; reviewer-authored evidence required",
                    analyzer="confirmed-source/v1",
                    evidence=(
                        Evidence(
                            "confirmed-source",
                            record.provenance.location,
                            "1",
                            record.fields[profile.fields.principal_parts_field],
                        ),
                    ),
                    profile=profile,
                    profile_version="principal-parts/v1",
                    alternatives=part.candidates,
                    supported=kind == "label" and not part.unresolved,
                    extraction=parsed.value.evidence,
                )
                bases.append(BoundClaim(note.latinitas_id, part.role, claim, authorable=not part.unresolved))
    return review_state(tuple(bases), result.source_path)


def principal_assessments(state: ReviewState) -> tuple[ClaimAssessment, ...]:
    # Missing optional proposals do not change the confirmed source-role gate.
    return assessments(tuple(b for b in state.claims if b.decision is not None))
