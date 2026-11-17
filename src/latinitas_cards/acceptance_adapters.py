"""Explicit lossless human projections; historical sidecars are never rewritten."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter

from .acceptance import (
    _immutable,
    _inside,
    check_envelope,
    load_stage,
    save_stage,
    stage_artifact,
)
from .acceptance_records import (
    Acceptance,
    Assertion,
    Binding,
    Check,
    Hash,
    Human,
    Record,
    Text,
    Versions,
    canonical,
    digest,
)
from .annotation_sidecar import Annotation
from .claim_review import Claim, Evidence, assess_claim
from .claim_sidecar import BoundClaim, CorpusBinding, Sidecar, SidecarV2, load_sidecar, resolve_claims
from .corpus_claims import corpus_claim_fingerprint, current_annotations, join_corpus_decisions, load_corpus_decisions
from .corpus_facts import accepted_facts, parse_morphology

_CLAIM = TypeAdapter(Claim)


class ProjectionMapping(Record):
    original_claim_fingerprint: Hash
    assertion_ids: tuple[Text, ...] = Field(min_length=1)


class HumanProjection(Record):
    format: Literal["latinitas-human-projection"] = "latinitas-human-projection"
    version: Literal[1] = 1
    adapter_version: Literal["human-claims/1", "corpus-facts/1"]
    source_sidecar_sha256: Hash
    original_claims: tuple[dict[str, object], ...]
    original_decisions: tuple[dict[str, object], ...]
    corpus_bindings: tuple[CorpusBinding, ...]
    mapping: tuple[ProjectionMapping, ...]
    binding: Binding
    reviewed_sha256: Hash


def claim_assertion(claim: Claim, identity: str) -> Assertion:
    return Assertion(
        id=identity,
        kind=claim.kind,
        value=claim.value,
        selected_sense=claim.selected_sense,
        alternatives=claim.alternatives,
        evidence=claim.evidence,
    )


def _corpus_assertion(row: Annotation, identity: str) -> Assertion:
    if row.layer not in ("segmentation", "morphology"):
        # Legacy translations have no selected-sense attestation. Never invent one.
        raise ValueError("unrepresentable_claim: corpus translation/correction needs fresh review")
    if row.layer == "morphology" and parse_morphology(row.value) is None:
        raise ValueError("unrepresentable_claim: unsupported morphology")
    return Assertion(
        id=identity,
        kind="segmentation" if row.layer == "segmentation" else "label",
        value=canonical(row.value).decode(),
        selected_sense=None,
        alternatives=tuple(canonical(a).decode() for a in row.alternatives or []),
        evidence=(
            Evidence(
                "corpus-annotation",
                row.annotation_id,
                row.producer.version,
                canonical(row.model_dump(mode="json")).decode(),
            ),
        ),
    )


def _corpus_binding(row: Annotation) -> CorpusBinding:
    return CorpusBinding(
        corpus_id=row.anchor.passage_id.split(":", 1)[0],
        passage_id=row.anchor.passage_id,
        passage_hash=row.anchor.passage_hash,
        annotation_id=row.annotation_id,
        layer=row.layer,
    )


def _project(proof: HumanProjection, sidecar_bytes: bytes) -> tuple[tuple[Assertion, ...], tuple[tuple[str, str], ...]]:
    """Reconstruct only values actually reviewed under their original schema."""
    if hashlib.sha256(sidecar_bytes).hexdigest() != proof.source_sidecar_sha256:
        raise ValueError("stale_evidence: legacy sidecar digest")
    raw_sidecar = json.loads(sidecar_bytes)
    if not isinstance(raw_sidecar, dict) or type(raw_sidecar.get("version")) is not int:
        raise ValueError("unsupported_version: legacy sidecar requires integer version")
    if not (len(proof.original_claims) == len(proof.original_decisions) == len(proof.mapping)):
        raise ValueError("unrepresentable_claim: incomplete projection")
    assertions = []
    reviewers = []
    if proof.adapter_version == "human-claims/1":
        original = Sidecar.model_validate_json(sidecar_bytes)
        subset = Sidecar.model_validate_json(canonical({"version": 1, "decisions": proof.original_decisions}))
        if proof.corpus_bindings or any(d not in original.decisions for d in subset.decisions):
            raise ValueError("unrepresentable_claim: foreign decisions")
        bases = tuple(
            BoundClaim(str(i), "projection", _CLAIM.validate_json(canonical(c)))
            for i, c in enumerate(proof.original_claims)
        )
        resolved, diagnostics = resolve_claims(bases, subset)
        original_resolved, _ = resolve_claims(bases, original)
        if diagnostics or original_resolved != resolved:
            raise ValueError("unrepresentable_claim: invalid legacy review")
        for answer in bases:
            if answer.claim.analyzer == "phrase-reference/v1" and not any(
                base.claim.analyzer == "phrase-combination/v1"
                and base.claim.candidate_text == answer.claim.candidate_text
                and base.claim.profile == answer.claim.profile
                and base.claim == current.claim
                for base, current in zip(bases, resolved, strict=True)
            ):
                raise ValueError("unrepresentable_claim: phrase combination approval missing")
        for bound, mapping in zip(resolved, proof.mapping, strict=True):
            claim, decision = bound.claim, bound.decision
            if (
                decision is None
                or assess_claim(claim, decision).status != "accepted"
                or claim.fingerprint != mapping.original_claim_fingerprint
                or len(mapping.assertion_ids) != 1
                or claim.profile.language_tag.split("-")[0].lower() != proof.binding.language
            ):
                raise ValueError("unrepresentable_claim: unreviewed content")
            assertions.append(claim_assertion(claim, mapping.assertion_ids[0]))
            reviewers.append((decision.reviewer, decision.reason))
    else:
        original_v2 = SidecarV2.model_validate_json(sidecar_bytes)
        subset_v2 = SidecarV2.model_validate_json(canonical({"version": 2, "decisions": proof.original_decisions}))
        if len(proof.corpus_bindings) != len(proof.mapping):
            raise ValueError("binding_mismatch: missing corpus bindings")
        for raw, entry, mapping, binding in zip(
            proof.original_claims, subset_v2.decisions, proof.mapping, proof.corpus_bindings, strict=True
        ):
            row = Annotation.model_validate_json(canonical(raw))
            fingerprint = corpus_claim_fingerprint(row)
            if (
                entry not in original_v2.decisions
                or entry.decision.status != "accepted"
                or entry.base_fingerprint != fingerprint
                or entry.binding != _corpus_binding(row)
                or entry.binding != binding
                or mapping.original_claim_fingerprint != fingerprint
                or len(mapping.assertion_ids) != 1
            ):
                raise ValueError("binding_mismatch: invalid corpus projection")
            if sum(d.base_fingerprint == fingerprint for d in original_v2.decisions) != 1:
                raise ValueError("unrepresentable_claim: conflicting corpus proofs")
            assertions.append(_corpus_assertion(row, mapping.assertion_ids[0]))
            reviewers.append((entry.decision.reviewer, entry.decision.reason))
    result = tuple(assertions)
    if digest({"binding": proof.binding, "assertions": result}) != proof.reviewed_sha256:
        raise ValueError("unrepresentable_claim: lossy projection")
    return result, tuple(reviewers)


def _publish(
    root: Path,
    proof: HumanProjection,
    source_bytes: bytes,
    assertions: tuple[Assertion, ...],
    versions: Versions,
) -> Acceptance:
    _, reviewers = _project(proof, source_bytes)
    _immutable(_inside(root, f"artifacts/legacy-{proof.source_sidecar_sha256}.json"), source_bytes)
    projection_id = f"projection-{digest({'proof': proof, 'versions': versions})}"
    projection = save_stage(
        root,
        run_id="legacy-adaptation",
        stage_id=projection_id,
        stage="ingest",
        binding=proof.binding,
        versions=versions,
        envelope=proof.model_dump(mode="json"),
    )
    assert projection.output_sha256 is not None
    envelope = check_envelope(
        proof.binding, versions, "human", assertions, prerequisite_output_sha256s=(projection.output_sha256,)
    )
    check_id = f"human-{digest(envelope)}"
    save_stage(
        root,
        run_id="legacy-adaptation",
        stage_id=check_id,
        stage="decide",
        binding=proof.binding,
        versions=versions,
        envelope=envelope,
        prerequisite_ids=(projection_id,),
    )
    check = Check(
        id=check_id,
        assertion_ids=tuple(a.id for a in assertions),
        method="human",
        version=versions.checker,
        result="pass",
        evidence=tuple(dict.fromkeys(e for a in assertions for e in a.evidence)),
        alternatives=(),
        objections=(),
        input_sha256=digest(envelope),
    )
    return Acceptance.create(
        binding=proof.binding,
        versions=versions,
        assertions=assertions,
        checks=(check,),
        origin="human",
        state="accepted",
        human=Human(
            reviewer="; ".join(dict.fromkeys(r for r, _ in reviewers)),
            reason="; ".join(dict.fromkeys(r for _, r in reviewers)),
            reviewed_sha256=proof.reviewed_sha256,
        ),
        stage_ids=(projection_id, check_id),
        reason="legacy_human_projection",
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


def adapt_deck(
    root: Path,
    *,
    sidecar: Path,
    claims: tuple[BoundClaim, ...],
    binding: Binding,
    versions: Versions,
) -> Acceptance:
    stored = load_sidecar(sidecar)
    source_bytes = sidecar.read_bytes()
    resolved, _ = resolve_claims(claims, stored)
    assertions = tuple(claim_assertion(b.claim, f"claim-{b.claim.fingerprint}") for b in resolved)
    entries = tuple(d for b in claims for d in stored.decisions if d.base_fingerprint == b.claim.fingerprint)
    proof = HumanProjection(
        adapter_version="human-claims/1",
        source_sidecar_sha256=hashlib.sha256(source_bytes).hexdigest(),
        original_claims=tuple(json.loads(_CLAIM.dump_json(b.claim)) for b in claims),
        original_decisions=tuple(d.model_dump(mode="json") for d in entries),
        corpus_bindings=(),
        mapping=tuple(
            ProjectionMapping(original_claim_fingerprint=b.claim.fingerprint, assertion_ids=(a.id,))
            for b, a in zip(resolved, assertions, strict=True)
        ),
        binding=binding,
        reviewed_sha256=digest({"binding": binding, "assertions": assertions}),
    )
    return _publish(root, proof, source_bytes, assertions, versions)


def adapt_corpus(
    root: Path,
    *,
    corpus: Path,
    annotations: Path,
    sidecar: Path,
    annotation_ids: tuple[str, ...],
    binding: Binding,
    versions: Versions,
) -> Acceptance:
    statuses, _ = join_corpus_decisions(corpus, annotations, sidecar)
    stored = load_corpus_decisions(sidecar, missing_ok=False)
    rows = {r.annotation_id: r for r in current_annotations(corpus, annotations)}
    view = accepted_facts(corpus, annotations, sidecar, target_language=binding.language)
    selected = tuple(rows[i] for i in annotation_ids)
    for row in selected:
        if statuses.get(row.annotation_id) != "accepted":
            raise ValueError("stale_evidence: corpus decision is not current")
        if row.layer == "morphology":
            passage = view.passages[row.anchor.passage_id]
            span = (row.anchor.span[0], row.anchor.span[1]) if row.anchor.span else None
            resolution = passage.anchor("morphology", span)
            if resolution.reason is not None:
                raise ValueError(f"unrepresentable_claim: {resolution.reason}")
            facts = [f for group in passage.facts.values() for f in group if f.annotation_id == row.annotation_id]
            if len(facts) != 1 or facts[0].reason is not None:
                raise ValueError("unrepresentable_claim: unusable corpus fact")
    entries = tuple(
        d for row in selected for d in stored.decisions if d.base_fingerprint == corpus_claim_fingerprint(row)
    )
    assertions = tuple(_corpus_assertion(row, f"annotation-{corpus_claim_fingerprint(row)}") for row in selected)
    source_bytes = sidecar.read_bytes()
    proof = HumanProjection(
        adapter_version="corpus-facts/1",
        source_sidecar_sha256=hashlib.sha256(source_bytes).hexdigest(),
        original_claims=tuple(r.model_dump(mode="json") for r in selected),
        original_decisions=tuple(d.model_dump(mode="json") for d in entries),
        corpus_bindings=tuple(_corpus_binding(r) for r in selected),
        mapping=tuple(
            ProjectionMapping(original_claim_fingerprint=corpus_claim_fingerprint(r), assertion_ids=(a.id,))
            for r, a in zip(selected, assertions, strict=True)
        ),
        binding=binding,
        reviewed_sha256=digest({"binding": binding, "assertions": assertions}),
    )
    return _publish(root, proof, source_bytes, assertions, versions)


def verify_projection(root: Path, decision: Acceptance, assertions: tuple[Assertion, ...]) -> None:
    proofs = []
    for identity in decision.stage_ids:
        stage = load_stage(root, identity)
        artifact = stage_artifact(root, stage)
        if not isinstance(artifact, dict) or artifact.get("format") != "latinitas-human-projection":
            continue
        proof = HumanProjection.read_json(canonical(artifact))
        source = _inside(root, f"artifacts/legacy-{proof.source_sidecar_sha256}.json")
        projected, reviewers = _project(proof, source.read_bytes())
        if proof.binding != decision.binding or projected != assertions:
            raise ValueError("unrepresentable_claim: projection content changed")
        if decision.human is None or (
            decision.human.reviewer != "; ".join(dict.fromkeys(r for r, _ in reviewers))
            or decision.human.reason != "; ".join(dict.fromkeys(r for _, r in reviewers))
        ):
            raise ValueError("unrepresentable_claim: original human attribution changed")
        proofs.append(proof)
    if len(proofs) != 1:
        raise ValueError("stale_evidence: missing or conflicting projection")
