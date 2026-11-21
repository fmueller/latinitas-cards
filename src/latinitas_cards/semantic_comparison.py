"""Bounded source-role comparison proofs; no policy or model-granted authority."""

from __future__ import annotations

from typing import Literal

from .acceptance import check_envelope
from .acceptance_records import Assertion, Check, decode_json, digest
from .meaning_evidence import meaning_checks
from .phrase_inputs import Features
from .pipeline_consumers import Rendered, Source, source_claims
from .providers import Call, ModelAnswer, ModelTask

VERSION = "source-role-comparison/1"
type Relation = Literal["equivalent", "compatible_alternatives", "contradiction", "unresolved"]


def _content(assertion: Assertion) -> dict[str, object] | None:
    """Structured source-bound claims, not prose/gloss equality."""
    try:
        value = decode_json(assertion.value)
        if assertion.kind == "segmentation":
            return value if isinstance(value, dict) else None
        if assertion.id == "construction" and assertion.kind == "label" and isinstance(value, dict):
            return value if set(value) == {"pattern", "components"} else None
        if not isinstance(value, dict) or set(value) not in (
            {"surface", "features", "source_role", "anchors", "rule"},
            {"surface", "features", "source_role", "anchors", "rule", "context"},
        ):
            return None
        features = Features.model_validate(value["features"], strict=True)
        return {**value, "features": features.model_dump()}
    except (ValueError, TypeError):
        return None


def compare_source_answers(
    source: Source,
    card: Rendered,
    task: ModelTask,
    proposal: ModelAnswer,
    counter: ModelAnswer,
    proposal_call: Call,
    counter_call: Call,
) -> dict[str, object]:
    """Build only after the provider reader verifies both immutable invocations.

    Source reconstruction is deliberately narrower than the requested model task.
    Meaning equivalence is limited to source-qualified utterances and projections;
    no general prose evaluator or alternative segmentation convention is installed.
    """
    if proposal_call.response_sha256 is None or counter_call.response_sha256 is None:
        raise ValueError("missing actual output commitments")
    expected, constraints, unsupported = source_claims(source, card.objective, card.versions)
    meaning_results = []
    for assertions in (card.claims.assertions, proposal.assertions, counter.assertions):
        meanings, references = meaning_checks(source, card, assertions, expected)
        meaning_results.append(meanings)
        constraints = tuple(dict.fromkeys((*constraints, *references)))
    expected_ids = {a.id for a in card.claims.assertions}
    proposal_ids = [a.id for a in proposal.assertions]
    counter_ids = [a.id for a in counter.assertions]
    complete = (
        len(set(proposal_ids)) == len(proposal_ids)
        and len(set(counter_ids)) == len(counter_ids)
        and set(proposal_ids) == set(counter_ids) == expected_ids
    )
    global_objections = tuple(dict.fromkeys((*proposal.objections, *counter.objections)))
    global_alternatives = tuple(dict.fromkeys((*proposal.alternatives, *counter.alternatives)))
    coverage_objections = global_objections
    if not complete:
        coverage_objections = (*coverage_objections, "incomplete_model_assertion_coverage")
    comparisons = []
    for assertion in card.claims.assertions:
        answers = tuple(
            next((a for a in answer.assertions if a.id == assertion.id), None) for answer in (proposal, counter)
        )
        alternatives = tuple(
            dict.fromkeys(
                (*assertion.alternatives, *global_alternatives, *(v for a in answers if a for v in a.alternatives))
            )
        )
        objections = global_objections
        relation: Relation = "unresolved"
        reason = {
            "source": "source_requires_lexical_and_sense_evidence",
            "rendered": "rendered_clauses_require_independent_evidence",
            "translation": "translation_requires_sense_government_and_semantic_evidence",
            "segmentation": "segmentation_requires_convention_and_reconstruction_evidence",
            "explanation": "explanation_requires_clause_evidence",
        }.get(assertion.id, unsupported)
        if assertion.id in expected:
            kind = "segmentation" if assertion.id == "stem" else "label"
            if any(a is None for a in answers):
                reason = "missing_model_assertion"
            elif objections or alternatives:
                reason = "open_objection_or_unqualified_alternative"
            elif any(
                a is not None
                and (a.kind != kind or a.selected_sense is not None or _content(a) != expected[assertion.id])
                for a in (assertion, *answers)
            ):
                relation, reason = "contradiction", "source_role_feature_or_surface_contradiction"
            else:
                relation = "equivalent"
                reason = "independently_attested_segmentation" if assertion.id == "stem" else unsupported
        if any(assertion.id in result for result in meaning_results):
            dispositions = {result.get(assertion.id, "unresolved") for result in meaning_results}
            if dispositions == {"equivalent"} and not objections and not global_alternatives:
                relation, reason = "equivalent", "independently_grounded_meaning_and_projection"
            elif relation != "contradiction":
                if "contradiction" in dispositions:
                    relation, reason = "contradiction", "bounded_meaning_contradiction"
                elif "unresolved" in dispositions:
                    relation, reason = "unresolved", "bounded_meaning_unresolved"
                elif "ambiguous" in dispositions:
                    relation, reason = "unresolved", "bounded_meaning_ambiguous"
                elif "compatible_alternatives" in dispositions:
                    relation, reason = "compatible_alternatives", "bounded_meaning_compatible_alternatives"
                else:
                    relation, reason = "unresolved", "open_objection_or_unqualified_alternative"
        if relation != "equivalent":
            objections = tuple(dict.fromkeys((*objections, reason)))
        envelope = {
            **check_envelope(
                card.binding,
                card.versions,
                "semantic_comparison",
                (assertion,),
                prerequisite_output_sha256s=(proposal_call.response_sha256, counter_call.response_sha256),
                reference_constraints=constraints,
            ),
            "proposal_output_sha256": proposal_call.response_sha256,
            "counter_output_sha256": counter_call.response_sha256,
        }
        check = Check(
            id="semantic/" + assertion.id,
            assertion_ids=(assertion.id,),
            method="semantic_comparison",
            version=VERSION,
            result="pass" if relation == "equivalent" else "fail" if relation == "contradiction" else "unresolved",
            evidence=constraints,
            alternatives=alternatives,
            objections=objections,
            input_sha256=digest(envelope),
        )
        comparisons.append({"assertion_id": assertion.id, "relation": relation, "check": check, "envelope": envelope})
    relations = {item["relation"] for item in comparisons}
    if "contradiction" in relations:
        relation = "contradiction"
    elif relations == {"equivalent"}:
        relation = "equivalent"
    else:
        relation = "unresolved"
    if relation != "contradiction" and (not complete or global_objections or global_alternatives):
        relation = "unresolved"
    blind_envelope = {
        "task": task,
        "request_sha256": counter_call.request_sha256,
        "response_sha256": counter_call.response_sha256,
        "call_id": counter_call.call_id,
    }
    blind_check = Check(
        id="blind/transport",
        assertion_ids=tuple(a.id for a in card.claims.assertions),
        method="blind_model",
        version="blind/1",
        result="pass" if complete and not global_objections and not global_alternatives else "unresolved",
        evidence=task.source_evidence,
        alternatives=global_alternatives,
        objections=coverage_objections,
        input_sha256=digest(blind_envelope),
    )
    return {
        "format": "latinitas-source-comparison",
        "version": 1,
        "comparison_version": VERSION,
        "binding": card.binding,
        "versions": card.versions,
        "assertions": card.claims.assertions,
        "proposal_call_id": proposal_call.call_id,
        "counter_call_id": counter_call.call_id,
        "proposal_output_sha256": proposal_call.response_sha256,
        "counter_output_sha256": counter_call.response_sha256,
        "blind_envelope": blind_envelope,
        "blind_input_sha256": digest(blind_envelope),
        "blind_check": blind_check,
        "complete_assertion_coverage": complete,
        "comparisons": comparisons,
        "relation": relation,
        "authorizes_acceptance": False,
        "objections": coverage_objections,
        "alternatives": global_alternatives,
    }
