"""Phrase learning-object identities and T064 sidecar adapters, not CSV application.

The recipe plan owns the exercise sidecar; lexical input decisions remain beside
the input file. Neither attestation authorizes an export or managed update.
"""

from dataclasses import asdict, replace
from html import escape
from pathlib import Path
from typing import Any, cast

from .claim_review import assess_claim
from .claim_sidecar import BoundClaim, Diagnostic, claim_report
from .identity import derive_latinitas_id
from .phrase_generation import load_saved_selections
from .phrase_inputs import load_inputs
from .phrase_recipes import PHRASE_LAYOUT, Reference, contextual_claim, load_recipe_plan, preview_plan, render_exercise
from .profile import load_profile
from .review_adapters import ReviewState, review_state
from .review_inspection import ReviewWorkspace


def phrase_workspace(source: Path, selections: Path, recipes: Path) -> tuple[dict[str, object], ReviewWorkspace]:
    work = load_inputs(source)
    profile = load_profile(source.parent / work.data.profile)
    plan = load_recipe_plan(recipes)
    report = preview_plan(work, load_saved_selections(selections), plan, profile)
    generation = cast(dict[str, Any], report["generation"])
    candidates = {c["candidate_key"]: c for c in generation["candidates"]}
    bases: list[BoundClaim] = []
    for exercise in plan.exercises:
        if exercise.candidate_key not in candidates:
            continue
        candidate = candidates[exercise.candidate_key]
        answer_claim = contextual_claim(candidate, exercise, profile)
        combination_claim = replace(
            answer_claim,
            candidate_id=exercise.candidate_key,
            value="Explicitly approve this combination's grammatical correctness and naturalness.",
            analyzer="phrase-combination/v1",
            alternatives=(),
            evidence=tuple(
                e
                for b in work.state.claims
                if b.claim.fingerprint in {c["fingerprint"] for c in candidate["input_claims"]}
                for e in b.claim.evidence
            ),
        )
        # One combination decision is shared by questions about the same phrase.
        if not any(b.claim.fingerprint == combination_claim.fingerprint for b in bases):
            bases.append(BoundClaim(exercise.candidate_key, "combination", combination_claim, authorable=False))
        bases.append(BoundClaim(exercise.semantic_key, "answer", answer_claim))
    state = review_state(tuple(bases), recipes)
    claims_by_identity = {(bound.identity, bound.feature): bound for bound in state.claims}
    diagnostics = [*work.state.diagnostics, *state.diagnostics]
    rendered = []
    fields = []
    cards = []
    for exercise in sorted(plan.exercises, key=lambda e: e.semantic_key):
        candidate = candidates.get(exercise.candidate_key)
        if candidate is None:
            diagnostics.append(
                Diagnostic("unavailable", f"Current reviewed phrase unavailable: {exercise.semantic_key}")
            )
            continue
        combination = claims_by_identity[(exercise.candidate_key, "combination")]
        answer = claims_by_identity[(exercise.semantic_key, "answer")]
        reasons = []
        for bound in (combination, answer):
            assessment = assess_claim(bound.claim, bound.decision)
            if assessment.status != "accepted":
                reasons.append(f"{bound.feature}: {assessment.reason}")
        try:
            reference = Reference.model_validate_json(answer.claim.value)
            corrected = exercise.model_copy(update={"reference": reference})
            result = render_exercise(candidate, corrected, profile, answer.decision, reference_claim=answer.claim)
        except ValueError as error:
            reasons.append(f"answer: invalid authored reference: {error}")
            result = render_exercise(candidate, exercise, profile)
        if (
            result["review_state"] == "withheld"
            and result["reason"] != assess_claim(answer.claim, answer.decision).reason
        ):
            reasons.append(result["reason"])
        eligible = not reasons and result["review_state"] == "reviewed"
        if not eligible and (combination.decision is not None or answer.decision is not None):
            result["review_state"] = "withheld"
            result["reason"] = "; ".join(reasons)
            if result["answer"].startswith("Withheld: "):
                result["answer"] = result["answer"].split("\n", 1)[1]
            result["answer"] = "Withheld: " + result["reason"] + "\n" + result["answer"]
            result["answer_html"] = "<div>" + escape(result["answer"]).replace("\n", "<br>") + "</div>"
        slot = next(t.slot for t in PHRASE_LAYOUT if t.slot.recipe == exercise.recipe)
        # Each target/distinction is a coherent exercise note; the canonical phrase
        # object remains explicit provenance, never a word-owned or merged note.
        identity = derive_latinitas_id(
            str(candidate["source_identity"]), exercise.semantic_key, source_scope=str(candidate["source_scope"])
        )
        claims = [claim_report(combination), claim_report(answer), *candidate["input_claims"]]
        result.update(
            latinitas_id=identity,
            object_key=exercise.candidate_key,
            slot=slot.ordinal,
            card_semantic_key=slot.semantic_key,
            eligible=eligible,
            reasons=reasons,
            claims=claims,
            enabled="1" if eligible else "",
        )
        if not eligible:
            diagnostics.append(Diagnostic("unreviewed", f"{exercise.semantic_key}: {'; '.join(reasons)}"))
        rendered.append(result)
        fields.append(
            {
                "LatinitasID": identity,
                slot.enabled_field: result["enabled"],
                slot.prompt_field: result["prompt_html"],
                slot.answer_field: result["answer_html"],
            }
        )
        cards.append(
            {
                "identity": identity,
                "object_key": exercise.candidate_key,
                "recipe": exercise.recipe,
                "slot": slot.ordinal,
                "semantic_key": slot.semantic_key,
                "exercise_key": exercise.semantic_key,
                "eligible": eligible,
                "prompt": result["prompt"],
                "answer": result["answer"],
                "readings": result["readings"],
                "claims": claims,
                "reasons": reasons,
                "constituents": candidate["constituents"],
            }
        )
    report.update(
        exercises=rendered,
        diagnostics=[asdict(d) for d in diagnostics],
        supported_subset="Explicit reviewed phrase exercises. Claim attestation is not export or managed-plan "
        "approval. CSV/destination application belongs to T088; native Desktop/Mobile acceptance remains pending.",
    )
    state = ReviewState(state.bases, state.claims, tuple(diagnostics))
    return report, ReviewWorkspace(state, tuple(fields), tuple(t.slot for t in PHRASE_LAYOUT), tuple(cards), profile)
