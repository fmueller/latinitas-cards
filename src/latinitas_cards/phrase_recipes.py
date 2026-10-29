"""Explicit phrase recipes and reference proposals; no object approval or export.

Contextual readings are authored, never inferred from an isolated paradigm. A
shared claim decision can attest the complete reference; T087 owns storing it.
The extended layout is a contract for T088, not a change to installed note types.
"""

from __future__ import annotations

import hashlib
from html import escape
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import Field, model_validator

from .cards import CardSlot
from .claim_review import Claim, Evidence, ReviewDecision, assess_claim
from .phrase_generation import SavedSelections, generate_proposals
from .phrase_inputs import Features, Key, Model, PhraseWorkspace, canonical
from .profile import DeckProfile
from .reference_templates import REFERENCE_CARD_TEMPLATES, REFERENCE_NOTE_TYPE_FIELDS, ReferenceCardTemplate
from .terminology import card_language, card_text, feature_name, feature_value

type Recipe = Literal["phrase_translation", "phrase_recognition"]
type Distinction = Literal["meaning", "analysis", "case", "number", "gender", "person", "tense", "mood", "voice"]
type Text = Annotated[str, Field(min_length=1, max_length=4096)]

PHRASE_LAYOUT_VERSION = 1
PHRASE_SCHEMA_VERSION = "phrase-v1"


def _phrase_template(recipe: Recipe, label: str, ordinal: int) -> ReferenceCardTemplate:
    slot = CardSlot(
        semantic_key=f"{recipe}:phrase",
        recipe=recipe,
        role="phrase",
        template_name=f"Phrase {label}",
        ordinal=ordinal,
        enabled_field=f"Phrase{label}Enabled",
        prompt_field=f"Phrase{label}Prompt",
        answer_field=f"Phrase{label}Answer",
    )
    return ReferenceCardTemplate(
        slot=slot,
        front=f"{{{{#{slot.enabled_field}}}}}{{{{{slot.prompt_field}}}}}{{{{/{slot.enabled_field}}}}}",
        back=f"{{{{#{slot.enabled_field}}}}}{{{{{slot.answer_field}}}}}{{{{/{slot.enabled_field}}}}}",
    )


# Existing ordinals, fields and templates remain exactly the old contract.
PHRASE_LAYOUT = (
    *REFERENCE_CARD_TEMPLATES,
    _phrase_template("phrase_translation", "Translation", 10),
    _phrase_template("phrase_recognition", "Recognition", 11),
)
PHRASE_LAYOUT_FIELDS = (
    *REFERENCE_NOTE_TYPE_FIELDS,
    *(
        name
        for template in PHRASE_LAYOUT[10:]
        for name in (template.slot.enabled_field, template.slot.prompt_field, template.slot.answer_field)
    ),
)

# Phrase input values map into the shared canonical display vocabulary.
_CANONICAL = {
    "case": {
        "nominative": "nominativus",
        "accusative": "accusativus",
        "genitive": "genetivus",
        "dative": "dativus",
        "ablative": "ablativus",
        "vocative": "vocativus",
        "locative": "locativus",
    },
    "number": {"singular": "singularis", "plural": "pluralis"},
    "gender": {"masculine": "masculinum", "feminine": "femininum", "neuter": "neutrum"},
    "person": {"1": "prima", "2": "secunda", "3": "tertia"},
    "tense": {
        "present": "praesens",
        "imperfect": "imperfectum",
        "future": "futurum",
        "perfect": "perfectum",
        "pluperfect": "plusquamperfectum",
        "future-perfect": "futurum exactum",
    },
    "mood": {"indicative": "indicativus", "subjunctive": "coniunctivus", "imperative": "imperativus"},
    "voice": {"active": "activa", "passive": "passiva"},
}


class Reading(Model):
    translations: tuple[Text, ...] = Field(min_length=1, max_length=16)
    features: dict[Key, Features] = Field(min_length=1, max_length=3)
    contributions: dict[Key, Text] = Field(min_length=1, max_length=3)
    explanation: Text

    @model_validator(mode="after")
    def coherent_roles(self) -> Reading:
        if self.features.keys() != self.contributions.keys():
            raise ValueError("Every analyzed role needs its contribution to the construction")
        if any(not f.model_dump(exclude_none=True) for f in self.features.values()):
            raise ValueError("Each contextual analysis needs applicable features")
        return self


class Reference(Model):
    language: Literal["de", "en"] = "de"
    readings: tuple[Reading, ...] = Field(min_length=1, max_length=16)
    evidence: tuple[Evidence, ...] = Field(min_length=1, max_length=16)


class Exercise(Model):
    candidate_key: Text
    recipe: Recipe
    target: Key | None = None
    distinction: Distinction
    reference: Reference

    @model_validator(mode="after")
    def primary_objective(self) -> Exercise:
        if self.recipe == "phrase_translation":
            if self.target is not None or self.distinction != "meaning":
                raise ValueError("Translation tests meaning, without a designated form")
        elif self.target is None or self.distinction == "meaning":
            raise ValueError("Recognition needs an explicit role target and tested distinction")
        return self

    @property
    def semantic_key(self) -> str:
        material = (self.candidate_key, self.recipe, self.target, self.distinction)
        return "phrase-exercise-v1-" + hashlib.sha256(canonical(material).encode()).hexdigest()


class RecipePlan(Model):
    version: Literal[1]
    max_exercises: Annotated[int, Field(strict=True, ge=1, le=1000)]
    exercises: tuple[Exercise, ...] = Field(max_length=1000)

    @model_validator(mode="after")
    def explicit_budget(self) -> RecipePlan:
        if len(self.exercises) > self.max_exercises:
            raise ValueError("Explicit recipe choices exceed max_exercises")
        keys = [e.semantic_key for e in self.exercises]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate semantic exercises; select each target/distinction once")
        return self


def contextual_claim(candidate: dict[str, object], exercise: Exercise, profile: DeckProfile) -> Claim:
    """Bind the reference to current phrase/input evidence, not display settings."""
    context = {k: v for k, v in candidate.items() if k not in {"input_claims", "review_state"}}
    context["input_fingerprints"] = sorted(
        str(c["fingerprint"]) for c in cast(list[dict[str, object]], candidate.get("input_claims", []))
    )
    return Claim(
        candidate_id=exercise.semantic_key,
        candidate_text=canonical(context),
        kind="explanation",
        value=canonical(exercise.reference.model_dump(mode="json")),
        analyzer="phrase-reference/v1",
        evidence=exercise.reference.evidence,
        profile=profile,
        profile_version="phrase-v1",
        alternatives=tuple(canonical(r.model_dump(mode="json")) for r in exercise.reference.readings)
        if len(exercise.reference.readings) > 1
        else (),
    )


def render_phrase_text(text: str, profile: DeckProfile) -> str:
    """Apply the shared reference styling without interpreting literal prose."""
    appearance_class = "" if profile.morphology.appearance == "auto" else f" morphology-{profile.morphology.appearance}"
    panel = f'<div class="morphology-v1 morphology-{profile.morphology.theme}{appearance_class}">'
    return panel + escape(text).replace("\n", "<br>") + "</div>"


def render_exercise(
    candidate: dict[str, object],
    exercise: Exercise,
    profile: DeckProfile,
    decision: ReviewDecision | None = None,
    *,
    reference_claim: Claim | None = None,
) -> dict[str, Any]:
    """Complete inspection, including proposed/withheld text; no export authorization."""
    if candidate["candidate_key"] != exercise.candidate_key:
        raise ValueError("Recipe candidate key differs from current phrase")
    # Generated candidates are internal JSON; validate the roles needed by the reference.
    constituents = candidate["constituents"]
    if not isinstance(constituents, list):
        raise ValueError("Phrase constituents must be a list")
    roles: dict[str, str] = {}
    for constituent in constituents:
        if not isinstance(constituent, dict) or not isinstance(constituent.get("form"), dict):
            raise ValueError("Invalid phrase constituent")
        roles[str(constituent["role"])] = str(constituent["form"]["text"])
    if exercise.target is not None and exercise.target not in roles:
        raise ValueError("Designated target is not a constituent of the phrase")
    if any(set(r.features) != set(roles) for r in exercise.reference.readings):
        raise ValueError("Each contextual reading must describe exactly the phrase's constituent roles")
    settings = profile.morphology.terminology
    language_tag = profile.language_tag
    prompt = str(candidate["text"]) + "\n"
    if exercise.recipe == "phrase_translation":
        prompt += card_text("phrase_translation_prompt", language_tag)
    else:
        assert exercise.target is not None
        distinction = (
            card_text("analysis", language_tag)
            if exercise.distinction == "analysis"
            else feature_name(exercise.distinction, settings, language_tag)
        )
        prompt += card_text(
            "phrase_recognition_prompt",
            language_tag,
            form=roles[exercise.target],
            target=exercise.target,
            distinction=distinction,
        )
    claim = reference_claim or contextual_claim(candidate, exercise, profile)
    assessment = assess_claim(claim, decision)
    state = "reviewed" if assessment.status == "accepted" else "proposed" if decision is None else "withheld"
    reason = assessment.reason
    if exercise.target and exercise.distinction != "analysis":
        values = {getattr(r.features[exercise.target], exercise.distinction) for r in exercise.reference.readings}
        if None in values or len(values) != 1:
            state, reason = "withheld", "The tested distinction is unresolved in the supplied contextual readings."
    # A reference in another language never reaches a card, even with an accepted decision.
    language = card_language(language_tag)
    mismatched = exercise.reference.language != language
    if mismatched:
        mismatch = f"Reference language {exercise.reference.language!r} differs from card language {language!r}"
        # Keep an unresolved-distinction reason; a review decision's reason is superseded.
        reason = f"{reason} {mismatch}" if state == "withheld" and reason != assessment.reason else mismatch
        state = "withheld"
    lines = []
    for index, reading in enumerate(() if mismatched else exercise.reference.readings, 1):
        lines.append(card_text("reading", language_tag, index=index) + " / ".join(reading.translations))
        if exercise.target:
            features = reading.features[exercise.target].model_dump(exclude_none=True)
            lines.extend(
                f"{feature_name(f, settings, language_tag)}: "
                f"{feature_value(f, _CANONICAL[f][v], settings, language_tag)}"
                for f, v in features.items()
            )
            lines.append(reading.contributions[exercise.target])
        lines.append(reading.explanation)
    if exercise.recipe == "phrase_translation" and not mismatched:
        lines.append(card_text("self_assessment", language_tag))
    heading = card_text("reviewed_reference" if state == "reviewed" else "proposed_reference", language_tag)
    answer = heading + "\n" + "\n".join(lines)
    if state == "withheld":
        answer = card_text("withheld", language_tag) + reason + "\n" + answer
    return {
        "semantic_key": exercise.semantic_key,
        "candidate_key": exercise.candidate_key,
        "recipe": exercise.recipe,
        "target": exercise.target,
        "distinction": exercise.distinction,
        "schema": PHRASE_SCHEMA_VERSION,
        "layout_version": PHRASE_LAYOUT_VERSION,
        "review_state": state,
        "reason": reason,
        "enabled": "1" if state == "reviewed" else "",
        "prompt": prompt,
        "answer": answer,
        "prompt_html": render_phrase_text(prompt, profile),
        "answer_html": render_phrase_text(answer, profile),
        "readings": [r.model_dump(mode="json") for r in exercise.reference.readings],
        "claim_fingerprint": claim.fingerprint,
        "grading": "self-assessment" if exercise.recipe == "phrase_translation" else "recognition",
    }


def load_recipe_plan(path: Path) -> RecipePlan:
    with path.open("rb") as stream:
        contents = stream.read(4 * 1024 * 1024 + 1)
    if len(contents) > 4 * 1024 * 1024:
        raise ValueError("Recipe plan exceeds 4 MiB; narrow explicit selections")
    return RecipePlan.model_validate_json(contents)


def preview_plan(
    work: PhraseWorkspace, saved: SavedSelections, plan: RecipePlan, profile: DeckProfile
) -> dict[str, object]:
    proposals = generate_proposals(work, saved)
    candidates = {str(c["candidate_key"]): c for c in cast(list[dict[str, object]], proposals["candidates"])}
    exercises = []
    skipped = []
    for exercise in sorted(plan.exercises, key=lambda e: e.semantic_key):
        candidate = candidates.get(exercise.candidate_key)
        if candidate is None:
            skipped.append({"semantic_key": exercise.semantic_key, "reason": "Current reviewed phrase unavailable"})
            continue
        exercises.append(render_exercise(candidate, exercise, profile))
    return {
        "version": 1,
        "schema": PHRASE_SCHEMA_VERSION,
        "layout_version": PHRASE_LAYOUT_VERSION,
        "terminology": profile.morphology.terminology.model_dump(),
        "limits": {"max_exercises": plan.max_exercises},
        "exercises": exercises,
        "skipped": skipped,
        "generation": proposals,
        "supported_subset": "Explicit contextual reference proposals only. No exact-string grading, "
        "object approval, export, setup or application. Review persistence belongs to T087.",
    }
