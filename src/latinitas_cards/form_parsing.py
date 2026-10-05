"""Offline contextual parsing: individually reviewed claims, never token inference.

Separate contextual objects and a dedicated frozen binding leave the principal
part registry and managed destination schema untouched. Automatic acceptance is
disabled by claim_review until measured category-specific policy exists.
"""

from __future__ import annotations

import html
import json
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cards import CardSlot
from .claim_review import Claim, Evidence, ReviewDecision, assess_claim
from .identity import derive_latinitas_id
from .preview_export import serialize_anki_csv
from .profile import DeckProfile
from .profile_setup import encode_unsafe_controls

type ParsingFeature = Literal["lemma", "person", "number", "tense", "mood", "voice", "case", "gender"]

PARSING_NOTE_TYPE = "Latinitas Contextual Form Parsing v1"
PARSING_SLOT = CardSlot(
    "form_parsing:contextual_analysis",
    "form_parsing",
    "contextual_analysis",
    "Contextual Analysis",
    0,
    "ParsingEnabled",
    "ParsingPrompt",
    "ParsingAnswer",
)
PARSING_FIELDS = (
    "LatinitasID",
    "Form",
    "Context",
    "ParsingEnabled",
    "ParsingPrompt",
    "ParsingAnswer",
    "Personal Notes",
)
PARSING_FRONT = "{{#ParsingEnabled}}{{ParsingPrompt}}{{/ParsingEnabled}}"
PARSING_BACK = (
    "{{#ParsingEnabled}}{{FrontSide}}<hr id=answer>{{ParsingAnswer}}"
    "{{#Personal Notes}}<div>{{Personal Notes}}</div>{{/Personal Notes}}{{/ParsingEnabled}}"
)
_LABELS: dict[ParsingFeature, str] = {
    "lemma": "Lemma",
    "person": "Persona",
    "number": "Numerus",
    "tense": "Tempus",
    "mood": "Modus",
    "voice": "Vox",
    "case": "Casus",
    "gender": "Genus",
}


class ParsingProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: ParsingFeature
    value: str = Field(min_length=1)
    analyzer: str = Field(min_length=1)
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    alternatives: tuple[str, ...] = ()
    supported: bool = True
    decision: ReviewDecision | None = None

    @model_validator(mode="after")
    def nonblank_claim(self) -> ParsingProposal:
        if not self.value.strip() or not self.analyzer.strip():
            raise ValueError("proposal value and analyzer must be nonblank")
        return self

    def to_claim(self, case: ParsingCase, profile: DeckProfile) -> Claim:
        # Every category, context and required-role choice participates in review
        # binding, unlike presentation and mutable text's stable note identity.
        binding = json.dumps(
            [
                case.latinitas_id,
                self.feature,
                case.form,
                case.context,
                case.applicable_features,
                case.required_features,
            ],
            ensure_ascii=False,
        )
        return Claim(
            candidate_id=binding,
            candidate_text=case.form,
            kind="label",
            value=self.value,
            analyzer=self.analyzer,
            evidence=self.evidence,
            profile=profile,
            profile_version="contextual-form-parsing/v1",
            alternatives=self.alternatives,
            supported=self.supported,
        )


class ParsingCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_identity: str = Field(min_length=1)
    source_scope: str = Field(min_length=1)
    object_key: str = Field(min_length=1)
    form: str = Field(min_length=1)
    context: str = Field(min_length=1)
    applicable_features: tuple[ParsingFeature, ...] = Field(min_length=2)
    required_features: tuple[ParsingFeature, ...] = Field(min_length=2)
    proposals: tuple[ParsingProposal, ...] = ()

    @model_validator(mode="after")
    def validate_roles(self) -> ParsingCase:
        if not all(
            s.strip() for s in (self.source_identity, self.source_scope, self.object_key, self.form, self.context)
        ):
            raise ValueError("contextual source identity, object, form and context must be nonblank")
        if "lemma" not in self.required_features or not set(self.required_features) <= set(self.applicable_features):
            raise ValueError("required features must include lemma and be applicable")
        if any(p.feature not in self.applicable_features for p in self.proposals):
            raise ValueError("proposed features must be applicable")
        for roles in (self.applicable_features, self.required_features):
            if len(set(roles)) != len(roles):
                raise ValueError("duplicate feature roles")
        return self

    @property
    def latinitas_id(self) -> str:
        return derive_latinitas_id(
            self.source_identity,
            "contextual-form:" + self.object_key,
            source_scope=self.source_scope,
        )


class ParsingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    profile: DeckProfile
    cases: tuple[ParsingCase, ...]

    @model_validator(mode="after")
    def unique_objects(self) -> ParsingInput:
        ids = [case.latinitas_id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate contextual object bindings")
        return self


class ParsingClaimReport(TypedDict):
    feature: str
    value: str
    fingerprint: str
    status: str
    reason: str
    alternatives: list[str]
    evidence: list[dict[str, str]]


class ParsingResult(TypedDict):
    latinitas_id: str
    semantic_key: str
    form: str
    context: str
    eligible: bool
    prompt: str
    answer: str
    skips: list[str]
    claims: list[ParsingClaimReport]


def _escape(value: str) -> str:
    return html.escape(encode_unsafe_controls(value), quote=True).replace("\n", "<br>")


def generate_parsing(data: ParsingInput) -> tuple[ParsingResult, ...]:
    results: list[ParsingResult] = []
    for case in data.cases:
        claims: list[ParsingClaimReport] = []
        accepted: dict[str, set[str]] = {}
        for proposal in case.proposals:
            claim = proposal.to_claim(case, data.profile)
            assessment = assess_claim(claim, proposal.decision)
            if assessment.status == "accepted":
                accepted.setdefault(proposal.feature, set()).add(proposal.value)
            claims.append(
                {
                    "feature": proposal.feature,
                    "value": proposal.value,
                    "fingerprint": claim.fingerprint,
                    "status": assessment.status,
                    "reason": assessment.reason,
                    "alternatives": list(proposal.alternatives),
                    "evidence": [
                        {"source": e.source, "reference": e.reference, "version": e.version, "text": e.text}
                        for e in proposal.evidence
                    ],
                }
            )
        conflicts = {feature for feature, values in accepted.items() if len(values) != 1}
        skips = [f"Conflicting accepted alternatives: {feature}" for feature in sorted(conflicts)]
        for feature in case.required_features:
            if feature not in accepted:
                skips.append(f"Required claim absent or withheld: {feature}")
        eligible = not skips
        settings = data.profile.morphology
        answer = (
            ""
            if not eligible
            else (
                f'<section class="morphology-v1 morphology-{settings.theme} morphology-{settings.appearance}">'
                + "<br>".join(
                    f"{_LABELS[feature]}: {_escape(next(iter(accepted[feature])))}"
                    for feature in case.applicable_features
                    if feature in accepted
                )
                + "</section>"
            )
        )
        results.append(
            {
                "latinitas_id": case.latinitas_id,
                "semantic_key": PARSING_SLOT.semantic_key,
                "form": case.form,
                "context": case.context,
                "eligible": eligible,
                "prompt": (
                    f"<strong>{_escape(case.form)}</strong><br>{_escape(case.context)}<br>Formam resolve."
                    if eligible
                    else ""
                ),
                "answer": answer,
                "skips": skips,
                "claims": claims,
            }
        )
    return tuple(results)


def parsing_csv(data: ParsingInput) -> bytes:
    """Fresh manual setup only; no managed plan, destination writes or provisioning."""
    return serialize_anki_csv(
        PARSING_NOTE_TYPE,
        data.profile.target_deck,
        (*PARSING_FIELDS[:-1], "Tags"),
        (
            (
                r["latinitas_id"],
                _escape(r["form"]),
                _escape(r["context"]),
                "1",
                r["prompt"],
                r["answer"],
                " ".join(data.profile.tags),
            )
            for r in generate_parsing(data)
            if r["eligible"]
        ),
    )
