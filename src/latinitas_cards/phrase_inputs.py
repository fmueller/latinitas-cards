"""Bounded reviewed lexical inputs; deliberately no phrase generation or export."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .claim_review import Claim, Evidence, assess_claim
from .claim_sidecar import BoundClaim, Diagnostic, claim_report, sidecar_path
from .identity import ResolvedSourceIdentity, resolve_source_identities
from .manifest import CsvIdentityManifest, reconcile_csv_manifest
from .profile import load_profile
from .review_adapters import ReviewState, review_state
from .sources import CanonicalSourceRecord, read_source_records

type Key = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")]
type Case = Literal["nominative", "accusative", "genitive", "dative", "ablative", "vocative", "locative"]
type Gender = Literal["masculine", "feminine", "neuter"]
type POS = Literal["noun", "adjective", "verb"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    @field_validator("version", mode="before", check_fields=False)
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("version must be an integer")
        return value


class Features(Model):
    case: Case | None = None
    number: Literal["singular", "plural"] | None = None
    gender: Gender | None = None
    person: Literal["1", "2", "3"] | None = None
    tense: Literal["present", "imperfect", "future", "perfect", "pluperfect", "future-perfect"] | None = None
    mood: Literal["indicative", "subjunctive", "imperative"] | None = None
    voice: Literal["active", "passive"] | None = None


class Form(Model):
    key: Key
    text: str = Field(min_length=1)
    features: Features


class Sense(Model):
    key: Key
    meaning: str = Field(min_length=1)
    valency: dict[Literal["subject", "object"], Case] = Field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()


class Entry(Model):
    identity: str = Field(min_length=1)
    pos: POS
    gender: Gender | None = None
    evidence: tuple[Evidence, ...] = ()
    senses: tuple[Sense, ...] = ()
    forms: tuple[Form, ...] = ()

    @model_validator(mode="after")
    def distinct_keys(self) -> Entry:
        for keys in ([s.key for s in self.senses], [f.key for f in self.forms]):
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate sense/form keys")
        return self


class Role(Model):
    pos: POS
    senses: tuple[str, ...] = Field(min_length=1, max_length=100)
    requires: Features = Field(default_factory=Features)


class Pattern(Model):
    key: Key
    family: Literal["preposition-noun", "noun-genitive", "verb-noun", "adjective-noun"]
    function_words: tuple[Annotated[str, Field(min_length=1)], ...]
    roles: dict[str, Role]
    evidence: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def bounded_structure(self) -> Pattern:
        expected: dict[str, POS]
        if self.family == "preposition-noun":
            expected = {"object": "noun"}
            if not self.function_words or self.roles.get("object") is None or not self.roles["object"].requires.case:
                raise ValueError("preposition pattern needs explicit function words and case government")
        elif self.family == "noun-genitive":
            expected = {"head": "noun", "dependent": "noun"}
            if self.roles.get("dependent") is None or self.roles["dependent"].requires.case != "genitive":
                raise ValueError("noun-genitive dependent must require genitive")
        elif self.family == "adjective-noun":
            expected = {"head": "noun", "modifier": "adjective"}
        else:
            expected = {"subject": "noun", "verb": "verb"}
            if "object" in self.roles:
                expected["object"] = "noun"
            if self.roles.get("subject") is None or self.roles["subject"].requires.case != "nominative":
                raise ValueError("short clauses require nominative subject")
            if self.roles.get("verb") is None or self.roles["verb"].requires.person != "3":
                raise ValueError("noun subjects require third-person verbs")
        if {key: role.pos for key, role in self.roles.items()} != expected:
            raise ValueError("roles must match bounded pattern family")
        return self


class Binding(Model):
    role: Key
    entry: str = Field(min_length=1)
    sense: Key
    form: Key


class Selection(Model):
    pattern: Key
    bindings: tuple[Binding, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_roles(self) -> Selection:
        if len({b.role for b in self.bindings}) != len(self.bindings):
            raise ValueError("duplicate constituent roles")
        return self


class Limits(Model):
    max_exercises: Annotated[int, Field(strict=True, ge=1, le=1000)] = 100
    variants_per_binding: Annotated[int, Field(strict=True, ge=1, le=32)] = 1


class SelectionRequest(Model):
    version: Literal[1]
    limits: Limits
    selections: tuple[Selection, ...] = Field(min_length=1, max_length=1000)


class PhraseInput(Model):
    version: Literal[1]
    source: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    manifest: str | None = None
    entries: tuple[Entry, ...] = Field(min_length=1, max_length=1000)
    patterns: tuple[Pattern, ...] = Field(min_length=1, max_length=100)
    example_selection: SelectionRequest | None = None

    @model_validator(mode="after")
    def unique_inputs(self) -> PhraseInput:
        for keys in ([e.identity for e in self.entries], [p.key for p in self.patterns]):
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate entry/pattern keys")
        senses = {f"{e.identity}/{s.key}" for e in self.entries for s in e.senses}
        if any(s not in senses for p in self.patterns for r in p.roles.values() for s in r.senses):
            raise ValueError("pattern compatibility names an unknown entry/sense")
        return self


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class PhraseWorkspace:
    data: PhraseInput
    state: ReviewState
    identities: dict[str, ResolvedSourceIdentity]
    protected: tuple[Path, ...]


def load_inputs(path: Path) -> PhraseWorkspace:
    data = PhraseInput.model_validate_json(path.read_text(encoding="utf-8"))
    source, profile_path = path.parent / data.source, path.parent / data.profile
    profile = load_profile(profile_path)
    records = read_source_records(source, source_id_field=profile.source_identity.field, note_type=profile.note_type)
    manifest_path = path.parent / data.manifest if data.manifest else Path(f"{source}.latinitas.json")
    scope: str | None = None
    manifest_ids: tuple[str | None, ...] | None = None
    if source.suffix.lower() == ".csv":
        manifest = CsvIdentityManifest.load(manifest_path)
        if not manifest.is_scoped:
            raise ValueError("CSV requires an existing confirmed source scope; use the source identity workflow first")
        scope = manifest.source_scope
        if profile.source_identity.strategy == "manifest":
            reconciliation = reconcile_csv_manifest(records, manifest)
            if reconciliation.reviews:
                raise ValueError("CSV manifest identities require source review before lexical import")
            manifest_ids = tuple(reconciliation.identities_by_row.get(i) for i in range(len(records)))
    resolved = resolve_source_identities(records, profile, source_scope=scope, manifest_identities=manifest_ids)
    identities = {identity.value: identity for identity in resolved}
    by_identity = dict(zip((i.value for i in resolved), records, strict=True))
    bases: list[BoundClaim] = []

    def proposal(identity: str, feature: str, value: object, evidence: tuple[Evidence, ...], context: object) -> None:
        provenance = evidence or (Evidence("phrase-input/v1", feature, "1", "Missing reviewed evidence"),)
        claim = Claim(
            candidate_id=f"{identity}/{feature}",
            candidate_text=canonical(context),
            kind="label",
            value=canonical(value),
            analyzer="phrase-input/v1",
            evidence=provenance,
            profile=profile,
            profile_version=str(profile.schema_version),
            supported=bool(evidence),
        )
        bases.append(BoundClaim(identity, feature, claim, authorable=False))

    for entry in data.entries:
        record: CanonicalSourceRecord | None = by_identity.get(entry.identity)
        if record is None:
            raise ValueError(f"unknown deck entry identity: {entry.identity}")
        if not record.fields.get(profile.fields.lexical_entry_field, "").strip():
            raise ValueError(f"missing confirmed lexical entry field: {entry.identity}")
        identity = identities[entry.identity]
        context = {"source": asdict(identity), "fields": dict(record.fields), "entry": entry.model_dump(mode="json")}
        if entry.gender is not None:
            proposal(entry.identity, "gender", entry.gender, entry.evidence, context)
        for form in entry.forms:
            proposal(entry.identity, f"form:{form.key}", form.model_dump(mode="json"), entry.evidence, context)
        for sense in entry.senses:
            proposal(entry.identity, f"sense:{sense.key}", sense.model_dump(mode="json"), sense.evidence, context)
    for pattern in data.patterns:
        value = pattern.model_dump(mode="json")
        proposal(f"pattern:{pattern.key}", "pattern", value, pattern.evidence, value)
    return PhraseWorkspace(
        data,
        review_state(tuple(bases), path),
        identities,
        (path, source, profile_path, manifest_path, sidecar_path(path), Path(f"{sidecar_path(path)}.lock")),
    )


def input_diagnostics(work: PhraseWorkspace) -> list[Diagnostic]:
    diagnostics = list(work.state.diagnostics)
    for entry in work.data.entries:
        if not entry.forms:
            diagnostics.append(Diagnostic("missing_paradigm", entry.identity))
        if entry.pos == "noun" and entry.gender is None:
            diagnostics.append(Diagnostic("missing_gender", entry.identity))
        if not entry.senses:
            diagnostics.append(Diagnostic("uncertain_sense", entry.identity))
    for bound in work.state.claims:
        if assess_claim(bound.claim, bound.decision).status != "accepted":
            code = "uncertain_sense" if bound.feature.startswith("sense:") else "unreviewed"
            diagnostics.append(Diagnostic(code, bound.claim.candidate_id))
    return diagnostics


def inspect_inputs(work: PhraseWorkspace) -> dict[str, object]:
    return {
        "version": 1,
        "entries": [
            e.model_dump(mode="json") | {"source": asdict(work.identities[e.identity])} for e in work.data.entries
        ],
        "patterns": [p.model_dump(mode="json") for p in work.data.patterns],
        "claims": [claim_report(b) for b in work.state.claims],
        "diagnostics": [asdict(d) for d in input_diagnostics(work)],
        "example_selection": work.data.example_selection.model_dump(mode="json")
        if work.data.example_selection
        else None,
    }


def bind_selections(work: PhraseWorkspace, request: SelectionRequest) -> tuple[dict[str, object], list[Diagnostic]]:
    diagnostics = list(work.state.diagnostics)
    entries = {e.identity: e for e in work.data.entries}
    patterns = {p.key: p for p in work.data.patterns}
    accepted = {
        (b.identity, b.feature) for b in work.state.claims if assess_claim(b.claim, b.decision).status == "accepted"
    }
    saved: list[dict[str, object]] = []
    for index, selection in enumerate(request.selections):

        def reject(code: str, message: str, item: int = index) -> None:
            diagnostics.append(Diagnostic(code, message, item))

        pattern = patterns.get(selection.pattern)
        if pattern is None:
            reject("unknown_selection", selection.pattern)
            continue
        if (f"pattern:{pattern.key}", "pattern") not in accepted:
            reject("unreviewed", f"pattern:{pattern.key}")
        if {b.role for b in selection.bindings} != set(pattern.roles):
            reject("incompatible", "constituent roles do not match pattern")
            continue
        forms: dict[str, Form] = {}
        senses: dict[str, Sense] = {}
        canonical_bindings: list[dict[str, object]] = []
        for binding in sorted(selection.bindings, key=lambda b: b.role):
            entry = entries.get(binding.entry)
            if entry is None:
                reject("unknown_selection", binding.entry)
                continue
            sense = next((s for s in entry.senses if s.key == binding.sense), None)
            form = next((f for f in entry.forms if f.key == binding.form), None)
            if sense is None:
                reject("uncertain_sense", f"{entry.identity}/{binding.sense}")
            if form is None:
                reject("missing_paradigm" if not entry.forms else "unsupported_form", binding.form)
            role = pattern.roles[binding.role]
            if entry.pos != role.pos or f"{entry.identity}/{binding.sense}" not in role.senses:
                reject("incompatible", f"{binding.role}: sense outside reviewed compatibility")
            required_claims = [f"sense:{binding.sense}", f"form:{binding.form}"]
            if entry.pos == "noun":
                required_claims.append("gender")
                if entry.gender is None:
                    reject("missing_gender", entry.identity)
            for feature in required_claims:
                if (entry.identity, feature) not in accepted:
                    reject("unreviewed", f"{entry.identity}/{feature}")
            if form is not None:
                forms[binding.role] = form
                required = (
                    ("case", "number", "gender")
                    if entry.pos != "verb"
                    else ("person", "number", "tense", "mood", "voice")
                )
                if any(getattr(form.features, f) is None for f in required):
                    reject("unsupported_form", f"{binding.role}: incomplete morphology")
                if entry.pos == "noun" and form.features.gender != entry.gender:
                    reject("agreement", f"{binding.role}: noun gender differs from reviewed gender")
                for feature, value in role.requires.model_dump(exclude_none=True).items():
                    if getattr(form.features, feature) != value:
                        reject(
                            "government" if feature == "case" else "unsupported_form",
                            f"{binding.role}: requires {feature}={value}",
                        )
            if sense is not None:
                senses[binding.role] = sense
            canonical_bindings.append(
                binding.model_dump(mode="json") | {"source": asdict(work.identities[entry.identity])}
            )
        if pattern.family == "adjective-noun" and {"head", "modifier"} <= forms.keys():
            for feature in ("case", "number", "gender"):
                if getattr(forms["head"].features, feature) != getattr(forms["modifier"].features, feature):
                    reject("agreement", f"modifier/head disagree in {feature}")
        if pattern.family == "verb-noun":
            if {"subject", "verb"} <= forms.keys() and forms["subject"].features.number != forms[
                "verb"
            ].features.number:
                reject("agreement", "subject/verb disagree in number")
            if "verb" in senses:
                valency = senses["verb"].valency
                noun_roles = set(pattern.roles) - {"verb"}
                if set(valency) != noun_roles:
                    reject("valency", "sense arguments do not match pattern roles")
                for role_name, case in valency.items():
                    if role_name in pattern.roles and pattern.roles[role_name].requires.case != case:
                        reject("valency", f"{role_name}: pattern government differs from sense")
                    if role_name in forms and forms[role_name].features.case != case:
                        reject("valency", f"{role_name}: sense governs {case}")
        # The single-source note contract receives a composite, never an arbitrarily owned word.
        source_material = [{k: b[k] for k in ("role", "source")} for b in canonical_bindings]
        source_identity = "phrase-source-v1-" + hashlib.sha256(canonical(source_material).encode()).hexdigest()
        object_material: dict[str, object] = {"pattern": pattern.key, "bindings": canonical_bindings}
        object_key = "phrase-object-v1-" + hashlib.sha256(canonical(object_material).encode()).hexdigest()
        saved.append(
            object_material
            | {"source_identity": source_identity, "source_scope": "phrase-input-v1", "object_key": object_key}
        )
    unique = {canonical(s): s for s in saved}
    if len(unique) > request.limits.max_exercises:
        diagnostics.append(Diagnostic("invalid_limits", "explicit selections exceed max_exercises"))
    return {
        "version": 1,
        "limits": request.limits.model_dump(),
        "selections": [unique[k] for k in sorted(unique)],
    }, diagnostics


def save_selections(output: Path, payload: dict[str, object], protected: tuple[Path, ...]) -> None:
    if any(
        output.resolve() == p.resolve() or (output.exists() and p.exists() and output.samefile(p)) for p in protected
    ):
        raise ValueError(
            "selection output must not overwrite an input, source, profile, identity, review sidecar or lock"
        )
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
