"""Confirmed deck mappings and bounded Latin corroboration, outside CLI orchestration.

Source morphology, lexical sense plausibility and bounded construction constraints
are separate checks. None approves rendered translations or explanatory clauses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from unicodedata import category

from pydantic import Field, model_validator

from .acceptance import check_envelope, load_stage, reusable_stage, save_stage, stage_artifact
from .acceptance_policy import policy_reason
from .acceptance_records import Acceptance, Assertion, Binding, Check, Record, Stage, Text, Versions, canonical, digest
from .claim_review import Evidence
from .html_text import source_html_readable_spans, source_html_to_text
from .identity import resolve_source_identity
from .latin_library import LatinLibrary, LibraryReading, ProcessingUnavailable
from .phrase_inputs import POS, Features, Form, Pattern, Sense, construction_diagnostics
from .principal_parts import PrincipalPartParseSuccess, parse_principal_parts
from .profile import DeckProfile
from .sources import CanonicalSourceRecord

CHECKER = "deck-evidence/1"
SCOPE = "mapped morphology and lexical sense plausibility; no translation entailment"
# Only fixed government is checked here; dual-government prepositions need context.
GOVERNMENT = {
    "cum": "ablative",
    "sine": "ablative",
    "ab": "ablative",
    "ā": "ablative",
    "ex": "ablative",
    "ē": "ablative",
    "dē": "ablative",
    "ad": "accusative",
    "per": "accusative",
    "propter": "accusative",
    "post": "accusative",
    "ante": "accusative",
}


class SourceCell(Record):
    field: Text
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]

    def extract(self, record: CanonicalSourceRecord) -> str:
        raw = record.fields.get(self.field, "")
        if self.start >= self.end or self.end > len(raw):
            raise ValueError("missing mapped source cell")
        readable = source_html_readable_spans(raw)
        indices = [i for i, (_, start, end) in enumerate(readable) if start < self.end and end > self.start]
        if not indices:
            raise ValueError("missing readable mapped source cell")
        first, last = indices[0], indices[-1]
        selected_spans = readable[first : last + 1]
        if any(start < self.start or end > self.end for _, start, end in selected_spans):
            raise ValueError("source cell splits a character reference")
        selected = "".join(char for char, _, _ in selected_spans)

        def word(char: str) -> bool:
            return char.isalpha() or category(char).startswith("M")

        if (first and word(readable[first - 1][0]) and word(selected[0])) or (
            last + 1 < len(readable) and word(selected[-1]) and word(readable[last + 1][0])
        ):
            raise ValueError("source cell splits a word")
        selected = " ".join(selected.split())
        if not selected:
            raise ValueError("missing readable mapped source cell")
        return selected


class DeckTask(Record):
    form: SourceCell
    meaning: SourceCell
    pos: POS
    features: Features
    source_role: Literal["present_1s", "present_infinitive", "perfect_1s", "supine", "ppp", "other"]
    selected_sense: Text | None
    preposition: Text | None
    extra_assertions: tuple[Assertion, ...]

    @model_validator(mode="after")
    def complete_features(self) -> DeckTask:
        required = (
            ("person", "number", "tense", "mood", "voice") if self.pos == "verb" else ("case", "number", "gender")
        )
        if any(getattr(self.features, k) is None for k in required):
            raise ValueError("complete requested morphology required")
        if any(a.id in ("morphology", "sense") for a in self.extra_assertions):
            raise ValueError("reserved evidence assertion ID")
        return self


class MappedSource(Record):
    lemma: Text
    form: Text
    meaning: Text
    task: DeckTask


@dataclass(frozen=True)
class DeckAnalysis:
    state: str
    checks: tuple[Check, ...]
    stage: Stage
    decision: Acceptance


def _reading(reading: LibraryReading) -> str:
    return canonical({"lemma": reading.lemma, "pos": reading.pos, "features": reading.features.model_dump()}).decode()


def analyze_deck(
    root: Path,
    *,
    record: CanonicalSourceRecord,
    profile: DeckProfile,
    source_scope: str,
    task: DeckTask,
    versions: Versions,
    library: LatinLibrary,
    manifest_identity: str | None = None,
) -> DeckAnalysis:
    """Persist exact source spans, hypotheses, checks and withheld/failed decisions.

    ``profile`` is the caller's confirmed effective mapping; manifest identities
    must come from the existing reconciliation workflow, not an analyzer result.
    Library implementations are application dependencies, never run-input plugins.
    """
    if versions.analyzer != library.version or versions.checker != CHECKER:
        raise ValueError("analyzer version or checker version mismatch")
    policy_reason(versions, "1", ())  # Validate installed policy before doing analysis.
    identity = resolve_source_identity(record, profile, source_scope=source_scope, manifest_identity=manifest_identity)
    if (
        record.source_identity is not None
        and profile.source_identity.strategy == "source_id_field"
        and record.fields.get(profile.source_identity.field or "") != record.source_identity
    ):
        raise ValueError("source identity mismatch")
    if task.form.field not in (profile.fields.lexical_entry_field, profile.fields.principal_parts_field):
        raise ValueError("form is outside confirmed form mapping")
    if task.meaning.field != profile.fields.meaning_field:
        raise ValueError("meaning is outside confirmed meaning mapping")
    form, meaning = task.form.extract(record), task.meaning.extract(record)
    lemma = source_html_to_text(record.fields.get(profile.fields.lexical_entry_field, "")).strip()
    if not lemma:
        raise ValueError("missing confirmed lemma")
    if task.source_role != "other":
        parsed = parse_principal_parts(record, profile)
        if not isinstance(parsed, PrincipalPartParseSuccess):
            raise ValueError("unconfirmed source role")
        part = parsed.value.by_role.get(task.source_role)
        if part is None or part.unresolved or part.is_omitted or part.display != form:
            raise ValueError("source role not confirmed for selected form")
        if task.form.field != profile.fields.principal_parts_field:
            raise ValueError("source role outside confirmed principal-part mapping")
    selected = {"lemma": lemma, "form": form, "meaning": meaning, "task": task.model_dump(mode="json")}
    mapping = {
        "fields": profile.fields.model_dump(),
        "roles": profile.principal_parts.model_dump(),
        "identity": profile.source_identity.model_dump(),
    }
    if profile.language_tag not in ("de", "en"):
        raise ValueError("deck evidence requires de or en")
    binding = Binding(
        source_scope=source_scope,
        source_identity=identity.value,
        source_sha256=digest(dict(record.fields)),
        extraction_sha256=digest(selected),
        mapping_sha256=digest(mapping),
        task_sha256=digest(task),
        language="de" if profile.language_tag == "de" else "en",
        config_sha256=digest({"scope": SCOPE, "checker": CHECKER}),
    )
    source_evidence = (
        Evidence("confirmed-deck-mapping", f"{source_scope}/{identity.value}", "1", canonical(selected).decode()),
    )
    morphology = Assertion(
        id="morphology",
        kind="label",
        value=canonical(task.features.model_dump()).decode(),
        selected_sense=None,
        alternatives=(),
        evidence=source_evidence,
    )
    sense_assertion = Assertion(
        id="sense",
        kind="label",
        value=canonical({"meaning": meaning, "selected_sense": task.selected_sense}).decode(),
        selected_sense=None,
        alternatives=(),
        evidence=source_evidence,
    )
    assertions = (morphology, sense_assertion, *task.extra_assertions)
    reason = ""
    senses: tuple[Sense, ...] = ()
    try:
        readings = library.analyze(form)
        senses = library.senses(lemma, binding.language)
        state, result, objections = _corroborate(task, lemma, readings, exhaustive=library.ambiguity_complete)
    except ProcessingUnavailable as exc:
        readings = ()
        state, result, objections, reason = "unavailable", "unresolved", ("processing_unavailable",), str(exc)
    alternatives = tuple(dict.fromkeys(_reading(r) for r in readings))
    sense_result, sense_objections, selected_sense = _sense(task, meaning, senses)
    if state == "unavailable":
        sense_result, sense_objections = "unresolved", ("processing_unavailable",)
    elif sense_result == "fail":
        state = "contradiction"
    elif state == "corroborated" and sense_result != "pass":
        state = "unresolved" if sense_result == "unresolved" else "unsupported"
    sense_alternatives = tuple(dict.fromkeys(canonical(s.model_dump(mode="json")).decode() for s in senses))
    library_evidence = Evidence(
        "latin-library",
        "lemma/POS/complete features; alternatives retained",
        library.version,
        canonical(
            {
                "readings": alternatives,
                "senses": sense_alternatives,
                "scope": SCOPE,
                "ambiguity_complete": library.ambiguity_complete,
            }
        ).decode(),
    )
    constraints: tuple[Evidence, ...] = source_evidence + (library_evidence,)
    if task.preposition is not None:
        constraints += (
            Evidence(
                "latinitas-latin-rules",
                "fixed-preposition-government/1",
                CHECKER,
                f"{task.preposition}: {GOVERNMENT.get(task.preposition, 'context required')}",
            ),
        )
    run_id = "deck-" + digest(
        {"binding": binding, "versions": versions, "constraints": constraints, "state": state, "reason": reason}
    )
    ingest = save_stage(
        root,
        run_id=run_id,
        stage_id=run_id + "-source",
        stage="ingest",
        binding=binding,
        versions=versions,
        envelope={"source": selected, "mapping": mapping, "raw_fields": dict(record.fields)},
    )
    assert ingest.output_sha256 is not None
    checks: list[Check] = []
    stages = [ingest.stage_id]
    check_stages: list[Stage] = []
    for index, assertion in enumerate(assertions):
        envelope = check_envelope(
            binding,
            versions,
            "library",
            (assertion,),
            prerequisite_output_sha256s=(ingest.output_sha256,),
            reference_constraints=constraints,
        )
        stage = save_stage(
            root,
            run_id=run_id,
            stage_id=run_id + f"-check-{index}",
            stage="analyze",
            binding=binding,
            versions=versions,
            envelope=envelope,
            prerequisite_ids=(ingest.stage_id,),
        )
        stages.append(stage.stage_id)
        check_stages.append(stage)
        if index == 0:
            assertion_result, assertion_objections = result, objections
        elif index == 1:
            assertion_result, assertion_objections = sense_result, sense_objections
        else:
            assertion_result, assertion_objections = "not_applicable", ("requires_separate_assertion_check",)
        checks.append(
            Check(
                id=f"check-{assertion.id}",
                assertion_ids=(assertion.id,),
                method="library",
                version=CHECKER,
                result=assertion_result,
                evidence=constraints,
                alternatives=sense_alternatives if index == 1 else alternatives,
                objections=assertion_objections,
                input_sha256=digest(envelope),
            )
        )
    report = {
        "format": "latinitas-deck-analysis",
        "version": 1,
        "source": selected,
        "scope": SCOPE,
        "library_version": library.version,
        "selected_sense": selected_sense.model_dump(mode="json") if selected_sense else None,
        "state": state,
        "reason": reason,
        "checks": [c.model_dump(mode="json") for c in checks],
        "prerequisite_output_sha256s": tuple(s.output_sha256 for s in check_stages),
    }
    summary = save_stage(
        root,
        run_id=run_id,
        stage_id=run_id + "-report",
        stage="analyze",
        binding=binding,
        versions=versions,
        envelope=report,
        prerequisite_ids=tuple(s.stage_id for s in check_stages),
    )
    stages.append(summary.stage_id)
    decision = Acceptance.create(
        binding=binding,
        versions=versions,
        assertions=assertions,
        checks=tuple(checks),
        origin="automatic",
        state="failed" if state == "unavailable" else "withheld",
        human=None,
        stage_ids=tuple(stages),
        reason="processing_unavailable" if state == "unavailable" else "policy_not_enabled",
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    return DeckAnalysis(state, tuple(checks), summary, decision)


def analyze_construction(
    root: Path,
    *,
    pattern: Pattern,
    components: dict[str, str],
    versions: Versions,
) -> DeckAnalysis:
    """Check four bounded families using persisted, source-bound constituents.

    A composite ingest retains each original binding and complete stage report.
    It does not pretend heterogeneous source bindings are same-binding stage
    prerequisites, nor inherit any constituent's decision as export authority.
    """
    if versions.checker != CHECKER:
        raise ValueError("checker version mismatch")
    policy_reason(versions, "1", ())
    if set(components) != set(pattern.roles):
        raise ValueError("constituent roles do not match pattern")
    snapshots: dict[str, object] = {}
    source_bindings: dict[str, Binding] = {}
    forms: dict[str, Form] = {}
    senses: dict[str, Sense] = {}
    issues: list[str] = []
    unavailable = False
    for role, identity in sorted(components.items()):
        stage = load_stage(root, identity)
        if stage.versions != versions:
            raise ValueError("stale constituent versions")
        reusable_stage(root, identity, binding=stage.binding, versions=versions, input_sha256=stage.input_sha256)
        report = stage_artifact(root, stage)
        if not isinstance(report, dict) or report.get("format") != "latinitas-deck-analysis":
            raise ValueError("expected deck analysis constituent")
        source = MappedSource.read_json(canonical(report["source"]))
        if digest(source.task) != stage.binding.task_sha256 or digest(source) != stage.binding.extraction_sha256:
            raise ValueError("constituent source/task binding mismatch")
        snapshots[role] = {"stage": stage, "report": report}
        source_bindings[role] = stage.binding
        checks = tuple(Check.read_json(canonical(c)) for c in report["checks"])
        if len(checks) < 2 or any(c.result != "pass" for c in checks[:2]):
            issues.append(f"{role}: constituent_evidence_unresolved")
            unavailable |= report["state"] == "unavailable"
            continue
        sense = Sense.model_validate_json(canonical(report["selected_sense"]))
        requirement = pattern.roles[role]
        if (
            source.task.pos != requirement.pos
            or f"{stage.binding.source_identity}/{sense.key}" not in requirement.senses
        ):
            issues.append(f"{role}: incompatible_bound_sense")
        if source.task.preposition is not None and (
            pattern.family != "preposition-noun"
            or role != "object"
            or pattern.function_words != (source.task.preposition,)
        ):
            issues.append(f"{role}: constituent_context_mismatch")
        forms[role] = Form(key=role, text=source.form, features=source.task.features)
        senses[role] = sense
    result: Literal["pass", "fail", "unresolved", "not_applicable"]
    if issues:
        result, state = "unresolved", "unavailable" if unavailable else "unresolved"
    else:
        issues, unsupported = _construction_issues(pattern, forms, senses)
        if unsupported:
            result, state = "not_applicable", "unsupported"
        elif issues:
            result, state = "fail", "contradiction"
        else:
            result, state = "pass", "corroborated"
    source_data: dict[str, object] = {"pattern": pattern.model_dump(mode="json"), "constituents": snapshots}
    languages = {b.language for b in source_bindings.values()}
    if len(languages) != 1:
        raise ValueError("constituent language mismatch")
    binding = Binding(
        source_scope="deck-construction/1",
        source_identity="construction-"
        + digest({r: (b.source_scope, b.source_identity) for r, b in source_bindings.items()}),
        source_sha256=digest({r: b.source_sha256 for r, b in source_bindings.items()}),
        extraction_sha256=digest(source_data),
        mapping_sha256=digest({r: b.mapping_sha256 for r, b in source_bindings.items()}),
        task_sha256=digest(pattern),
        language=next(iter(languages)),
        config_sha256=digest({"checker": CHECKER, "family": pattern.family}),
    )
    evidence = (Evidence("latinitas-latin-rules", f"{pattern.family}/1", CHECKER, canonical(source_data).decode()),)
    assertion = Assertion(
        id="construction",
        kind="label",
        value=canonical(pattern.model_dump(mode="json")).decode(),
        selected_sense=None,
        alternatives=(),
        evidence=evidence,
    )
    run_id = "construction-" + digest({"binding": binding, "versions": versions, "result": result, "issues": issues})
    ingest = save_stage(
        root,
        run_id=run_id,
        stage_id=run_id + "-source",
        stage="ingest",
        binding=binding,
        versions=versions,
        envelope=source_data,
    )
    assert ingest.output_sha256 is not None
    envelope = check_envelope(
        binding,
        versions,
        "rule",
        (assertion,),
        prerequisite_output_sha256s=(ingest.output_sha256,),
        reference_constraints=evidence,
    )
    check_stage = save_stage(
        root,
        run_id=run_id,
        stage_id=run_id + "-check",
        stage="analyze",
        binding=binding,
        versions=versions,
        envelope=envelope,
        prerequisite_ids=(ingest.stage_id,),
    )
    check = Check(
        id="check-construction",
        assertion_ids=(assertion.id,),
        method="rule",
        version=CHECKER,
        result=result,
        evidence=evidence,
        alternatives=(),
        objections=tuple(issues),
        input_sha256=digest(envelope),
    )
    report = {
        "format": "latinitas-construction-analysis",
        "version": 1,
        "source": source_data,
        "state": state,
        "checks": (check,),
        "prerequisite_output_sha256s": (check_stage.output_sha256,),
    }
    summary = save_stage(
        root,
        run_id=run_id,
        stage_id=run_id + "-report",
        stage="analyze",
        binding=binding,
        versions=versions,
        envelope=report,
        prerequisite_ids=(check_stage.stage_id,),
    )
    decision = Acceptance.create(
        binding=binding,
        versions=versions,
        assertions=(assertion,),
        checks=(check,),
        origin="automatic",
        state="failed" if unavailable else "withheld",
        human=None,
        stage_ids=(ingest.stage_id, check_stage.stage_id, summary.stage_id),
        reason="processing_unavailable" if unavailable else "policy_not_enabled",
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    return DeckAnalysis(state, (check,), summary, decision)


def _construction_issues(
    pattern: Pattern,
    forms: dict[str, Form],
    senses: dict[str, Sense],
) -> tuple[list[str], bool]:
    """Return rule objections and whether the construction lacks support."""
    issues: list[str] = []
    for role, form in forms.items():
        for feature, value in pattern.roles[role].requires.model_dump(exclude_none=True).items():
            if getattr(form.features, feature) != value:
                issues.append(f"{role}: requires {feature}={value}")
    issues.extend(f"{code}: {message}" for code, message in construction_diagnostics(pattern, forms, senses))
    unsupported = False
    if pattern.family == "preposition-noun":
        government = GOVERNMENT.get(pattern.function_words[0]) if len(pattern.function_words) == 1 else None
        if government is None:
            unsupported = True
            issues.append("unsupported_or_ambiguous_government")
        elif pattern.roles["object"].requires.case != government or forms["object"].features.case != government:
            issues.append("government_contradiction")
    elif pattern.function_words:
        unsupported = True
        issues.append("unsupported_function_words")
    if pattern.family == "verb-noun" and forms["verb"].features.voice != "active":
        unsupported = True
        issues.append("unsupported_voice_specific_valency")
    return issues, unsupported


def _sense(
    task: DeckTask,
    meaning: str,
    senses: tuple[Sense, ...],
) -> tuple[Literal["pass", "fail", "unresolved", "not_applicable"], tuple[str, ...], Sense | None]:
    """Check only explicitly referenced lexical glosses, not rendered translation equivalence."""
    if not senses:
        return "not_applicable", ("missing_sense_corroboration",), None
    references: dict[str, set[str]] = {}
    for sense in senses:
        references.setdefault(sense.key, set()).add(
            canonical({"meaning": sense.meaning, "valency": sense.valency}).decode()
        )
    if any(len(propositions) > 1 for propositions in references.values()):
        return "unresolved", ("inconsistent_sense_reference",), None
    matching = [s for s in senses if s.meaning == meaning and s.evidence]
    if not matching or task.selected_sense is None:
        return "unresolved", ("unresolved_source_meaning_or_sense",), None
    if len({s.key for s in matching}) != 1:
        return "unresolved", ("ambiguous_source_sense",), None
    if matching[0].key != task.selected_sense:
        return "fail", ("selected_sense_contradicts_reference",), None
    return "pass", (), matching[0]


def _corroborate(
    task: DeckTask,
    lemma: str,
    readings: tuple[LibraryReading, ...],
    *,
    exhaustive: bool,
) -> tuple[str, Literal["pass", "fail", "unresolved", "not_applicable"], tuple[str, ...]]:
    if not readings:
        return "unsupported", "not_applicable", ("missing_corroboration_or_unsupported_paradigm",)
    required = ("person", "number", "tense", "mood", "voice") if task.pos == "verb" else ("case", "number", "gender")
    if not exhaustive or any(any(getattr(r.features, k) is None for k in required) for r in readings):
        return "unresolved", "unresolved", ("incomplete_analyzer_or_ambiguous_form_or_sense",)
    if not any(r.lemma == lemma and r.pos == task.pos and r.features == task.features for r in readings):
        return "contradiction", "fail", ("morphology_or_lemma_contradiction",)
    remaining = readings
    if task.preposition is not None:
        case = GOVERNMENT.get(task.preposition)
        if case is None:
            return "unresolved", "unresolved", ("ambiguous_government_or_unsupported_preposition",)
        if task.pos != "noun" or task.features.case != case:
            return "contradiction", "fail", ("government_contradiction",)
        remaining = tuple(r for r in readings if r.features.case == case)
    if any(r.lemma != lemma or r.pos != task.pos or r.features != task.features for r in remaining):
        return "unresolved", "unresolved", ("ambiguous_form_or_sense",)
    return "corroborated", "pass", ()
