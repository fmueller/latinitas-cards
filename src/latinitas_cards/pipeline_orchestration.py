"""Bounded pipeline-local execution over the existing evidence and batch stores.

Transport results are inspectable hypotheses. Neither agreement nor this runner
supplies missing class-specific evidence, human review, or curated approval.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Literal, cast
from uuid import uuid4

from pydantic import JsonValue

from .acceptance import (
    _immutable,
    check_envelope,
    load_stage,
    reusable_stage,
    save_stage,
    save_stage_record,
    stage_artifact,
)
from .acceptance_policy import SupportRegistry, implementation_digest
from .acceptance_records import (
    Assertion,
    Binding,
    Check,
    Record,
    Stage,
    StageName,
    Text,
    Versions,
    canonical,
    decode_json,
    digest,
)
from .acceptance_registry import SUPPORT_REGISTRY
from .claim_review import Evidence
from .deck_evidence import CHECKER, DeckTask, analyze_deck
from .html_text import source_html_to_text
from .latin_library import LocalStanza
from .phrase_inputs import Features
from .pipeline_batches import _atomic, run_lock
from .pipeline_consumers import (
    FORM_ASSERTION_VALUE_CONTRACTS,
    FORM_DIRECTIONS,
    FORM_PRESENTATION,
    FORM_RENDERED_CONTRACTS,
    Analysis,
    FormObjective,
    Objectives,
    Rendered,
    Run,
    Source,
    _cell,
    _decide,
    _entry,
    _export,
    _feature_terms,
    _features,
    _form,
    _versions,
    current_cards,
    load_run,
    load_source,
    preview,
    rule_checks,
)
from .pipeline_profile import PipelineProfile, load_pipeline_profile
from .pipeline_reporting import build_report, safe_snapshot
from .providers import (
    PROMPT_VERSIONS,
    Endpoint,
    HostedPermission,
    HTTPTransport,
    Limits,
    ModelTask,
    ProviderRun,
    Role,
    Scope,
    Span,
    compare_answers,
)
from .semantic_comparison import compare_source_answers

STAGES: tuple[StageName, ...] = (
    "ingest",
    "analyze",
    "propose",
    "counter_check",
    "decide",
    "select",
    "render",
    "export",
)
PREREQUISITES: dict[StageName, StageName] = {"analyze": "ingest", "propose": "analyze", "counter_check": "propose"}
COVERAGE = (
    "Only supplied, confirmed source cells and finite rule recipes are enumerated. "
    "Independent Bennett diagnostics cover explicitly attested form/component/context cells and "
    "constructed phrases over five selected bilingual lexical readings, "
    "not arbitrary paradigms or exhaustive interpretation. "
    "Finite reviewed meaning/clause projections cover only their exact constructed cells and utterance vocabulary. "
    "Library morphology alone is not rendered-clause/translation entailment. Unsupported assertions remain withheld; "
    "model agreement grants no authority. Human reviews remain separately bound; curated approvals require T196."
)


class Execution(Record):
    format: Literal["latinitas-pipeline-execution"] = "latinitas-pipeline-execution"
    version: Literal[1] = 1
    run_id: Text
    ledger_config_sha256: Text
    output: Text
    library_resources: Text | None
    offline: bool = False
    local_only: bool = False


class RequestContext(Record):
    endpoint: Endpoint
    task: ModelTask
    recipe: Text
    scope: tuple[Scope, ...]


class Cancellation(Event):
    """The cancel command can interrupt HTTP polling without taking the writer lock."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.path = root / "cancelled"

    def is_set(self) -> bool:
        return super().is_set() or self.path.exists()


def load_execution(root: Path) -> Execution:
    return Execution.read_json((root / "execution.json").read_bytes())


def _config(profile: PipelineProfile) -> str:
    # Input commitments are checked per request, so editing a sibling does not
    # invalidate unrelated completed calls. Permissions still bind exact bodies.
    return digest(
        {
            "language": profile.language,
            "layout": profile.layout,
            "policy": profile.policy,
            "checker": implementation_digest(),
        }
    )


def _provider(root: Path, state: Execution, profile: PipelineProfile) -> ProviderRun:
    return ProviderRun(
        root / "providers",
        run_id=state.run_id,
        config_sha256=_config(profile),
        ledger_config_sha256=state.ledger_config_sha256,
        proposal=profile.providers.proposal,
        counter_check=profile.providers.counter_check,
        limits=Limits(**profile.processing.model_dump(include=set(Limits.model_fields))),
        accounting=profile.accounting,
    )


def prepare(
    profile_path: Path,
    output: Path,
    objectives_path: Path | None = None,
    library_resources: Path | None = None,
    *,
    offline: bool = False,
    local_only: bool = False,
) -> Path:
    if output.exists():
        raise ValueError("choose a new output directory; use resume for an existing run")
    profile = load_pipeline_profile(profile_path)
    objectives = (
        Objectives.read_json(objectives_path.read_bytes())
        if objectives_path
        else Objectives(format="latinitas-grammar-objectives", version=1, forms=(), phrases=())
    )
    root = output / "run"
    current_cards(profile_path, profile, objectives, root)
    # Validate operational hard limits before allocating a run or dispatching.
    Limits(**profile.processing.model_dump(include=set(Limits.model_fields)))
    if local_only and any(
        e and e.locality != "local" for e in (profile.providers.proposal, profile.providers.counter_check)
    ):
        raise ValueError("local_only requires all endpoints and upstream processing to be local")
    output.parent.mkdir(parents=True, exist_ok=True)
    # No public run exists until both manifests and its empty spending ledger
    # are committed. A published run must never recreate a missing ledger.
    with TemporaryDirectory(prefix=".pipeline-initializing-", dir=output.parent) as temporary:
        staging = Path(temporary) / "output"
        staged_root = staging / "run"
        staging.mkdir(mode=0o700)
        staged_root.mkdir(mode=0o700)
        run_id = uuid4().hex
        _atomic(
            staged_root / "run.json",
            canonical(
                Run(
                    format="latinitas-grammar-run",
                    version=1,
                    profile_path=str(profile_path.resolve()),
                    profile=profile,
                    objectives=objectives,
                )
            ),
        )
        state = Execution(
            run_id=run_id,
            ledger_config_sha256=digest({"run_id": run_id}),
            output=str(output.resolve()),
            library_resources=str(library_resources.resolve()) if library_resources else None,
            offline=offline,
            local_only=local_only,
        )
        _atomic(staged_root / "execution.json", canonical(state))
        _atomic(staged_root / "accounting-custody.json", canonical(profile.accounting))
        _provider(staged_root, state, profile).initialize()
        staging.rename(output)
    return root


def _task(card: Rendered, source: Source, profile: PipelineProfile) -> tuple[ModelTask, tuple[Scope, ...]]:
    objective = card.objective
    bindings = (
        ((objective.entry, objective.sense, objective.form, objective.anchors),)
        if isinstance(objective, FormObjective)
        else tuple((b.entry, b.sense, b.form, ()) for b in objective.selection.bindings)
    )
    evidence: list[Evidence] = []
    scopes: list[Scope] = []
    for identity, sense_key, form_key, anchors in bindings:
        entry, sense = _entry(source, identity, sense_key)
        record = source.records[identity]
        fields = source.deck.fields
        selected = []
        lemma = source_html_to_text(record.fields.get(fields.lexical_entry_field, "")).strip()
        if fields.meaning_field is None:
            raise ValueError("confirmed meaning mapping required")
        cells = [
            _cell(record, (fields.lexical_entry_field,), lemma),
            _cell(record, (fields.meaning_field,), sense.meaning),
        ]
        for key in (form_key,) if form_key else anchors:
            form = _form(source, entry, key)
            cells.append(_cell(record, (fields.lexical_entry_field, fields.principal_parts_field), form.text))
        for cell in dict.fromkeys(cells):
            text = cell.extract(record)
            extracted = " ".join(source_html_to_text(record.fields[cell.field]).split())
            start = extracted.find(text)
            if start < 0:
                raise ValueError("mapped span is absent from extracted field")
            selected.append({"field": cell.field, "start": start, "end": start + len(text), "text": text})
            scopes.append(
                Scope(
                    source_identity=identity,
                    fields=(cell.field,),
                    spans=(Span(start=start, end=start + len(text)),),
                    language=profile.language,
                )
            )
        evidence.append(
            Evidence("confirmed-deck-mapping", source.scope + "/" + identity, "1", canonical(selected).decode())
        )
        supplied = canonical({"entry": entry, "identity": asdict(source.identities[identity])}).decode()
        evidence.append(
            Evidence(
                "unverified-form-entry" if isinstance(objective, FormObjective) else "unverified-phrase-entry",
                source.scope + "/" + identity,
                "1",
                supplied,
            )
        )
        scopes.append(
            Scope(
                source_identity=identity,
                fields=("phrase-input/entry",),
                spans=(Span(start=0, end=len(supplied)),),
                language=profile.language,
            )
        )
    # Bindings that reuse one entry share its identical evidence and scopes instead of repeating them.
    evidence, scopes = list(dict.fromkeys(evidence)), list(dict.fromkeys(scopes))
    registry = SupportRegistry.read_json(canonical(SUPPORT_REGISTRY))
    keys = next(e.target for e in registry.entries if e.recipe == card.recipe)
    features = objective.target if isinstance(objective, FormObjective) else None
    if (
        isinstance(objective, FormObjective)
        and objective.form
        and objective.recipe not in ("verb_production", "noun_declension", "adjective_declension")
    ):
        features = _form(source, _entry(source, objective.entry, objective.sense)[0], objective.form).features
    elif not isinstance(objective, FormObjective) and objective.target:
        selected_binding = next(b for b in objective.selection.bindings if b.role == objective.target)
        features = _form(
            source, _entry(source, selected_binding.entry, selected_binding.sense)[0], selected_binding.form
        ).features
    feature_values = cast(dict[str, JsonValue], features.model_dump(exclude_none=True)) if features else {}
    values: dict[str, JsonValue] = {
        "meaning": profile.language,
        "features": feature_values,
        "context": objective.context if isinstance(objective, FormObjective) else "selected controlled phrase",
        **feature_values,
    }
    target: dict[str, JsonValue] = {key: values[key] for key in keys}
    binding = card.binding.model_copy(update={"config_sha256": _config(profile)})
    if isinstance(objective, FormObjective):
        assert features is not None
        direction = FORM_DIRECTIONS[card.recipe]
        contracts = {**FORM_ASSERTION_VALUE_CONTRACTS, "rendered": FORM_RENDERED_CONTRACTS[direction]}
        request_objective = canonical(
            {
                "recipe": card.recipe,
                "direction": direction,
                "selected_form": objective.form,
                "selected_sense": objective.sense,
                "anchors": objective.anchors,
                "context": objective.context,
                "distinction": objective.distinction,
                "presentation": _presentation(card, source, profile.language, features),
                "feature_terms": _feature_terms(source.deck),
                "feature_order": tuple(Features.model_fields),
                "assertions": tuple(
                    {
                        "id": a.id,
                        "kind": a.kind,
                        "value_contract": contracts.get(
                            a.id,
                            "Plain translated text for the selected sense."
                            if a.kind == "translation"
                            else "Plain explanatory prose, independently supported by evidence.",
                        ),
                    }
                    for a in card.claims.assertions
                ),
            }
        ).decode()
    else:
        pattern = next(p for p in source.data.patterns if p.key == objective.selection.pattern)
        supplied_pattern = canonical(pattern).decode()
        evidence.append(
            Evidence(
                "unverified-phrase-pattern", source.scope + "/" + card.binding.source_identity, "1", supplied_pattern
            )
        )
        scopes.append(
            Scope(
                source_identity=card.binding.source_identity,
                fields=("phrase-input/pattern",),
                spans=(Span(start=0, end=len(supplied_pattern)),),
                language=profile.language,
            )
        )
        contexts = objective.reference.contexts if card.recipe == "phrase_translation" else ()
        for context in contexts:
            evidence.append(
                Evidence("unverified-phrase-context", source.scope + "/" + card.binding.source_identity, "1", context)
            )
            scopes.append(
                Scope(
                    source_identity=card.binding.source_identity,
                    fields=("phrase-objective/context",),
                    spans=(Span(start=0, end=len(context)),),
                    language=profile.language,
                )
            )
        request_objective = canonical(
            {
                "recipe": card.recipe,
                "family": pattern.family,
                "function_words": pattern.function_words,
                "bindings": objective.selection.bindings,
                "contexts": contexts,
                "target": objective.target,
                "distinction": objective.distinction,
                "assertions": tuple({"id": a.id, "kind": a.kind} for a in card.claims.assertions),
            }
        ).decode()
    task = ModelTask(
        format="latinitas-model-task",
        version=1,
        task_sha256=binding.task_sha256,
        binding=binding,
        objective=request_objective,
        target=target,
        source_evidence=tuple(evidence),
        reference_constraints=(),
    )
    task.validate_recipe(card.recipe)
    return task, tuple(scopes)


def _presentation(card: Rendered, source: Source, language: str, features: Features) -> dict[str, str]:
    """Only the direction's templates, with values the task already fixes filled in."""
    objective = cast(FormObjective, card.objective)
    direction = FORM_DIRECTIONS[card.recipe]
    templates = FORM_PRESENTATION[language]

    def literal(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    prompt = templates[direction + "_prompt"].replace("{features}", literal(_features(features, source.deck)))
    if objective.context:
        placeholder = "{context}" if direction == "cloze" else "{context_or_surface}"
        prompt = prompt.replace(placeholder, literal(objective.context))
    entry = _entry(source, objective.entry, objective.sense)[0]
    if direction == "production" and objective.form and _form(source, entry, objective.form).source_role == "ppp":
        prompt += "\n" + literal(templates["ppp_suffix"])
    presentation = {direction + "_prompt": prompt}
    if direction == "translation":
        presentation["translation_answer"] = templates["translation_answer"]
    if any(a.id == "stem" for a in card.claims.assertions):
        presentation["stem_line"] = templates["stem_line"]
    return presentation


def _stage(root: Path, state: Execution, card: Rendered, name: StageName, envelope: dict[str, object]) -> str:
    identity = digest({"stage": name, "binding": card.binding, "versions": card.versions, "envelope": envelope})
    return save_stage(
        root,
        run_id=state.run_id,
        stage_id=identity,
        stage=name,
        binding=card.binding,
        versions=card.versions,
        envelope=envelope,
    ).stage_id


def _analyze(root: Path, state: Execution, card: Rendered, source: Source) -> dict[str, object]:
    library = LocalStanza(Path(state.library_resources)) if state.library_resources else None
    independent = rule_checks(source, card, root)
    context = digest(
        (card.binding, card.versions, card.claims.assertions, library.version if library else None, independent)
    )
    metadata_path = root / "analysis-contexts" / card.analysis_path.name
    if metadata_path.exists():
        metadata = decode_json(metadata_path.read_bytes())
        if not isinstance(metadata, dict):
            raise ValueError("invalid analysis context")
        if metadata.get("context") == context:
            if hashlib.sha256(card.analysis_path.read_bytes()).hexdigest() != metadata["analysis_digest"]:
                raise ValueError("analysis artifact digest mismatch")
            old = Analysis.read_json(card.analysis_path.read_bytes())
            for identity in old.stage_ids:
                stage_artifact(root, load_stage(root, identity))
            cached = cast(dict[str, object], metadata["outcome"])
            if cached.get("state") != "failed":
                return {**cached, "_reused": True}
    constraints: tuple[Evidence, ...] = ()
    library_state = "not_configured"
    if library and isinstance(card.objective, FormObjective) and card.objective.form:
        objective = card.objective
        entry, sense = _entry(source, objective.entry, objective.sense)
        assert objective.form is not None
        form = _form(source, entry, objective.form)
        meaning_field = source.deck.fields.meaning_field
        if meaning_field is None:
            raise ValueError("confirmed meaning mapping required")
        try:
            analysis = analyze_deck(
                root,
                record=source.records[entry.identity],
                profile=source.deck,
                source_scope=source.scope,
                task=DeckTask(
                    form=_cell(
                        source.records[entry.identity],
                        (source.deck.fields.principal_parts_field, source.deck.fields.lexical_entry_field),
                        form.text,
                    ),
                    meaning=_cell(source.records[entry.identity], (meaning_field,), sense.meaning),
                    pos=entry.pos,
                    features=form.features,
                    source_role=form.source_role,
                    selected_sense=sense.key,
                    preposition=None,
                    extra_assertions=(),
                ),
                versions=card.versions.model_copy(update={"analyzer": library.version, "checker": CHECKER}),
                library=library,
            )
            library_state = analysis.state
            constraints = tuple(dict.fromkeys(e for check in analysis.checks for e in check.evidence))
        except ValueError:
            library_state = "unsupported_library_task"
    stages, checks = [], []
    for assertion, rule_check in zip(card.claims.assertions, independent, strict=True):
        evidence = tuple(dict.fromkeys((*rule_check.evidence, *constraints)))
        envelope = check_envelope(card.binding, card.versions, "rule", (assertion,), reference_constraints=evidence)
        stages.append(_stage(root, state, card, "analyze", envelope))
        check_result, objections = rule_check.result, rule_check.objections
        if library_state == "unavailable":
            check_result, objections = "unresolved", ("processing_unavailable",)
        elif check_result == "pass" and library_state == "contradiction":
            check_result, objections = "fail", ("library_contradiction",)
        checks.append(
            Check(
                id=assertion.id,
                assertion_ids=(assertion.id,),
                method="rule",
                version=card.versions.checker,
                result=check_result,
                evidence=evidence,
                alternatives=(),
                objections=objections,
                input_sha256=digest(envelope),
            )
        )
    result = Analysis(
        format="latinitas-grammar-analysis",
        version=1,
        binding=card.binding,
        versions=card.versions,
        assertions=card.claims.assertions,
        checks=tuple(checks),
        stage_ids=tuple(stages),
    )
    card.analysis_path.parent.mkdir(exist_ok=True, mode=0o700)
    # Independent analyses still pass through the full public decision reader.
    if card.analysis_path.exists() and not metadata_path.exists():
        old = Analysis.read_json(card.analysis_path.read_bytes())
        if old.binding == card.binding and old.versions == card.versions and old.assertions == card.claims.assertions:
            for stage_id in old.stage_ids:
                stage = load_stage(root, stage_id)
                if stage.binding != card.binding or stage.versions != card.versions:
                    raise ValueError("stale analysis stage context")
                stage_artifact(root, stage)
            result = old
    _atomic(card.analysis_path, canonical(result))
    outcome: dict[str, object] = {
        "analysis": str(card.analysis_path),
        "library": library_state,
        "coverage_limit": COVERAGE,
    }
    if library_state == "unavailable":
        outcome.update(state="failed", reason="processing_unavailable")
    metadata_path.parent.mkdir(exist_ok=True, mode=0o700)
    _atomic(
        metadata_path,
        canonical(
            {
                "context": context,
                "analysis_digest": hashlib.sha256(card.analysis_path.read_bytes()).hexdigest(),
                "outcome": outcome,
            }
        ),
    )
    return outcome


def _summary(
    root: Path,
    state: Execution,
    source: Source,
    profile: PipelineProfile,
    name: StageName,
    rows: list[dict[str, object]],
    *,
    binding: Binding | None = None,
    prerequisites: tuple[str, ...] = (),
) -> str:
    binding = binding or Binding.model_validate(source.data.binding)
    versions = _versions(profile)
    envelope = {
        "effective_profile": profile,
        "outcomes": [{key: value for key, value in row.items() if key != "reuse"} for row in rows],
        "prerequisite_output_sha256s": tuple(load_stage(root, p).output_sha256 for p in prerequisites),
    }
    identity = digest({"stage": name, "binding": binding, "versions": versions, "envelope": envelope})
    failure = next((r for r in rows if r.get("state") in ("unknown", "failed", "deferred", "cancelled")), None)
    if failure is not None:
        status = cast(Literal["unknown", "failed", "deferred", "cancelled"], failure["state"])
        save_stage_record(
            root,
            Stage(
                run_id=state.run_id,
                stage_id=identity,
                stage=name,
                binding=binding,
                versions=versions,
                prerequisite_ids=prerequisites,
                state=status,
                input_sha256=digest(envelope),
                output_sha256=None,
                artifact=None,
                call_ids=tuple(str(r["call_id"]) for r in rows if r.get("call_id")),
                reason=str(failure.get("reason", status)),
                started_at=None,
                finished_at=None,
            ),
        )
    else:
        save_stage(
            root,
            run_id=state.run_id,
            stage_id=identity,
            stage=name,
            binding=binding,
            versions=versions,
            envelope=envelope,
            prerequisite_ids=prerequisites,
        )
    return identity


def _card_key(card: Rendered, name: StageName) -> str:
    return name + "/" + digest((card.claims.note_id, card.claims.card_key))


def _model_envelope(
    root: Path, card: Rendered, role: Role, *, max_output_tokens: int | None = None
) -> dict[str, object]:
    saved = load_run(root)
    profile = saved.profile
    if max_output_tokens is not None:
        profile = profile.model_copy(
            update={"processing": profile.processing.model_copy(update={"max_output_tokens": max_output_tokens})}
        )
    source = load_source(Path(saved.profile_path), profile)
    task, scope = _task(card, source, profile)
    provider = _provider(root, load_execution(root), profile)
    result = provider.completed(role, task, scope)
    if result is None or result.answer is None:
        raise ValueError("missing successful proposal-free invocation")
    envelope: dict[str, object] = {"task": task, "result": result.answer, "call_id": result.call_id}
    if role == "proposal":
        return envelope
    proposal = provider.completed("proposal", task, scope)
    if proposal is None or proposal.answer is None:
        raise ValueError("missing successful proposal-free invocations")
    calls = {call.call_id: call for call in provider.calls()}
    assert proposal.call_id is not None and result.call_id is not None
    envelope["semantic_comparison"] = compare_source_answers(
        source, card, task, proposal.answer, result.answer, calls[proposal.call_id], calls[result.call_id]
    )
    return envelope


def read_source_comparison(root: Path, card: Rendered) -> dict[str, object] | None:
    """Rebuild a persisted proof from current sources and actual saved calls."""
    if card.versions.proposal_prompt is None or card.versions.counter_prompt is None:
        return None
    path = root / "stage-index.json"
    if not path.exists():
        return None
    index = decode_json(path.read_bytes())
    if not isinstance(index, dict):
        raise ValueError("stage index must be an object")
    identity = _prerequisite(root, index, card, "counter_check")
    if identity is None:
        return None
    wrapper_stage = load_stage(root, identity)
    wrapper = cast(dict[str, object], stage_artifact(root, wrapper_stage))
    for row in cast(list[dict[str, object]], wrapper.get("outcomes", [])):
        stage_id = row.get("stage_id")
        if not isinstance(stage_id, str):
            continue
        if stage_id not in wrapper_stage.prerequisite_ids:
            raise ValueError("uncommitted model outcome reference")
        model_stage = load_stage(root, stage_id)
        envelope = stage_artifact(root, model_stage)
        expected = _model_envelope(root, card, "counter_check")
        if model_stage.stage != "counter_check" or canonical(envelope) != canonical(expected):
            raise ValueError("source comparison proof mismatch")
        return cast(dict[str, object], expected["semantic_comparison"])
    # LLM-off stages have no model artifact and supply no semantic proof.
    return None


def read_acceptance_model_checks(
    root: Path, binding: Binding, versions: Versions, assertions: tuple[Assertion, ...]
) -> tuple[tuple[Check, ...], tuple[str, ...]]:
    """Reconstruct the complete original card and both actual invocations.

    The original whole-task blind check cannot be projected to a subset.
    Returning verified records does not make unresolved siblings eligible and
    does not change the diagnostic artifact's lack of acceptance authority.
    """
    if versions.proposal_prompt is None or versions.counter_prompt is None:
        return (), ()
    saved = load_run(root)
    cards, _ = current_cards(Path(saved.profile_path), saved.profile, saved.objectives, root)
    matches = [
        card
        for card in cards
        if card.binding == binding and card.versions == versions and card.claims.assertions == assertions
    ]
    if len(matches) != 1:
        raise ValueError("model evidence requires the complete current card")
    card = matches[0]
    proof = read_source_comparison(root, card)
    if proof is None:
        return (), ()
    comparisons = cast(list[dict[str, object]], proof["comparisons"])
    checks = tuple(
        Check.read_json(canonical(check)) for check in (proof["blind_check"], *(c["check"] for c in comparisons))
    )
    index = cast(dict[str, str], decode_json((root / "stage-index.json").read_bytes()))
    return checks, (index[_card_key(card, "counter_check")],)


def _prerequisite(root: Path, index: dict[str, JsonValue], card: Rendered, name: StageName) -> str | None:
    identity = index.get(_card_key(card, name))
    if not isinstance(identity, str):
        return None
    stage = load_stage(root, identity)
    if stage.stage != name:
        raise ValueError("indexed prerequisite stage type mismatch")
    if stage.state != "succeeded" or stage.binding != card.binding or stage.versions != card.versions:
        return None
    reusable_stage(root, identity, binding=card.binding, versions=card.versions, input_sha256=stage.input_sha256)
    # Wrapper integrity alone does not validate the mutable analysis files it
    # commits. Follow ancestors even when their IDs left the current index.
    capacity = load_run(root).profile.processing.max_output_tokens
    stale = False
    pending, visited = [(stage, capacity)], set[str]()
    while pending:
        current, historical_capacity = pending.pop()
        if current.stage_id in visited:
            continue
        visited.add(current.stage_id)
        envelope = cast(dict[str, object], stage_artifact(root, current))
        if current.stage in ("propose", "counter_check"):
            if set(envelope) == {"effective_profile", "outcomes", "prerequisite_output_sha256s"}:
                historical_capacity = PipelineProfile.read_json(
                    canonical(envelope["effective_profile"])
                ).processing.max_output_tokens
                stale |= historical_capacity != capacity
                for outcome in cast(list[dict[str, object]], envelope["outcomes"]):
                    if "stage_id" in outcome and outcome["stage_id"] not in current.prerequisite_ids:
                        raise ValueError("uncommitted model outcome reference")
            else:
                role: Role = "proposal" if current.stage == "propose" else "counter_check"
                if canonical(envelope) != canonical(
                    _model_envelope(root, card, role, max_output_tokens=historical_capacity)
                ):
                    raise ValueError("source comparison or model stage proof mismatch")
        for row in cast(list[dict[str, object]], envelope.get("outcomes", [])):
            if "analysis" not in row:
                continue
            data = Path(str(row["analysis"])).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["analysis_sha256"]:
                raise ValueError("analysis prerequisite digest mismatch")
            analysis = Analysis.read_json(data)
            assertions = {a.id: a for a in analysis.assertions}
            if (
                analysis.binding != card.binding
                or analysis.versions != card.versions
                or any(assertions.get(a.id) != a for a in card.claims.assertions)
            ):
                raise ValueError("stale analysis prerequisite")
            # Independent proofs retain their own run identities; they are
            # separately validated roots, not foreign orchestration children.
            for proof_id in analysis.stage_ids:
                proof = load_stage(root, proof_id)
                reusable_stage(
                    root, proof_id, binding=card.binding, versions=card.versions, input_sha256=proof.input_sha256
                )
        pending.extend((load_stage(root, p), historical_capacity) for p in current.prerequisite_ids)
    return None if stale else identity


def cancel(root: Path) -> dict[str, object]:
    state = load_execution(root)
    _atomic(root / "cancelled", b"cancelled\n")
    return {"run_id": state.run_id, "status": "cancelled", "applied_to_anki": False}


def execute(
    root: Path,
    *,
    profile_path: Path | None = None,
    only: StageName | None = None,
    permission_path: Path | None = None,
    offline: bool = False,
    local_only: bool = False,
    continue_cancelled: bool = False,
    retry_unknown: bool = False,
    confirm_duplicate_billing: bool = False,
    disclose: Callable[[str], None] | None = None,
    reconcile_receipt: str | None = None,
) -> dict[str, object]:
    with run_lock(root):
        state = load_execution(root)
        saved = load_run(root)
        path = profile_path or Path(saved.profile_path)
        profile = load_pipeline_profile(path) if profile_path else saved.profile
        custody_path = root / "accounting-custody.json"
        original_accounting = decode_json(custody_path.read_bytes()) if custody_path.exists() else None
        if canonical(profile.accounting) != canonical(original_accounting):
            raise ValueError("accounting mode/purpose is immutable for a run; use its original profile")
        source = load_source(path, profile)
        cards, skipped = current_cards(path, profile, saved.objectives, root)
        provider = _provider(root, state, profile)
        offline = offline or state.offline
        local_only = local_only or state.local_only
        if local_only and any(e and e.locality != "local" for e in provider.endpoints.values()):
            raise ValueError("local_only requires all endpoints and upstream processing to be local")
        # Invocation restrictions can only narrow this run. Start a new run to
        # relax them; neither profile replacement nor a hosted grant does so.
        if (offline, local_only) != (state.offline, state.local_only):
            state = state.model_copy(update={"offline": offline, "local_only": local_only})
            _atomic(root / "execution.json", canonical(state))
        permission = HostedPermission.read_json(permission_path.read_bytes()) if permission_path else None
        if permission is not None:
            if permission.run_id != state.run_id:
                raise ValueError("hosted permission belongs to another run")
            _immutable(root / "permissions" / (permission.permission_id + ".json"), canonical(permission))
        if profile_path:
            _atomic(
                root / "run.json",
                canonical(saved.model_copy(update={"profile_path": str(path.resolve()), "profile": profile})),
            )
        cancellation = Cancellation(root)
        if continue_cancelled:
            cancellation.path.unlink(missing_ok=True)
        report: dict[str, object] = {
            "run_id": state.run_id,
            "effective_profile": profile.model_dump(mode="json"),
            "status": "cancelled" if cancellation.is_set() else "completed",
            "coverage_limit": COVERAGE,
            "applied_to_anki": False,
            "skipped": skipped,
            "network_policy": {"offline": offline, "local_only": local_only},
        }
        if not cancellation.is_set() and any(s["reason"] == "processing_limit" for s in skipped):
            report["status"] = "deferred"
        outcomes: dict[str, list[dict[str, object]]] = {}
        index_path = root / "stage-index.json"
        index = decode_json(index_path.read_bytes()) if index_path.exists() else {}
        if not isinstance(index, dict):
            raise ValueError("invalid stage index")
        invalid_stages: dict[str, str] = {}
        for key, identity in index.items():
            try:
                stage = load_stage(root, str(identity))
                if stage.state == "succeeded":
                    stage_artifact(root, stage)
            except (OSError, ValueError, KeyError, TypeError) as error:
                invalid_stages[key] = str(error)
        report["invalid_stages"] = [
            {"index_key": key, "stage_id": index[key], "state": "failed", "detail": detail}
            for key, detail in invalid_stages.items()
        ]
        if invalid_stages and not cancellation.is_set():
            report["status"] = "failed"
        llm_off = offline or not any(provider.endpoints.values())
        disclosures: list[object] = []
        tasks: dict[tuple[str, str], tuple[ModelTask, tuple[Scope, ...]]] = {}
        if not llm_off:
            for card in cards:
                task, scope = _task(card, source, profile)
                tasks[card.claims.note_id, card.claims.card_key] = task, scope
                disclosures.append(decode_json(provider.disclosure(task, scope, recipe=card.recipe)))
            disclosure_text = canonical(
                {"run_id": state.run_id, "config_sha256": _config(profile), "tasks": disclosures}
            ).decode()
            _atomic(root / "disclosure.json", disclosure_text.encode())
            if disclose:
                disclose(disclosure_text)
        status_rank = {"completed": 0, "succeeded": 0, "deferred": 1, "failed": 2, "unknown": 3, "cancelled": 4}
        for name in (only,) if only else STAGES:
            if cancellation.is_set():
                report["status"] = "cancelled"
                break
            rows: list[dict[str, object]] = []
            children: list[str] = []
            if name in ("ingest", "analyze", "propose", "counter_check"):
                for card in cards:
                    if cancellation.is_set():
                        report["status"] = "cancelled"
                        break
                    required = PREREQUISITES.get(name)
                    problem = None
                    invalid_current = False
                    reuse = "not_processed"
                    try:
                        _prerequisite(root, index, card, name)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        invalid_current, problem = True, str(error)
                    previous = None
                    if problem is None:
                        try:
                            previous = _prerequisite(root, index, card, required) if required else None
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            problem = str(error)
                    if problem is not None:
                        row: dict[str, object] = {
                            "state": "failed",
                            "reason": "prerequisite_artifact_invalid",
                            "detail": problem,
                        }
                    elif required and previous is None and not (name in ("propose", "counter_check") and llm_off):
                        row = {
                            "state": "deferred",
                            "reason": "prerequisite_" + required + "_incomplete",
                        }
                    elif name == "ingest":
                        row = {"stage_id": _stage(root, state, card, name, {"binding": card.binding})}
                    elif name == "analyze":
                        row = _analyze(root, state, card, source)
                        reuse = "reused" if row.pop("_reused", False) else "processed"
                        row["analysis_sha256"] = hashlib.sha256(card.analysis_path.read_bytes()).hexdigest()
                    elif llm_off:
                        row = {"state": "not_requested", "reason": "LLM-off"}
                    else:
                        task, scope = tasks[card.claims.note_id, card.claims.card_key]
                        role: Role = "proposal" if name == "propose" else "counter_check"
                        endpoint = provider.endpoints[role]
                        credential = (
                            os.environ.get(endpoint.credential_ref) if endpoint and endpoint.credential_ref else None
                        )
                        proposal = provider.completed("proposal", task, scope) if role == "counter_check" else None
                        completed = provider.completed(role, task, scope)
                        if role == "counter_check" and proposal is None:
                            row = {"state": "deferred", "reason": "prerequisite_proposal_incomplete"}
                        elif completed is None and endpoint and endpoint.credential_ref and not credential:
                            row = {"state": "deferred", "reason": "missing_runtime_credential"}
                        else:
                            assert endpoint is not None
                            context = RequestContext(endpoint=endpoint, task=task, recipe=card.recipe, scope=scope)
                            _immutable(root / "request-contexts" / (digest(context) + ".json"), canonical(context))
                            bound = None if profile.accounting else input_token_bound(endpoint)
                            prior_calls = {call.call_id for call in provider.calls()}
                            result = completed or provider.request(
                                role,
                                task,
                                recipe=card.recipe,
                                scope=scope,
                                prompt_version=PROMPT_VERSIONS[role],
                                transport=HTTPTransport(credential),
                                token_bound=bound,
                                permission=permission,
                                cancel=cancellation,
                                retry_unknown=retry_unknown,
                                duplicate_warning_confirmed=confirm_duplicate_billing,
                            )
                            row = asdict(result)
                            dispatched = any(
                                call.call_id not in prior_calls and call.reason != "not_dispatched"
                                for call in provider.calls()
                            )
                            reuse = "reused" if completed else "processed" if dispatched else "not_processed"
                            if result.answer is not None:
                                row["reason"] = "transport_only_not_acceptance"
                                envelope = _model_envelope(root, card, role)
                                if role == "counter_check":
                                    proof = cast(dict[str, object], envelope["semantic_comparison"])
                                    row["semantic_comparison"] = proof["relation"]
                                row["stage_id"] = _stage(
                                    root,
                                    state,
                                    card,
                                    name,
                                    envelope,
                                )
                                if role == "counter_check":
                                    row["authority"] = "transport_only_not_acceptance"
                                    assert proposal is not None and proposal.answer is not None
                                    row["comparison"] = asdict(compare_answers(proposal.answer, result.answer))
                    prerequisites: tuple[str, ...] = (previous,) if previous else ()
                    if isinstance(row.get("stage_id"), str):
                        prerequisites += (str(row["stage_id"]),)
                    execution_stage = _summary(
                        root,
                        state,
                        source,
                        profile,
                        name,
                        [row],
                        binding=card.binding,
                        prerequisites=prerequisites,
                    )
                    # Retain the damaged commitment: replay cannot replace it
                    # with a failure wrapper and then treat the branch as fresh.
                    if not invalid_current:
                        index[_card_key(card, name)] = execution_stage
                    if load_stage(root, execution_stage).state == "succeeded":
                        children.append(execution_stage)
                    # Persist per-card progress before processing another card.
                    _atomic(index_path, canonical(index))
                    rows.append(
                        {
                            "note_id": card.claims.note_id,
                            "card_key": card.claims.card_key,
                            **row,
                            "execution_stage_id": execution_stage,
                            **({"reuse": reuse} if name != "ingest" else {}),
                        }
                    )
            elif name == "decide":
                rows = [_decide(root, None)]
                if rows[0]["analysis_errors"]:
                    rows[0].update(state="failed", reason="analysis_processing_failed")
            elif name in ("select", "render"):
                rows = [preview(root)]
            elif name == "export":
                if cancellation.is_set():
                    report["status"] = "cancelled"
                    break
                if profile.accounting and profile.accounting.purpose != "runtime":
                    rows = [{"state": "not_requested", "reason": "isolated_artifact_not_learning_export"}]
                else:
                    output = Path(state.output) / ("cards-" + digest(preview(root)) + ".csv")
                    exported = _export(root, output, None, None, None, reconcile_receipt)
                    report["file_export"] = exported
                    rows = [exported]
            outcomes[name] = rows
            # Decision/selection/export validate their independent proof and
            # eligibility stores directly, so unrelated provider failures do
            # not bar valid human/library-supported cards.
            if name not in invalid_stages:
                index[name] = _summary(root, state, source, profile, name, rows, prerequisites=tuple(children))
            _atomic(index_path, canonical(index))
            for row in rows:
                status = str(row.get("state", "completed"))
                if status_rank.get(status, 0) > status_rank.get(str(report["status"]), 0):
                    report["status"] = status
        report["stages"] = outcomes
        report["stage_ids"] = index
        report["disclosure"] = disclosures
        report["provider_calls"] = [c.model_dump(mode="json") for c in provider.calls()]
        if provider.campaign:
            with provider.campaign.lock():
                report["accounting_provenance"] = provider.campaign.load().model_dump(mode="json")
            report["actual_plan_deduction"] = "unknown; reported usage is not a billing receipt"
        if only in ("select", "render"):
            report.update(preview(root))
        report = build_report(
            root,
            report,
            hashlib.sha256(permission_path.read_bytes()).hexdigest() if permission_path else None,
            permission,
        )
        secrets = tuple(
            value
            for endpoint in provider.endpoints.values()
            if endpoint and endpoint.credential_ref
            if (value := os.environ.get(endpoint.credential_ref))
        )
        try:
            report = safe_snapshot(report, secrets)
        except (ValueError, TypeError):
            raise ValueError("pipeline_report_redaction_or_validation_failed") from None
        _atomic(Path(state.output) / "report.json", canonical(report))
        return cast(dict[str, object], decode_json(canonical(report)))


def input_token_bound(endpoint: Endpoint) -> Callable[[bytes], int] | None:
    """No installed adapter qualifies complete service-side input accounting.

    Compatible model labels do not establish tokenizer/framing bounds. New
    dispatch fails closed; controlled engineering tests inject their own bound.
    Completed exact calls remain reusable without this dispatch prerequisite.
    """
    return None


def reconcile_call(
    root: Path,
    call_id: str,
    *,
    response: Path | None,
    abandon: bool,
    request_sha256: str | None,
    provider_request_id: str | None,
) -> dict[str, object]:
    with run_lock(root):
        state, saved = load_execution(root), load_run(root)
        provider = _provider(root, state, saved.profile)
        if abandon:
            if response is not None:
                raise ValueError("choose response retrieval or abandonment, not both")
            provider.abandon(call_id)
            return {
                "call_id": call_id,
                "state": "abandoned",
                "warning": "reservation retained; retry may duplicate billing",
            }
        if response is None or request_sha256 is None or provider_request_id is None:
            raise ValueError("response, original request digest and provider request ID are required")
        call = next((c for c in provider.calls() if c.call_id == call_id), None)
        if call is None:
            raise ValueError("unknown call ID")
        for path in (root / "request-contexts").glob("*.json"):
            context = RequestContext.read_json(path.read_bytes())
            if path.stem != digest(context):
                raise ValueError("request context digest mismatch")
            if digest(context.endpoint) == call.endpoint_sha256 and context.task.task_sha256:
                try:
                    result = provider.reconcile(
                        call_id,
                        endpoint=context.endpoint,
                        task=context.task,
                        request_sha256=request_sha256,
                        provider_request_id=provider_request_id,
                        response=response.read_bytes(),
                        credential=os.environ.get(context.endpoint.credential_ref)
                        if context.endpoint.credential_ref
                        else None,
                    )
                except ValueError as error:
                    if str(error) == "reconciliation task mismatch":
                        continue
                    raise
                return cast(dict[str, object], decode_json(canonical(asdict(result))))
        raise ValueError("original request context is unavailable")


def report(root: Path) -> dict[str, object]:
    try:
        state = load_execution(root)
        return safe_snapshot(decode_json((Path(state.output) / "report.json").read_bytes()))
    except (OSError, ValueError, TypeError):
        # Validation errors can quote corrupted records or credential-bearing
        # exception text. The report command must not print those raw inputs.
        raise ValueError("pipeline_report_unavailable_or_invalid") from None
