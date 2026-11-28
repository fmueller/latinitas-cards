"""Stored, source-rebound grammar consumers; no inference, approval or Anki writes.

Adaptation persists explicit objectives, not accepted cards. Preview reconstructs
assertions from current inputs. Export repeats that reconstruction and the shared
full-card gate at the actual writer boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from .acceptance import AcceptanceReader, check_envelope, load_acceptance, save_acceptance
from .acceptance_policy import ScopeCell, implementation_digest
from .acceptance_presentation import ACCEPTANCE_FIELDS, Presentation, metadata, note_tags, require_installation
from .acceptance_records import (
    Acceptance,
    Assertion,
    Binding,
    Check,
    Record,
    Text,
    Versions,
    canonical,
    decode_json,
    digest,
)
from .card_acceptance import CardClaims, DecisionSource, export_cards
from .claim_review import Evidence
from .deck_evidence import SourceCell, _construction_issues
from .export_receipts import (
    IMPORT_ACTION,
    MANUAL_ACTION,
    Assessment,
    Cursor,
    ExportCard,
    ExportFile,
    ExportReceipt,
    Handoff,
    ReceiptCardRef,
    invalidated_cards,
)
from .form_evidence import attested_segmentation, contextual_ppp
from .form_recipes import RULE_VERSION, derive_form, form_translations
from .grammar_layout import COLUMNS, GRAMMAR_LAYOUT, LAYOUT_SHA256
from .html_text import source_html_to_text
from .identity import ResolvedSourceIdentity, derive_latinitas_id, resolve_source_identities
from .latin_library import ProcessingUnavailable, attested_paradigms
from .manifest import CsvIdentityManifest, reconcile_csv_manifest
from .phrase_inputs import EntryV2, Features, Form, FormV2, PhraseInputV2, Selection, Sense
from .phrase_recipes import _CANONICAL, Reference, render_phrase_objective, render_phrase_text
from .pipeline_batches import Batches, CardRef, Receipt, run_lock
from .pipeline_profile import PipelineProfile, RecipeOverrides, load_pipeline_profile
from .preview_export import serialize_anki_csv
from .principal_parts import PrincipalPartParseSuccess, parse_principal_parts
from .profile import DeckProfile, load_profile
from .providers import PROMPT_VERSIONS
from .sources import CanonicalSourceRecord, read_source_records
from .terminology import feature_name, feature_value


class FormObjective(Record):
    object_key: Text
    entry: Text
    sense: Text
    form: Text | None
    anchors: tuple[Text, ...]
    recipe: Literal[
        "form_translation",
        "form_recognition",
        "ppp_translation",
        "ppp_recognition",
        "verb_production",
        "noun_declension",
        "adjective_declension",
        "form_cloze",
    ]
    target: Features
    distinction: Literal["analysis", "case", "number", "gender", "person", "tense", "mood", "voice"]
    context: str
    translations: tuple[Text, ...]
    explanation: str
    caveats: tuple[Text, ...]
    evidence: tuple[Evidence, ...]


class PhraseObjective(Record):
    object_key: Text
    selection: Selection
    recipe: Literal["phrase_translation", "phrase_recognition", "phrase_form_cloze"]
    target: Text | None
    distinction: Literal["meaning", "analysis", "case", "number", "gender", "person", "tense", "mood", "voice"]
    reference: Reference


class Objectives(Record):
    format: Literal["latinitas-grammar-objectives"]
    version: Literal[1]
    forms: tuple[FormObjective, ...]
    phrases: tuple[PhraseObjective, ...]


class Run(Record):
    format: Literal["latinitas-grammar-run"]
    version: Literal[1]
    profile_path: Text
    profile: PipelineProfile
    objectives: Objectives


class Analysis(Record):
    """Complete application analysis, never a run-supplied policy or approval."""

    format: Literal["latinitas-grammar-analysis"]
    version: Literal[1]
    binding: Binding
    versions: Versions
    assertions: tuple[Assertion, ...]
    checks: tuple[Check, ...]
    stage_ids: tuple[Text, ...]


@dataclass(frozen=True)
class Source:
    data: PhraseInputV2
    deck: DeckProfile
    records: dict[str, CanonicalSourceRecord]
    identities: dict[str, ResolvedSourceIdentity]
    scope: str


@dataclass(frozen=True)
class Rendered:
    recipe: str
    prompt: str
    answer: str
    claims: CardClaims
    binding: Binding
    versions: Versions
    material: str
    analysis_path: Path
    objective: FormObjective | PhraseObjective


def _checked_file(base: Path, path: str, sha256: str) -> Path:
    result = (base / path).resolve()
    if hashlib.sha256(result.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"changed profile/input: {path}")
    return result


def load_source(profile_path: Path, profile: PipelineProfile) -> Source:
    base = profile_path.resolve().parent
    deck_path = _checked_file(base, profile.deck_profile.path, profile.deck_profile.sha256)
    deck = load_profile(deck_path)
    if deck.language_tag != profile.language:
        raise ValueError("source and output language differ; confirm the effective deck profile")
    inputs = [(item, _checked_file(base, item.path, item.sha256)) for item in profile.inputs]
    phrase_paths = [path for item, path in inputs if item.kind == "phrases"]
    if len(phrase_paths) != 1:
        raise ValueError("one explicitly upgraded phrase input is required")
    path = phrase_paths[0]
    data = PhraseInputV2.model_validate(decode_json(path.read_bytes()))
    if (path.parent / data.profile).resolve() != deck_path:
        raise ValueError("phrase input uses another deck profile")
    source_path = (path.parent / data.source).resolve()
    if source_path not in [p for item, p in inputs if item.kind == "deck"]:
        raise ValueError("confirmed source must be a hashed deck input")
    records = read_source_records(source_path, source_id_field=deck.source_identity.field, note_type=deck.note_type)
    scope: str | None = None
    manifest_ids: tuple[str | None, ...] | None = None
    if source_path.suffix.lower() == ".csv":
        manifest_path = path.parent / data.manifest if data.manifest else Path(f"{source_path}.latinitas.json")
        manifest = CsvIdentityManifest.load(manifest_path)
        if not manifest.is_scoped:
            raise ValueError("confirmed source scope required")
        scope = manifest.source_scope
        if deck.source_identity.strategy == "manifest":
            reconciled = reconcile_csv_manifest(records, manifest)
            if reconciled.reviews:
                raise ValueError("source identity reconciliation required")
            manifest_ids = tuple(reconciled.identities_by_row.get(i) for i in range(len(records)))
    identities = resolve_source_identities(records, deck, source_scope=scope, manifest_identities=manifest_ids)
    by_id = {i.value: r for i, r in zip(identities, records, strict=True)}
    if len(by_id) != len(records) or any(e.identity not in by_id for e in data.entries):
        raise ValueError("missing or ambiguous source identity")
    supplied = Binding.model_validate(data.binding)
    expected = input_binding(data, deck, source_path.read_bytes(), scope or supplied.source_scope)
    if supplied != expected:
        raise ValueError("stale phrase input binding")
    return Source(data, deck, by_id, {i.value: i for i in identities}, expected.source_scope)


def input_binding(data: PhraseInputV2, deck: DeckProfile, source_bytes: bytes, scope: str) -> Binding:
    """Exact v2 input commitment, excluding its own envelope to avoid a cycle."""
    return Binding(
        source_scope=scope,
        source_identity="phrase-input-v2",
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        extraction_sha256=digest({"entries": data.entries, "patterns": data.patterns}),
        mapping_sha256=digest(deck.model_dump()),
        task_sha256=digest({"kind": "phrase-input-v2", "registry": data.registry_version}),
        language="de" if deck.language_tag == "de" else "en",
        config_sha256=digest({"registry": data.registry_version}),
    )


def _cell(record: CanonicalSourceRecord, fields: tuple[str, ...], text: str) -> SourceCell:
    for field in fields:
        raw = record.fields.get(field, "")
        start = raw.find(text)
        while start >= 0:
            cell = SourceCell(field=field, start=start, end=start + len(text))
            try:
                if cell.extract(record) == text:
                    return cell
            except ValueError:
                pass
            start = raw.find(text, start + 1)
    raise ValueError(f"form/meaning absent from confirmed source mapping: {text}")


def _entry(source: Source, identity: str, sense_key: str) -> tuple[EntryV2, Sense]:
    entry = next((e for e in source.data.entries if e.identity == identity), None)
    if entry is None:
        raise ValueError("missing source entry")
    sense = next((s for s in entry.senses if s.key == sense_key), None)
    if sense is None or not sense.evidence or not entry.evidence or not source.deck.fields.meaning_field:
        raise ValueError("missing confirmed sense/evidence")
    record = source.records[identity]
    if source.deck.include_source_tag and source.deck.include_source_tag not in record.source_tags:
        raise ValueError("source entry outside confirmed source-tag filter")
    _cell(record, (source.deck.fields.meaning_field,), sense.meaning)
    return entry, sense


def _form(source: Source, entry: EntryV2, key: str) -> FormV2:
    form = next((f for f in entry.forms if f.key == key), None)
    if form is None:
        raise ValueError("missing confirmed source form")
    record = source.records[entry.identity]
    fields = (
        (source.deck.fields.principal_parts_field,)
        if form.source_role != "other"
        else (
            source.deck.fields.lexical_entry_field,
            source.deck.fields.principal_parts_field,
        )
    )
    cell = _cell(record, fields, form.text)
    if form.source_role != "other":
        parsed = parse_principal_parts(record, source.deck)
        if not isinstance(parsed, PrincipalPartParseSuccess):
            raise ValueError("unconfirmed source role")
        part = parsed.value.by_role.get(form.source_role)
        if (
            part is None
            or part.unresolved
            or part.is_omitted
            or part.display != form.text
            or cell.field != source.deck.fields.principal_parts_field
        ):
            raise ValueError("source role mismatch; supine/fourth position cannot establish PPP")
    return form


def _feature_terms(deck: DeckProfile) -> dict[str, dict[str, str]]:
    return {
        key: {
            "label": feature_name(key, deck.morphology.terminology, deck.language_tag),
            **{
                value: feature_value(key, canonical_value, deck.morphology.terminology, deck.language_tag)
                for value, canonical_value in values.items()
            },
        }
        for key, values in _CANONICAL.items()
    }


def _features(features: Features, deck: DeckProfile) -> str:
    terms = _feature_terms(deck)
    return "; ".join(f"{terms[k]['label']}: {terms[k][v]}" for k, v in features.model_dump(exclude_none=True).items())


FORM_PRESENTATION = {
    "de": {
        "production_prompt": "Bilde die lateinische Form: {cue}\n{features}",
        "translation_prompt": "{surface}\nÜbersetze.",
        "translation_answer": "{translations}\nReferenz als Orientierung; Selbstbeurteilung.",
        "recognition_prompt": "{context_or_surface}\nBestimme die Form; mögliche Lesarten angeben.",
        "cloze_prompt": "{context}\n{features}\n{lemma} — {meaning}",
        "ppp_suffix": "PPP (Partizip Perfekt Passiv)",
        "stem_line": "Stamm/Basis: {stem}",
    },
    "en": {
        "production_prompt": "Supply the Latin form: {cue}\n{features}",
        "translation_prompt": "{surface}\nTranslate.",
        "translation_answer": "{translations}\nReference guidance; self-assessment.",
        "recognition_prompt": "{context_or_surface}\nAnalyze the form; give possible readings.",
        "cloze_prompt": "{context}\n{features}\n{lemma} — {meaning}",
        "ppp_suffix": "PPP (perfect passive participle)",
        "stem_line": "Stem/base: {stem}",
    },
}


FORM_DIRECTIONS = {
    "verb_production": "production",
    "noun_declension": "production",
    "adjective_declension": "production",
    "form_translation": "translation",
    "ppp_translation": "translation",
    "form_recognition": "recognition",
    "ppp_recognition": "recognition",
    "form_cloze": "cloze",
}
FORM_ASSERTION_VALUE_CONTRACTS = {
    "source": (
        "JSON object {entry, sense}: entry is the whole entry object from the unverified-form-entry evidence, "
        "not its lemma; sense is the object in entry.senses whose key is objective.sense_key."
    ),
    "target": (
        "JSON object {surface, features, source_role, anchors, rule}, plus context only for a contextual "
        "recognition/cloze task. surface is Latin; features uses canonical feature names and values, with "
        "unspecified features null. source_role is the selected form's role, or derived if none is selected. "
        "anchors contains the full supplied form objects in selected anchor order (or []). "
        "rule is null without derivation; otherwise identify the applicable derivation: "
        "regular-first-present-active/1, regular-first-declension/1 or regular-us-a-um/1. "
        "A selected PPP can anchor an adjective derivation while retaining its source_role."
    ),
    "stem": (
        "JSON object {convention, components, operation, surface}. components is an ordered array of "
        "{role, text} objects: stem then personal-ending (verb) or case-ending (noun/adjective). "
        "Use bennett-present-stem/1 (infinitive without -re) for verbs, bennett-declensional-stem/1 for nouns, "
        "or bounded-generator-base/1 (masculine nominative without -us) for adjectives, including an adjective "
        "derived from a PPP. "
        "surface is the target surface. Copy Latin literally, including long-vowel marks. "
        "operation is concatenation only when stem text + ending text equals surface exactly; "
        "otherwise unverified with ending text null. Examples: vocāre 1st plural is "
        "vocā + mus = vocāmus; vocō is stem vocā, ending null, unverified. porta genitive plural is "
        "portā + rum = portārum; porta is stem portā, ending null, unverified."
    ),
}
_RENDERED_ANSWERS = {
    "production": "Production cue: second anchor for a verb with two anchors, otherwise first anchor, otherwise "
    "mapped lemma. answer line 1 is the Latin target surface; if stem_line is supplied, line 2 is stem_line "
    "with the stem component text.",
    "translation": "answer begins with translation_answer, joining readings with ' / '.",
    "recognition": "answer joins distinct feature displays for all same-surface source forms with newlines, "
    "prefixed by 'PPP — ' for PPP. Feature displays use feature_order, omit nulls, and join 'label: value' "
    "terms with '; '.",
    "cloze": "answer line 1 is the Latin target surface.",
}
FORM_RENDERED_CONTRACTS = {
    direction: "JSON object {prompt, answer}, both strings: independently construct the whole visible card. "
    "prompt is the supplied prompt template with each remaining placeholder replaced by the value you solve "
    "independently; keep all other template text exactly. "
    + text
    + " Then add independently supported explanation and caveat clauses in assertion order, each on a new "
    "answer line."
    for direction, text in _RENDERED_ANSWERS.items()
}


def _assertion(
    identity: str,
    kind: Literal["label", "translation", "segmentation", "explanation"],
    value: str,
    evidence: tuple[Evidence, ...],
    *,
    sense: str | None = None,
    alternatives: tuple[str, ...] = (),
) -> Assertion:
    return Assertion(
        id=identity,
        kind=kind,
        value=value,
        selected_sense=sense,
        alternatives=tuple(dict.fromkeys(alternatives)),
        evidence=tuple(dict.fromkeys(evidence)),
    )


def source_role_target(
    source: Source, objective: FormObjective | PhraseObjective, versions: Versions
) -> tuple[dict[str, object] | None, tuple[Evidence, ...], str]:
    """Reconstruct a source-role, attested derived cell or exact attested context.

    An entire serialized entry includes lexical/sense claims this rule cannot
    establish. A target proof does not cover translations or free clauses.
    """
    expected: dict[str, object] | None = None
    constraints: tuple[Evidence, ...] = ()
    reason = "unsupported_assertion_class_or_rule_evidence"
    if isinstance(objective, FormObjective):
        entry, sense = _entry(source, objective.entry, objective.sense)
        anchors = tuple(_form(source, entry, key) for key in objective.anchors)
        selected = _form(source, entry, objective.form) if objective.form else None
        record = source.records[entry.identity]
        lemma = source_html_to_text(record.fields.get(source.deck.fields.lexical_entry_field, "")).strip()
        present = Features(person="1", number="singular", tense="present", mood="indicative", voice="active")
        if (
            objective.recipe in ("form_recognition", "form_translation")
            and entry.pos == "verb"
            and selected is not None
            and selected.source_role == "present_1s"
            and selected.text == lemma
            and selected.text.endswith("ō")
            and not anchors
            and not objective.context
        ):
            expected = {
                "surface": selected.text,
                "features": present.model_dump(),
                "source_role": "present_1s",
                "anchors": [],
                "rule": None,
            }
            constraints = (
                Evidence(
                    "latinitas-principal-part-rules",
                    "confirmed-present-1s-active-role/1",
                    versions.checker,
                    "Consistency with confirmed present_1s mapping of an active -ō citation form: "
                    "first person singular present indicative active. Not independent lexical, "
                    "paradigm applicability or generated-cell verification.",
                ),
            )
        elif (
            objective.recipe in ("ppp_recognition", "form_cloze")
            and entry.pos == "verb"
            and selected is not None
            and selected.source_role == "ppp"
            and not anchors
            and objective.context
        ):
            reason = "context_requires_independent_applicability_evidence"
            try:
                features, constraints = contextual_ppp(
                    lemma, selected.text, objective.context, cloze=objective.recipe == "form_cloze"
                )
            except ProcessingUnavailable:
                features = None
                reason = "attested_context_resource_unavailable"
            if features is not None:
                expected = {
                    "surface": selected.text,
                    "features": features.model_dump(),
                    "source_role": "ppp",
                    "anchors": [],
                    "rule": None,
                    "context": objective.context,
                }
                reason = "independently_attested_contextual_ppp"
        elif objective.recipe in ("verb_production", "noun_declension", "adjective_declension"):
            reason = "generated_cell_requires_independent_applicability_evidence"
            try:
                paradigms = attested_paradigms()
            except ProcessingUnavailable:
                paradigms = ()
                reason = "attested_morphology_resource_unavailable"
            for paradigm in paradigms:
                if (
                    paradigm.pos != entry.pos
                    or paradigm.lemma != lemma
                    or tuple(f.text for f in anchors) != tuple(a.surface for a in paradigm.anchors)
                    or any(
                        getattr(attested.features, name) != value
                        for anchor, attested in zip(anchors, paradigm.anchors, strict=True)
                        for name, value in anchor.features.model_dump(exclude_none=True).items()
                    )
                    or selected is not None
                    or objective.context
                ):
                    continue
                for cell in paradigm.cells:
                    if cell.features != objective.target:
                        continue
                    expected = cast(
                        dict[str, object],
                        decode_json(
                            canonical(
                                {
                                    "surface": cell.surface,
                                    "features": cell.features,
                                    "source_role": "derived",
                                    "anchors": anchors,
                                    "rule": paradigm.rule,
                                }
                            )
                        ),
                    )
                    constraints = (paradigm.evidence,)
                    reason = "independently_attested_lexeme_and_cell"
        else:
            reason = "missing_applicable_source_role_rule"
        if expected is not None:
            # _form verified each role against the confirmed principal-part parser.
            constraints = (
                Evidence(
                    "confirmed-deck-mapping",
                    source.scope + "/" + entry.identity,
                    "1",
                    canonical(
                        {
                            "fields": dict(record.fields),
                            "mapping": source.deck.fields,
                            "roles": source.deck.principal_parts,
                            "selected_sense": sense,
                            "forms": anchors or (selected,),
                        }
                    ).decode(),
                ),
                *constraints,
            )
    return expected, constraints, reason


def source_claims(
    source: Source, objective: FormObjective | PhraseObjective, versions: Versions, root: Path | None = None
) -> tuple[dict[str, object], tuple[Evidence, ...], str]:
    if isinstance(objective, PhraseObjective):
        from .phrase_evidence import phrase_claims

        return phrase_claims(source, objective, versions, root)
    target, constraints, reason = source_role_target(source, objective, versions)
    if target is None:
        return {}, constraints, reason
    expected: dict[str, object] = {"target": target}
    if not isinstance(objective, FormObjective) or target["source_role"] != "derived":
        return expected, constraints, reason
    entry, _ = _entry(source, objective.entry, objective.sense)
    lemma = source_html_to_text(
        source.records[entry.identity].fields.get(source.deck.fields.lexical_entry_field, "")
    ).strip()
    try:
        stem, references = attested_segmentation(
            lemma, entry.pos, str(target["surface"]), Features.model_validate(target["features"])
        )
    except ProcessingUnavailable:
        return expected, constraints, reason
    if stem is not None:
        expected["stem"] = stem
        constraints = tuple(dict.fromkeys((*constraints, *references)))
    return expected, constraints, reason


def rule_checks(source: Source, card: Rendered, root: Path | None = None) -> tuple[Check, ...]:
    from .meaning_evidence import meaning_checks

    expected, constraints, reason = source_claims(source, card.objective, card.versions, root)
    meanings, meaning_constraints = meaning_checks(source, card, card.claims.assertions, expected)
    constraints = tuple(dict.fromkeys((*constraints, *meaning_constraints)))
    unsupported = {
        "source": "source_requires_lexical_and_sense_evidence",
        "translation": "translation_requires_sense_government_and_semantic_evidence",
        "segmentation": "segmentation_requires_convention_and_reconstruction_evidence",
        "explanation": "explanation_requires_clause_evidence",
    }
    checks = []
    for assertion in card.claims.assertions:
        result: Literal["pass", "fail", "not_applicable"] = "not_applicable"
        objections: tuple[str, ...] = (unsupported.get(assertion.id, unsupported.get(assertion.kind, reason)),)
        if assertion.id in expected and assertion.kind == ("segmentation" if assertion.id == "stem" else "label"):
            try:
                matches = decode_json(assertion.value) == expected[assertion.id]
            except ValueError:
                matches = False
            matches = matches and assertion.selected_sense is None and not assertion.alternatives
            result = "pass" if matches else "fail"
            objections = () if matches else ("source_bound_target_contradiction",)
        if assertion.id in meanings:
            meaning = meanings[assertion.id]
            result = "pass" if meaning == "equivalent" else "fail" if meaning == "contradiction" else "not_applicable"
            objections = () if meaning == "equivalent" else ("bounded_meaning_" + meaning,)
        envelope = check_envelope(card.binding, card.versions, "rule", (assertion,), reference_constraints=constraints)
        checks.append(
            Check(
                id=assertion.id,
                assertion_ids=(assertion.id,),
                method="rule",
                version=card.versions.checker,
                result=result,
                evidence=constraints,
                alternatives=(),
                objections=objections,
                input_sha256=digest(envelope),
            )
        )
    return tuple(checks)


def _render_form(
    source: Source, objective: FormObjective, language: str
) -> tuple[str, str, tuple[Assertion, ...], str, str]:
    presentation = FORM_PRESENTATION[language]
    entry, sense = _entry(source, objective.entry, objective.sense)
    selected = _form(source, entry, objective.form) if objective.form else None
    anchors = tuple(_form(source, entry, key) for key in objective.anchors)
    evidence = (
        entry.evidence
        + sense.evidence
        + objective.evidence
        + tuple(e for f in (*anchors, *((selected,) if selected else ())) for e in f.role_evidence)
    )
    if not evidence:
        raise ValueError("missing evidence")
    ppp = selected is not None and selected.source_role == "ppp"
    if objective.recipe.startswith("ppp_") and not ppp:
        raise ValueError("confirmed PPP required")
    if ppp and (
        selected is None
        or selected.features.tense != "perfect"
        or selected.features.voice != "passive"
        or any(getattr(selected.features, k) is None for k in ("case", "number", "gender"))
    ):
        raise ValueError("incomplete or contradictory PPP analysis")
    if selected and entry.pos == "verb" and selected.features.case and not ppp:
        raise ValueError("nominal verb form requires confirmed PPP evidence")
    if selected and selected.source_role == "supine":
        raise ValueError("supine not in the standalone PPP/form subset")
    material = "confirmed-ppp/1" if ppp else "source-form/1"
    production = objective.recipe in ("verb_production", "noun_declension", "adjective_declension")
    target = objective.target
    derived = None
    if production:
        expected_pos = {"verb_production": "verb", "noun_declension": "noun", "adjective_declension": "adjective"}[
            objective.recipe
        ]
        if entry.pos != expected_pos and not (ppp and objective.recipe == "adjective_declension"):
            raise ValueError("unsupported production POS")
        if ppp and objective.recipe == "adjective_declension":
            assert selected is not None
            if (
                not selected.text.endswith("us")
                or selected.features.case != "nominative"
                or selected.features.number != "singular"
                or selected.features.gender != "masculine"
            ):
                raise ValueError("unsupported PPP derivation anchor")
            base = selected.text[:-2]
            derived = derive_form("adjective", (base + "us", base + "a", base + "um"), target)
            surface = derived.surface
            evidence += (derived.evidence,)
        elif anchors:
            if entry.pos == "verb" and tuple(f.source_role for f in anchors) != ("present_1s", "present_infinitive"):
                raise ValueError("verified principal-part anchors required")
            if entry.pos == "verb" and anchors[0].features != Features(
                person="1", number="singular", tense="present", mood="indicative", voice="active"
            ):
                raise ValueError("contradictory verb anchor analysis")
            if entry.pos == "noun" and tuple(f.features for f in anchors) != (
                Features(case="nominative", number="singular", gender=entry.gender),
                Features(case="genitive", number="singular", gender=entry.gender),
            ):
                raise ValueError("contradictory nominative/genitive anchors")
            if entry.pos == "adjective" and tuple(f.features for f in anchors) != tuple(
                Features(case="nominative", number="singular", gender=gender)
                for gender in ("masculine", "feminine", "neuter")
            ):
                raise ValueError("contradictory adjective gender anchors")
            derived = derive_form(entry.pos, tuple(f.text for f in anchors), target)
            surface = derived.surface
            evidence += (derived.evidence,)
        elif selected and selected.features == target:
            required = (
                ("person", "number", "tense", "mood", "voice") if entry.pos == "verb" else ("case", "number", "gender")
            )
            if any(getattr(target, k) is None for k in required):
                raise ValueError("incomplete production target")
            surface = selected.text
        else:
            raise ValueError("missing applicable source cell/derivation")
        if entry.pos == "noun" and target.gender != entry.gender:
            raise ValueError("contradictory noun gender")
        if entry.pos == "verb" and len(anchors) == 2:
            cue = anchors[1].text
        elif anchors:
            cue = anchors[0].text
        else:
            cue = source.records[entry.identity].fields[source.deck.fields.lexical_entry_field]
        prompt = presentation["production_prompt"].format(cue=cue, features=_features(target, source.deck))
        if ppp:
            prompt += "\n" + presentation["ppp_suffix"]
        answer = surface
        if surface in prompt:
            raise ValueError("answer revealed by production prompt")
    else:
        if selected is None:
            raise ValueError("explicit source form required")
        surface = selected.text
        target = selected.features
        if objective.recipe.endswith("translation"):
            prompt = presentation["translation_prompt"].format(surface=surface)
            translations = objective.translations or form_translations(
                entry.pos, target, sense.meaning, language, selected.source_role
            )
            answer = presentation["translation_answer"].format(translations=" / ".join(translations))
        elif objective.recipe.endswith("recognition"):
            matching_forms = tuple(f for f in entry.forms if f.text == surface)
            readings = tuple(dict.fromkeys(_features(f.features, source.deck) for f in matching_forms))
            if not readings or any(not f.features.model_dump(exclude_none=True) for f in matching_forms):
                raise ValueError("incomplete recognition readings")
            if objective.distinction != "analysis":
                values = {getattr(f.features, objective.distinction) for f in matching_forms}
                if None in values or len(values) != 1:
                    raise ValueError("ambiguous recognition distinction")
            prompt = presentation["recognition_prompt"].format(context_or_surface=objective.context or surface)
            answer = "\n".join(readings)
            if ppp:
                answer = "PPP — " + answer
        elif objective.recipe == "form_cloze":
            if objective.context.count("____") != 1 or surface in objective.context:
                raise ValueError("cloze needs one explicit blank without the answer")
            required = (
                ("case", "number", "gender")
                if ppp or entry.pos != "verb"
                else ("person", "number", "tense", "mood", "voice")
            )
            if any(getattr(target, key) is None for key in required):
                raise ValueError("complete cloze target features required")
            cue = source.records[entry.identity].fields[source.deck.fields.lexical_entry_field]
            if surface in cue or surface in sense.meaning:
                raise ValueError("cloze cue reveals the target")
            prompt = presentation["cloze_prompt"].format(
                context=objective.context, features=_features(target, source.deck), lemma=cue, meaning=sense.meaning
            )
            answer = surface
            material = "target-cloze/1"
        else:
            raise ValueError("unsupported form recipe")
    assertions = [
        _assertion("source", "label", canonical({"entry": entry, "sense": sense}).decode(), evidence),
        _assertion(
            "target",
            "label",
            canonical(
                {
                    "surface": surface,
                    "features": target,
                    "source_role": selected.source_role if selected else "derived",
                    "anchors": anchors,
                    "rule": derived.rule if derived else None,
                    **(
                        {"context": objective.context}
                        if objective.context
                        and objective.recipe in ("form_recognition", "ppp_recognition", "form_cloze")
                        else {}
                    ),
                }
            ).decode(),
            evidence,
        ),
    ]
    if objective.recipe.endswith("translation"):
        assertions.append(
            _assertion(
                "translation",
                "translation",
                " / ".join(translations),
                evidence,
                sense=sense.key,
                alternatives=translations[1:],
            )
        )
    if derived:
        answer += "\n" + presentation["stem_line"].format(stem=derived.stem)
        concatenates = surface.startswith(derived.stem)
        # The derivation, not the entry, selects the convention: a PPP anchors an adjective derivation.
        convention, ending = {
            "regular-first-present-active/1": ("bennett-present-stem/1", "personal-ending"),
            "regular-first-declension/1": ("bennett-declensional-stem/1", "case-ending"),
            "regular-us-a-um/1": ("bounded-generator-base/1", "case-ending"),
        }[derived.rule]
        segmentation = {
            "convention": convention,
            "components": [
                {"role": "stem", "text": derived.stem},
                {"role": ending, "text": surface[len(derived.stem) :] if concatenates else None},
            ],
            "operation": "concatenation" if concatenates else "unverified",
            "surface": surface,
        }
        assertions.append(_assertion("stem", "segmentation", canonical(segmentation).decode(), evidence))
    if objective.explanation:
        answer += "\n" + objective.explanation
        assertions.append(_assertion("explanation", "explanation", objective.explanation, evidence))
    for index, caveat in enumerate(objective.caveats):
        answer += "\n" + caveat
        assertions.append(_assertion(f"caveat-{index}", "explanation", caveat, evidence))
    return prompt, answer, tuple(assertions), material, entry.identity


def _render_phrase(
    source: Source, objective: PhraseObjective, language: str
) -> tuple[str, str, tuple[Assertion, ...], str, str]:
    pattern = next((p for p in source.data.patterns if p.key == objective.selection.pattern), None)
    if pattern is None or set(pattern.roles) != {b.role for b in objective.selection.bindings}:
        raise ValueError("unsupported pattern/bindings")
    forms: dict[str, Form] = {}
    senses: dict[str, Sense] = {}
    evidence = pattern.evidence + objective.reference.evidence
    components = {}
    for binding in objective.selection.bindings:
        entry, sense = _entry(source, binding.entry, binding.sense)
        form = _form(source, entry, binding.form)
        role = pattern.roles[binding.role]
        if entry.pos != role.pos or f"{entry.identity}/{sense.key}" not in role.senses:
            raise ValueError("sense outside pattern compatibility")
        required = (
            ("person", "number", "tense", "mood", "voice") if entry.pos == "verb" else ("case", "number", "gender")
        )
        if any(getattr(form.features, k) is None for k in required) or (
            entry.pos == "noun" and form.features.gender != entry.gender
        ):
            raise ValueError("incomplete/contradictory constituent morphology")
        forms[binding.role], senses[binding.role] = form, sense
        evidence += entry.evidence + sense.evidence + form.role_evidence
        components[binding.role] = {
            "identity": asdict(source.identities[entry.identity]),
            "entry": entry,
            "sense": sense,
            "form": form,
        }
    issues, unsupported = _construction_issues(pattern, forms, senses)
    if issues or unsupported:
        raise ValueError("; ".join(issues))
    if objective.reference.language != language or not pattern.evidence:
        raise ValueError("missing matching language/pattern evidence")
    readings = objective.reference.readings
    for reading in readings:
        if set(reading.features) != set(forms) or any(reading.features[r] != f.features for r, f in forms.items()):
            raise ValueError("contextual reference contradicts bound morphology")
    order = {
        "preposition-noun": ("object",),
        "noun-genitive": ("head", "dependent"),
        "adjective-noun": ("head", "modifier"),
        "verb-noun": ("subject", "object", "verb"),
    }[pattern.family]
    words = [*pattern.function_words, *(forms[r].text for r in order if r in forms)]
    text = " ".join(words)
    assertions = [
        _assertion(
            "construction", "label", canonical({"pattern": pattern, "components": components}).decode(), evidence
        )
    ]
    target = objective.target
    if objective.recipe == "phrase_translation":
        if target is not None or objective.distinction != "meaning":
            raise ValueError("translation has no designated target")
    else:
        if target not in forms or objective.distinction == "meaning":
            raise ValueError("explicit recognition/cloze target required")
        assert target is not None
        if (
            objective.distinction != "analysis"
            and len({getattr(r.features[target], objective.distinction) for r in readings}) != 1
        ):
            raise ValueError("ambiguous distinction")
    prompt, lines = render_phrase_objective(
        text,
        {r: f.text for r, f in forms.items()},
        objective.recipe,
        target,
        objective.distinction,
        objective.reference,
        source.deck,
    )
    if objective.recipe == "phrase_form_cloze":
        assert target is not None
        target_binding = next(b for b in objective.selection.bindings if b.role == target)
        cue = source.records[target_binding.entry].fields[source.deck.fields.lexical_entry_field]
        if forms[target].text in cue:
            cue = senses[target].meaning
        if forms[target].text in cue:
            raise ValueError("cloze cue reveals the target")
        prompt += "\n" + cue
        lines.insert(0, forms[target].text)
    answer = "\n".join(lines)
    for index, reading in enumerate(readings):
        translation = " / ".join(reading.translations)
        assertions.extend(
            (
                _assertion(
                    f"translation-{index}",
                    "translation",
                    translation,
                    evidence,
                    sense=canonical({r: s.key for r, s in senses.items()}).decode(),
                    alternatives=reading.translations[1:],
                ),
                _assertion(f"explanation-{index}", "explanation", reading.explanation, evidence),
                _assertion(f"roles-{index}", "explanation", canonical(reading.contributions).decode(), evidence),
            )
        )
    identity = "phrase-source-v2-" + digest(
        {b.role: asdict(source.identities[b.entry]) for b in objective.selection.bindings}
    )
    return (
        prompt,
        answer,
        tuple(assertions),
        "target-cloze/1" if objective.recipe.endswith("cloze") else pattern.family + "/1",
        identity,
    )


def _versions(profile: PipelineProfile) -> Versions:
    current = implementation_digest()
    proposal, counter = profile.providers.proposal, profile.providers.counter_check
    model_path = proposal is not None and counter is not None
    return Versions(
        producer="grammar-recipes/1:" + current,
        analyzer=RULE_VERSION,
        proposal_prompt=PROMPT_VERSIONS["proposal"] if model_path else None,
        counter_prompt=PROMPT_VERSIONS["counter_check"] if model_path else None,
        checker="grammar-card/1:" + current,
        comparison="source-role-comparison/1:" + current,
        policy=profile.policy.id,
        policy_sha256=profile.policy.sha256,
        proposal_model=proposal.model + ":" + digest(proposal) if model_path and proposal else None,
        counter_model=counter.model + ":" + digest(counter) if model_path and counter else None,
    )


def _automatic_objectives(source: Source, recipes: tuple[str, ...]) -> tuple[FormObjective, ...]:
    result = []
    for entry in sorted(source.data.entries, key=lambda e: e.identity):
        if len(entry.senses) != 1:
            continue
        for form in sorted(entry.forms, key=lambda f: f.key):
            if form.source_role == "supine":
                continue
            choices: tuple[
                Literal["ppp_translation", "ppp_recognition", "form_translation", "form_recognition"], ...
            ] = (
                ("ppp_translation", "ppp_recognition")
                if form.source_role == "ppp"
                else ("form_translation", "form_recognition")
            )
            for recipe in choices:
                if recipe not in recipes:
                    continue
                if form.source_role == "present_infinitive" and recipe == "form_recognition":
                    continue
                result.append(
                    FormObjective(
                        object_key="form:" + form.key,
                        entry=entry.identity,
                        sense=entry.senses[0].key,
                        form=form.key,
                        anchors=(),
                        recipe=recipe,
                        target=form.features,
                        distinction="analysis",
                        context="",
                        translations=(),
                        explanation="",
                        caveats=(),
                        evidence=entry.evidence,
                    )
                )
    return tuple(result)


def current_cards(
    profile_path: Path,
    profile: PipelineProfile,
    objectives: Objectives,
    root: Path,
    *,
    resolve_decisions: bool = True,
) -> tuple[tuple[Rendered, ...], tuple[dict[str, str], ...]]:
    if profile.layout.id == "acceptance-1":
        require_installation(
            profile.layout.id,
            profile.layout.version,
            profile.layout.template_sha256,
            profile_path,
            "grammar",
        )
    elif (profile.layout.id, profile.layout.version, profile.layout.template_sha256) != (
        "grammar-v1",
        1,
        LAYOUT_SHA256,
    ):
        raise ValueError("explicit grammar-v1 layout installation/migration required")
    source = load_source(profile_path, profile)
    versions = _versions(profile)
    selected: list[FormObjective | PhraseObjective] = [*objectives.forms, *objectives.phrases]
    explicit = {(o.entry, o.form, o.recipe) for o in objectives.forms}
    selected.extend(
        o for o in _automatic_objectives(source, profile.recipes) if (o.entry, o.form, o.recipe) not in explicit
    )
    selected.sort(
        key=lambda o: (
            o.entry if isinstance(o, FormObjective) else min(b.entry for b in o.selection.bindings),
            "" if isinstance(o, FormObjective) else o.selection.pattern,
            ""
            if isinstance(o, FormObjective)
            else canonical(sorted((b.role, b.form) for b in o.selection.bindings)).decode(),
            o.object_key,
            profile.recipes.index(o.recipe) if o.recipe in profile.recipes else len(profile.recipes),
            digest(o),
        )
    )
    limit = profile.processing.max_source_units
    rendered = []
    skipped: list[dict[str, str]] = []
    seen: dict[tuple[str, str], tuple[object, ...]] = {}
    conflicted: set[tuple[str, str]] = set()
    variants: dict[str, set[str]] = {}
    processed = 0
    for objective in selected:
        if objective.recipe not in profile.recipes:
            continue
        if (
            isinstance(objective, FormObjective)
            and profile.selection.entries
            and objective.entry not in profile.selection.entries
        ):
            continue
        if (
            isinstance(objective, PhraseObjective)
            and profile.selection.patterns
            and objective.selection.pattern not in profile.selection.patterns
        ):
            continue
        if (
            isinstance(objective, PhraseObjective)
            and profile.selection.entries
            and any(b.entry not in profile.selection.entries for b in objective.selection.bindings)
        ):
            continue
        work = {
            "work_id": digest(objective),
            "source_entries": canonical(
                [objective.entry]
                if isinstance(objective, FormObjective)
                else sorted({b.entry for b in objective.selection.bindings})
            ).decode(),
        }
        if isinstance(objective, PhraseObjective):
            work["source_pattern"] = objective.selection.pattern
        if limit is not None and processed >= limit:
            skipped.append({**work, "object_key": objective.object_key, "reason": "processing_limit"})
            continue
        processed += 1
        try:
            if isinstance(objective, FormObjective):
                prompt, answer, assertions, material, identity = _render_form(source, objective, profile.language)
                source_records = {identity: dict(source.records[identity].fields)}
                target = (
                    objective.target.model_dump(exclude_none=True)
                    if objective.recipe in ("verb_production", "noun_declension", "adjective_declension")
                    else {"form": objective.form}
                )
            else:
                prompt, answer, assertions, material, identity = _render_phrase(source, objective, profile.language)
                group = digest(
                    (
                        objective.selection.pattern,
                        sorted((b.role, b.entry, b.sense) for b in objective.selection.bindings),
                    )
                )
                variant = digest(
                    sorted(
                        (b.role, _form(source, _entry(source, b.entry, b.sense)[0], b.form).text)
                        for b in objective.selection.bindings
                    )
                )
                kept = variants.setdefault(group, set())
                if variant not in kept and len(kept) >= profile.selection.variants_per_binding:
                    skipped.append(
                        {
                            **work,
                            "object_key": objective.object_key,
                            "candidate_key": digest(objective),
                            "reason": "variant_limit",
                        }
                    )
                    continue
                kept.add(variant)
                source_records = {b.entry: dict(source.records[b.entry].fields) for b in objective.selection.bindings}
                target = {"role": objective.target}
            key = next(t.slot.semantic_key for t in GRAMMAR_LAYOUT if t.slot.recipe == objective.recipe)
            exercise_object = "grammar-object-v1-" + digest(
                {
                    "object": objective.object_key,
                    "target": target,
                    "distinction": objective.distinction,
                    "language": profile.language,
                    "senses": objective.sense
                    if isinstance(objective, FormObjective)
                    else {b.role: b.sense for b in objective.selection.bindings},
                    "pattern": objective.selection.pattern if isinstance(objective, PhraseObjective) else None,
                }
            )
            note_id = derive_latinitas_id(identity, exercise_object, source_scope=source.scope)
            ref = (note_id, key)
            representation = (prompt, answer, assertions)
            if ref in seen:
                if seen[ref] == representation:
                    skipped.append({**work, "object_key": objective.object_key, "reason": "duplicate_card_ref"})
                    continue
                conflicted.add(ref)
                raise ValueError("duplicate semantic exercise")
            seen[ref] = representation
            evidence = tuple(e for a in assertions for e in a.evidence)
            assertions += (
                _assertion("rendered", "label", canonical({"prompt": prompt, "answer": answer}).decode(), evidence),
            )
            binding = Binding(
                source_scope=source.scope,
                source_identity=identity,
                source_sha256=digest(source_records),
                extraction_sha256=digest(
                    {
                        "entries": tuple(e for e in source.data.entries if e.identity in source_records),
                        "patterns": ()
                        if isinstance(objective, FormObjective)
                        else tuple(p for p in source.data.patterns if p.key == objective.selection.pattern),
                    }
                ),
                mapping_sha256=digest(
                    source.deck.apply_overrides(
                        {"morphology": {"theme": "muted", "appearance": "light", "comparison": "static"}}
                    )
                ),
                task_sha256=digest(objective),
                language=profile.language,
                config_sha256=digest(
                    {
                        "language": profile.language,
                        "layout": {"id": "grammar-v1", "version": 1, "template_sha256": LAYOUT_SHA256},
                        "registry": source.data.registry_version,
                    }
                ),
            )
            cells = tuple(
                ScopeCell(claim_class=a.kind, recipe=objective.recipe, language=profile.language, material=material)
                for a in assertions
            )
            reader = AcceptanceReader(root, binding, versions, assertions, cells, profile.policy.version)
            reference = digest({"note": note_id, "card": key})
            analysis_path = root / "analyses" / (reference + ".json")
            revision = (
                hashlib.sha256(analysis_path.read_bytes()).hexdigest()
                if resolve_decisions and analysis_path.exists()
                else "unanalysed"
            )
            path = root / "decisions" / reference / (revision + ".json")
            claims = CardClaims(note_id, key, assertions, (DecisionSource(reader, path),))
            human_path = path.with_name("unanalysed.json")
            if resolve_decisions and human_path.exists():
                try:
                    if load_acceptance(human_path).origin == "human":
                        decision, eligibility = reader.read(human_path)
                        if eligibility.eligible:
                            claims = CardClaims(note_id, key, assertions, (DecisionSource(reader, human_path),))
                except (ValueError, OSError, TypeError):
                    pass
            index_path = root / "stage-index.json"
            if revision != "unanalysed" and claims.decisions[0].path != human_path and index_path.exists():
                index = decode_json(index_path.read_bytes())
                if not isinstance(index, dict):
                    raise ValueError("invalid automatic stage index")
                stage_key = "counter_check/" + digest((note_id, key))
                if versions.counter_prompt is not None and stage_key in index:
                    model_stage = index[stage_key]
                    if not isinstance(model_stage, str) or not model_stage:
                        raise ValueError("invalid indexed counter stage")
                    revision = digest({"analysis_sha256": revision, "model_stage_id": model_stage})
                    path = path.with_name(revision + ".json")
                    claims = CardClaims(note_id, key, assertions, (DecisionSource(reader, path),))
            rendered.append(
                Rendered(
                    objective.recipe, prompt, answer, claims, binding, versions, material, analysis_path, objective
                )
            )
        except (ValueError, KeyError) as exc:
            skipped.append({**work, "object_key": objective.object_key, "reason": str(exc)})
    for card in rendered:
        if (card.claims.note_id, card.claims.card_key) in conflicted:
            objective = card.objective
            skipped.append(
                {
                    "work_id": digest(objective),
                    "source_entries": canonical(
                        [objective.entry]
                        if isinstance(objective, FormObjective)
                        else sorted({b.entry for b in objective.selection.bindings})
                    ).decode(),
                    **(
                        {"source_pattern": objective.selection.pattern}
                        if isinstance(objective, PhraseObjective)
                        else {}
                    ),
                    "object_key": objective.object_key,
                    "reason": "conflicting_card_ref",
                }
            )
    rendered = [r for r in rendered if (r.claims.note_id, r.claims.card_key) not in conflicted]
    available = {r.recipe for r in rendered}
    skipped.extend(
        {"object_key": recipe, "reason": "missing_content_evidence_or_approval"}
        for recipe in profile.recipes
        if recipe not in available
    )
    return tuple(rendered), tuple(skipped)


def adapt(
    profile_path: Path, objectives_path: Path, root: Path, effective_profile: PipelineProfile | None = None
) -> None:
    if root.exists():
        raise ValueError("choose a new run directory")
    profile = effective_profile or load_pipeline_profile(profile_path)
    objectives = Objectives.read_json(objectives_path.read_bytes())
    # Validate current inputs/targets before persisting, without allocating source identities.
    current_cards(profile_path, profile, objectives, root)
    run = Run(
        format="latinitas-grammar-run",
        version=1,
        profile_path=str(profile_path.resolve()),
        profile=profile,
        objectives=objectives,
    )
    root.mkdir(parents=True)
    (root / "run.json").write_bytes(canonical(run))
    (root / "accounting-custody.json").write_bytes(canonical(profile.accounting))


def load_run(root: Path) -> Run:
    return Run.read_json((root / "run.json").read_bytes())


def decide(root: Path, profile_path: Path | None = None) -> dict[str, object]:
    """Project the complete current analysis through T190, not an accepted flag."""
    with run_lock(root):
        return _decide(root, profile_path)


def _analysis_decision(card: Rendered) -> Acceptance:
    destination = card.claims.decisions[0]
    analysis = Analysis.read_json(card.analysis_path.read_bytes())
    if analysis.binding != card.binding or analysis.versions != card.versions:
        raise ValueError("stale analysis context")
    current = {a.id: a for a in analysis.assertions}
    if any(current.get(a.id) != a for a in card.claims.assertions):
        raise ValueError("analysis assertion mismatch")
    checks, stages = destination.reader.model_checks()
    decision = destination.reader.decide_automatic(
        checks=(*analysis.checks, *checks), stage_ids=(*analysis.stage_ids, *stages)
    )
    if destination.path.exists():
        stored, _ = destination.reader.read(destination.path)
        if stored.checks != decision.checks or stored.stage_ids != decision.stage_ids:
            raise ValueError("stored decision differs from current analysis")
    return decision


def _decide(root: Path, profile_path: Path | None) -> dict[str, object]:
    from .pipeline_orchestration import read_source_comparison

    run = load_run(root)
    path = profile_path or Path(run.profile_path)
    profile = load_pipeline_profile(path) if profile_path else run.profile
    cards, skipped = current_cards(path, profile, run.objectives, root)
    decisions = []
    errors = []
    for card in cards:
        destination = card.claims.decisions[0]
        path = card.analysis_path
        if not path.exists() or destination.path.name == "unanalysed.json":
            continue
        try:
            comparison = read_source_comparison(root, card)
            decision = _analysis_decision(card)
            if not destination.path.exists():
                save_acceptance(destination.path, decision)
        except (ValueError, OSError, TypeError) as error:
            errors.append({"note_id": card.claims.note_id, "card_key": card.claims.card_key, "reason": str(error)})
            continue
        decisions.append(
            {
                "note_id": card.claims.note_id,
                "card_key": card.claims.card_key,
                "state": decision.state,
                "reason": decision.reason,
                "semantic_comparison": comparison["relation"] if comparison else "not_available",
            }
        )
    return {"decisions": decisions, "skipped": skipped, "analysis_errors": errors}


def preview(
    root: Path,
    profile_path: Path | None = None,
    overrides: RecipeOverrides | None = None,
    *,
    effective_profile: PipelineProfile | None = None,
) -> dict[str, object]:
    run = load_run(root)
    path = profile_path or Path(run.profile_path)
    profile = effective_profile or (load_pipeline_profile(path) if profile_path else run.profile)
    if overrides is not None:
        profile = profile.overridden(overrides)
    cards, skipped = current_cards(path, profile, run.objectives, root)
    selected, suppressed = _selected(cards)
    limit = profile.selection.output_limit
    source = load_source(path, profile)
    return {
        "recipes": profile.recipes,
        "language": profile.language,
        "preset": profile.preset,
        "cards": [
            {
                "note_id": c.claims.note_id,
                "card_key": c.claims.card_key,
                "recipe": c.recipe,
                "prompt": c.prompt,
                "answer": c.answer,
                **(
                    {"front_html": render_phrase_text(c.prompt, source.deck), "back_html": _answer(c, source, profile)}
                    if profile.layout.id == "acceptance-1"
                    else {}
                ),
                "state": "accepted" if c.claims.assess().eligible else "withheld",
                "reason": c.claims.assess().reason,
                "assertions": [a.model_dump(mode="json") for a in c.claims.assertions],
            }
            for c in cards
        ],
        "ranking": "selection-v1",
        "selection": profile.selection.model_dump(mode="json"),
        "suppressed": [*suppressed, *(s for s in skipped if s["reason"] in ("variant_limit", "duplicate_card_ref"))],
        "accepted_unselected": [{"note_id": c.claims.note_id, "card_key": c.claims.card_key} for c in selected[limit:]]
        if limit is not None
        else [],
        "deferred_source_work": [s for s in skipped if s["reason"] == "processing_limit"],
        "skipped": skipped,
        "grading": "self-assessment; translations are reference guidance",
    }


def _selected(cards: tuple[Rendered, ...]) -> tuple[list[Rendered], list[dict[str, str]]]:
    accepted = {(c.note_id, c.card_key) for c in export_cards(tuple(c.claims for c in cards))}
    ordered = sorted(
        cards,
        key=lambda c: (
            c.binding.source_scope,
            c.binding.source_identity,
            c.objective.object_key,
            c.recipe,
            c.objective.target is not None,
            canonical(c.objective.target).decode(),
            c.objective.distinction,
            digest(c.objective),
        ),
    )
    selected = []
    suppressed = []
    seen: dict[str, Rendered] = {}
    for card in ordered:
        if (card.claims.note_id, card.claims.card_key) not in accepted:
            continue
        group = digest(
            (
                card.binding.source_scope,
                card.binding.source_identity,
                card.binding.language,
                card.objective.object_key,
                card.recipe,
                card.objective.target,
                card.objective.distinction,
                sorted(digest(a) for a in card.claims.assertions),
            )
        )
        retained = seen.get(group)
        if retained is not None:
            suppressed.append(
                {
                    "note_id": card.claims.note_id,
                    "card_key": card.claims.card_key,
                    "candidate_key": digest(card.objective),
                    "reason": "redundant_variant",
                    "retained_candidate_key": digest(retained.objective),
                }
            )
            continue
        selected.append(card)
        seen[group] = card
    return selected, suppressed


def export(
    root: Path,
    output: Path,
    *,
    profile_path: Path | None = None,
    output_limit: int | None = None,
    after_receipt: str | None = None,
    reconcile_receipt: str | None = None,
    effective_profile: PipelineProfile | None = None,
) -> dict[str, object]:
    with run_lock(root):
        return _export(root, output, profile_path, output_limit, after_receipt, reconcile_receipt, effective_profile)


def _card_projection(items: list[Rendered]) -> object:
    return [
        (
            c.claims.note_id,
            c.claims.card_key,
            c.binding,
            c.versions,
            c.claims.assertions,
            c.prompt,
            c.answer,
            tuple(s.reader.read(s.path)[0].decision_id for s in c.claims.decisions),
        )
        for c in items
    ]


def _selection_snapshot(
    selected: list[Rendered],
    skipped: tuple[dict[str, str], ...],
    suppressed: list[dict[str, str]],
    profile: PipelineProfile,
    source: Source,
) -> str:
    inherited = {}
    if profile.layout.id == "acceptance-1":
        for card in selected:
            tags = _inherited_tags((card,), source)
            note_tags((presentation(card, profile.language),), tags)
            inherited[card.claims.note_id] = tags
    return digest(
        {
            "ranking": "selection-v1",
            "cards": _card_projection(selected),
            "skipped": skipped,
            "suppressed": suppressed,
            "selection": profile.selection.model_dump(exclude={"output_limit"}),
            "recipes": profile.recipes,
            **({"source_tags": inherited} if profile.layout.id == "acceptance-1" else {}),
        }
    )


def _inherited_tags(cards: tuple[Rendered, ...], source: Source) -> tuple[str, ...]:
    tags: list[str] = []
    for card in cards:
        if isinstance(card.objective, FormObjective):
            entries: tuple[str, ...] = (card.objective.entry,)
        else:
            entries = tuple(binding.entry for binding in card.objective.selection.bindings)
        for entry in entries:
            tags.extend(source.records[entry].source_tags)
    return tuple(dict.fromkeys(tags))


def presentation(card: Rendered, language: str) -> Presentation:
    assessment = card.claims.assess()
    decisions = tuple(s.reader.read(s.path)[0] for s in card.claims.decisions) if assessment.eligible else ()
    checks = tuple(c for d in decisions for c in d.checks)
    methods = {c.method for c in checks}
    categories = {"source"} | (methods & {"rule", "library"})
    if methods & {"blind_model", "semantic_comparison"}:
        categories.add("model")
    basis = tuple(sorted(categories)) + tuple(
        dict.fromkeys(f"{e.source} / {e.reference} ({e.version})" for a in card.claims.assertions for e in a.evidence)
    )
    return Presentation(
        card.claims.card_key,
        assessment.origin if assessment.eligible else "none",
        tuple(d.decision_id for d in decisions),
        basis,
        tuple(dict.fromkeys(f"{c.method}: {c.result}" for c in checks)),
        (
            (
                "Referenzantwort; Selbstbeurteilung. Keine empirische Richtigkeitsquote."
                if language == "de"
                else "Reference answer; self-assessment. No empirical accuracy estimate."
            ),
            *(card.objective.caveats if isinstance(card.objective, FormObjective) else ()),
            *(
                ("Alternative Lesarten beachten." if language == "de" else "Allow alternate readings.",)
                if any(a.alternatives for a in card.claims.assertions)
                else ()
            ),
            *((assessment.reason,) if not assessment.eligible else ()),
        ),
    )


def _answer(card: Rendered, source: Source, profile: PipelineProfile | None) -> str:
    if profile is None or profile.layout.id != "acceptance-1":
        return render_phrase_text(card.answer, source.deck)

    text = card.answer
    if isinstance(card.objective, FormObjective) and not card.recipe.endswith("recognition"):
        target = cast(dict[str, object], decode_json(next(a.value for a in card.claims.assertions if a.id == "target")))
        analysis = _features(Features.model_validate(target["features"]), source.deck)
        text += "\n" + ("Analyse: " if profile.language == "de" else "Analysis: ") + analysis
    answer = render_phrase_text(text, source.deck)
    panel = presentation(card, profile.language).answer(profile.language)
    return answer.removesuffix("</div>") + panel + "</div>"


def _export_card(card: Rendered, source: Source, profile: PipelineProfile | None = None) -> ExportCard | None:
    assessment = card.claims.assess()
    if not assessment.eligible:
        return None
    decisions = tuple(s.reader.read(s.path)[0].decision_id for s in card.claims.decisions)
    slot = next(t.slot for t in GRAMMAR_LAYOUT if t.slot.recipe == card.recipe)
    return ExportCard(
        note_id=card.claims.note_id,
        card_key=card.claims.card_key,
        decision_ids=decisions,
        assertion_ids=tuple(a.id for a in card.claims.assertions),
        rendered_sha256=digest((render_phrase_text(card.prompt, source.deck), _answer(card, source, profile))),
        acceptance=cast(Literal["human", "automatic", "mixed"], assessment.origin),
        owned_fields=(
            "LatinitasID",
            "CardKey",
            slot.enabled_field,
            slot.prompt_field,
            slot.answer_field,
            *(ACCEPTANCE_FIELDS if profile and profile.layout.id == "acceptance-1" else ()),
        ),
        owned_tags=note_tags((presentation(card, profile.language),), ())
        if profile and profile.layout.id == "acceptance-1"
        else (),
    )


def installed_handoff(
    batches: Batches, cards: tuple[Rendered, ...], source: Source | None, profile: PipelineProfile | None = None
) -> tuple[tuple[Handoff, ...], dict[CardRef, ExportCard]]:
    """Reassess all prior exports, including other branches of the batch history."""
    current = {}
    for card in cards:
        assert source is not None
        ref = CardRef(note_id=card.claims.note_id, card_key=card.claims.card_key)
        current[ref] = Assessment(_export_card(card, source, profile), card.claims.assess().reason)
    if profile and profile.layout.id == "acceptance-1":
        for ref, assessment in tuple(current.items()):
            siblings = tuple(c for c in cards if c.claims.note_id == ref.note_id)
            tags = note_tags(tuple(presentation(c, profile.language) for c in siblings), ())
            if assessment.card:
                current[ref] = Assessment(assessment.card.model_copy(update={"owned_tags": tags}), assessment.reason)
    previous = []
    legacy: dict[CardRef, Handoff] = {}
    for receipt_id in sorted(set(batches.index.requests.values())):
        batch = batches.read(receipt_id)
        raw = batch.result.get("export_receipt")
        if raw is not None:
            receipt = ExportReceipt.read_json(canonical(raw))
            if receipt.run_id != batches.run_id or receipt.files != (
                ExportFile(path=batch.output, sha256=batch.output_sha256),
            ):
                raise ValueError("export receipt publication mismatch")
            previous.append(receipt)
        else:
            # Historical selection receipts cannot prove the assertion/policy
            # commitments needed to declare an installed card still current.
            for ref in batch.emitted_cards:
                legacy[ref] = Handoff(
                    **ref.model_dump(),
                    previous_decision_ids=(),
                    reason="legacy_receipt_missing_acceptance_commitments",
                    resolution="manual",
                    required_action=MANUAL_ACTION,
                )
    handoffs = {CardRef(note_id=h.note_id, card_key=h.card_key): h for h in invalidated_cards(previous, current)}
    handoffs.update(legacy)
    return (
        tuple(handoffs[ref] for ref in sorted(handoffs, key=lambda r: (r.note_id, r.card_key))),
        {ref: a.card for ref, a in current.items() if a.card is not None},
    )


def handoff(root: Path, profile_path: Path | None = None) -> dict[str, object]:
    """Inspect unresolved installed content even when current inputs cannot load."""
    with run_lock(root):
        run = load_run(root)
        path = profile_path or Path(run.profile_path)
        profile = load_pipeline_profile(path) if profile_path else run.profile
        source = None
        cards: tuple[Rendered, ...] = ()
        try:
            source = load_source(path, profile)
            cards, _ = current_cards(path, profile, run.objectives, root)
        except (ValueError, OSError):
            # This cannot authorize export. Preserve known identities rather
            # than treating unavailable input as proof of successful resolution.
            pass
        actions, _ = installed_handoff(Batches(root, digest(str(root.resolve()))), cards, source, profile)
        return {
            "state": "not_applied",
            "installation": "presumed_from_export_not_observed",
            "ineligible": [a.model_dump(mode="json") for a in actions],
            "required_action": IMPORT_ACTION,
        }


def _export(
    root: Path,
    output: Path,
    profile_path: Path | None,
    output_limit: int | None,
    after_receipt: str | None,
    reconcile_receipt: str | None,
    effective_profile: PipelineProfile | None = None,
) -> dict[str, object]:
    if output.is_symlink() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("choose an output outside the evidence store")
    run = load_run(root)
    path = profile_path or Path(run.profile_path)
    profile = effective_profile or (load_pipeline_profile(path) if profile_path else run.profile)
    custody_path = root / "accounting-custody.json"
    custody = decode_json(custody_path.read_bytes()) if custody_path.exists() else None
    if (isinstance(custody, dict) and custody.get("purpose") != "runtime") or (
        profile.accounting and profile.accounting.purpose != "runtime"
    ):
        raise ValueError("isolated development/evaluation artifacts cannot become learning exports")
    if canonical(profile.accounting) != canonical(custody):
        raise ValueError("accounting custody differs from the original run")
    limit = profile.selection.output_limit if output_limit is None else output_limit
    if limit is not None and limit <= 0:
        raise ValueError("output_limit must be positive")
    source = load_source(path, profile)
    cards, skipped = current_cards(path, profile, run.objectives, root)
    selected, suppressed = _selected(cards)
    # Reconstruct the whole snapshot, not just this batch, before replay/publication.
    repeated, repeated_skipped = current_cards(path, profile, run.objectives, root)
    repeated_selected, repeated_suppressed = _selected(repeated)

    if (_card_projection(selected), skipped, suppressed) != (
        _card_projection(repeated_selected),
        repeated_skipped,
        repeated_suppressed,
    ):
        raise ValueError("eligibility changed before export")
    snapshot = _selection_snapshot(selected, skipped, suppressed, profile, source)
    batches = Batches(root, digest(str(root.resolve())))
    batches.recover(snapshot)
    handoffs, commitments = installed_handoff(batches, cards, source, profile)
    request_id, replay, emitted = batches.request(
        snapshot, limit, profile.layout.template_sha256, after_receipt, reconcile_receipt
    )
    unresolved_notes = {h.note_id for h in handoffs}
    if replay is not None:
        raw = replay.result.get("export_receipt")
        prior = ExportReceipt.read_json(canonical(raw)) if raw is not None else None
        if handoffs != (prior.ineligible if prior is not None else ()) or (
            prior is not None and any(c.note_id in unresolved_notes for c in prior.cards)
        ):
            raise ValueError("installed content changed; explicit reconciliation required")
        return batches.result(replay)
    if output.exists():
        raise ValueError("choose a new output outside the evidence store")
    safe_selected = [c for c in selected if c.claims.note_id not in unresolved_notes]
    refs = [CardRef(note_id=c.claims.note_id, card_key=c.claims.card_key) for c in safe_selected]
    remaining = [c for c, ref in zip(safe_selected, refs, strict=True) if ref not in emitted]
    eligible = remaining if limit is None else remaining[:limit]
    touched_notes = {c.claims.note_id for c in eligible}
    carried = [
        c for c, ref in zip(safe_selected, refs, strict=True) if ref in emitted and c.claims.note_id in touched_notes
    ]
    opted_in = profile.layout.id == "acceptance-1"
    columns = (*(c for c in COLUMNS[:-1] if c != "Personal Notes"), *ACCEPTANCE_FIELDS, "Tags") if opted_in else COLUMNS
    notes: dict[str, dict[str, str]] = {}
    for card in (*eligible, *carried):
        fields = notes.setdefault(card.claims.note_id, dict.fromkeys(columns, ""))
        fields["LatinitasID"] = card.claims.note_id
        fields["CardKey"] = " ".join(filter(None, (fields["CardKey"], card.claims.card_key)))
        slot = next(t.slot for t in GRAMMAR_LAYOUT if t.slot.recipe == card.recipe)
        if fields[slot.enabled_field]:
            raise ValueError("conflicting objectives for one note/template slot")
        fields[slot.enabled_field] = "1"
        fields[slot.prompt_field] = render_phrase_text(card.prompt, source.deck)
        fields[slot.answer_field] = _answer(card, source, profile)
    if opted_in:
        for note_id, fields in notes.items():
            siblings = tuple(c for c in cards if c.claims.note_id == note_id)
            panels = tuple(presentation(c, profile.language) for c in siblings)
            inherited = _inherited_tags(siblings, source)
            owned = note_tags(panels, inherited)
            fields.update(metadata(panels, profile.language, batches.run_id))
            fields["Tags"] = " ".join((*inherited, *owned))
            for c in (*eligible, *carried):
                if c.claims.note_id == note_id:
                    ref = CardRef(note_id=note_id, card_key=c.claims.card_key)
                    commitments[ref] = commitments[ref].model_copy(update={"owned_tags": owned})
    rows = tuple(tuple(fields[column] for column in columns) for fields in notes.values())
    payload = serialize_anki_csv(source.deck.generated_note_type, source.deck.target_deck, columns, rows)
    exported_refs = tuple(CardRef(note_id=c.claims.note_id, card_key=c.claims.card_key) for c in eligible)
    cursor = Cursor(
        selection_sha256=snapshot,
        ordered_card_refs_sha256=digest(refs),
        emitted_cards=tuple(ReceiptCardRef(**r.model_dump()) for r in dict.fromkeys((*emitted, *exported_refs))),
    )
    previous_id = after_receipt or reconcile_receipt
    previous_export = batches.read(previous_id).result.get("export_receipt") if previous_id is not None else None
    export_record = {
        "format": "latinitas-export-receipt",
        "version": 1,
        "run_id": batches.run_id,
        "previous_receipt_id": ExportReceipt.read_json(canonical(previous_export)).receipt_id
        if previous_export
        else None,
        "selection_sha256": snapshot,
        "layout_sha256": profile.layout.template_sha256,
        "files": (ExportFile(path=str(output.resolve()), sha256=hashlib.sha256(payload).hexdigest()),),
        "cards": tuple(
            commitments[CardRef(note_id=c.claims.note_id, card_key=c.claims.card_key)] for c in (*eligible, *carried)
        ),
        "ineligible": handoffs,
        "batch_cursor": cursor,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    export_receipt = ExportReceipt.model_validate({**export_record, "receipt_id": digest(export_record)})
    result: dict[str, object] = {
        "exported": len(eligible),
        "notes": len(rows),
        "output": str(output.resolve()),
        "cards": [r.model_dump() for r in exported_refs],
        "carried_siblings": [
            CardRef(note_id=c.claims.note_id, card_key=c.claims.card_key).model_dump() for c in carried
        ],
        "selection_sha256": snapshot,
        "ranking": "selection-v1",
        "effective_profile": {
            **profile.model_dump(mode="json"),
            "selection": {**profile.selection.model_dump(mode="json"), "output_limit": limit},
        },
        "output_limit": {"value": limit, "source": "cli" if output_limit is not None else "profile"},
        "suppressed": [*suppressed, *(s for s in skipped if s["reason"] in ("variant_limit", "duplicate_card_ref"))],
        "accepted_unselected": [
            CardRef(note_id=c.claims.note_id, card_key=c.claims.card_key).model_dump()
            for c in remaining[len(eligible) :]
        ],
        "deferred_source_work": [s for s in skipped if s["reason"] == "processing_limit"],
        "reconciliation": {
            "receipt_id": reconcile_receipt,
            "retained": [r.model_dump() for r in emitted if r in refs],
            "ineligible": [r.model_dump() for r in emitted if r not in refs],
        },
        "skipped": skipped,
        "withheld": [
            {"note_id": c.claims.note_id, "card_key": c.claims.card_key, "reason": c.claims.assess().reason}
            for c in cards
            if not c.claims.assess().eligible
        ],
        "export_receipt": export_receipt.model_dump(mode="json"),
        "unresolved_note_rows": sorted(unresolved_notes),
        "application": {"state": "not_applied", "resolution": "manual", "required_action": IMPORT_ACTION},
        "requires": "explicit grammar-v1 new note type installation and manual import; "
        "destination inspection/approval and manual resolution are separate; no Anki application",
    }
    receipt = Receipt(
        format="latinitas-selection-receipt",
        version=1,
        run_id=batches.run_id,
        request_id=request_id,
        selection_sha256=snapshot,
        previous_receipt_id=after_receipt,
        reconciled_receipt_id=reconcile_receipt,
        ordered_card_refs_sha256=digest(refs),
        emitted_cards=tuple(dict.fromkeys((*emitted, *exported_refs))),
        output=str(output.resolve()),
        output_sha256=hashlib.sha256(payload).hexdigest(),
        result=result,
    )
    # Rendering/publication must not widen the authority checked above.
    load_source(path, profile)
    final_cards, final_skipped = current_cards(path, profile, run.objectives, root)
    final_selected, final_suppressed = _selected(final_cards)
    if (_card_projection(final_selected), final_skipped, final_suppressed) != (
        _card_projection(selected),
        skipped,
        suppressed,
    ):
        raise ValueError("eligibility changed before export")
    return batches.publish(receipt, payload)
