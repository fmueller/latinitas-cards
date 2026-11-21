"""Source reconstruction for controlled phrases and their constituent proofs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from .acceptance import stage_artifact
from .acceptance_records import Versions, canonical, decode_json
from .claim_review import Evidence
from .deck_evidence import CHECKER, DeckTask, analyze_construction, analyze_deck
from .html_text import source_html_to_text
from .latin_library import ProcessingUnavailable
from .lexical_evidence import BennettLexicon, construction_references

if TYPE_CHECKING:
    from .pipeline_consumers import PhraseObjective, Source


def phrase_claims(
    source: Source, objective: PhraseObjective, versions: Versions, root: Path | None = None
) -> tuple[dict[str, object], tuple[Evidence, ...], str]:
    """Recompute from current retained bytes, not saved decisions or model answers."""
    if root is None:
        with TemporaryDirectory(prefix="latinitas-phrase-proof-") as directory:
            return phrase_claims(source, objective, versions, Path(directory))
    from .pipeline_consumers import _cell, _entry, _form

    pattern = next(p for p in source.data.patterns if p.key == objective.selection.pattern)
    try:
        library = BennettLexicon()
        references = construction_references(pattern.family)
    except ProcessingUnavailable:
        return {}, (), "lexical_reference_unavailable"
    proof_versions = versions.model_copy(update={"analyzer": library.version, "checker": CHECKER})
    components: dict[str, object] = {}
    stage_ids = {}
    objections = []
    evidence: list[Evidence] = list(references)
    for selected in objective.selection.bindings:
        entry, sense = _entry(source, selected.entry, selected.sense)
        form = _form(source, entry, selected.form)
        record = source.records[entry.identity]
        meaning_field = source.deck.fields.meaning_field
        if meaning_field is None:
            return {}, tuple(evidence), "missing_confirmed_meaning_mapping"
        analysis = analyze_deck(
            root,
            record=record,
            profile=source.deck,
            source_scope=source.scope,
            task=DeckTask(
                form=_cell(
                    record,
                    (source.deck.fields.principal_parts_field, source.deck.fields.lexical_entry_field),
                    form.text,
                ),
                meaning=_cell(record, (meaning_field,), sense.meaning),
                pos=entry.pos,
                features=form.features,
                source_role=form.source_role,
                selected_sense=sense.key,
                preposition=None,
                extra_assertions=(),
                interpretation="constructed",
            ),
            versions=proof_versions,
            library=library,
            manifest_identity=source.identities[entry.identity].value
            if source.deck.source_identity.strategy == "manifest"
            else None,
        )
        report = stage_artifact(root, analysis.stage)
        evidence.extend(e for check in analysis.checks for e in check.evidence)
        evidence.append(
            Evidence("source-bound-constituent", selected.role, library.version, canonical(report).decode())
        )
        lemma = source_html_to_text(record.fields[source.deck.fields.lexical_entry_field]).strip()
        if not any(
            s.key == sense.key and s.meaning == sense.meaning and s.valency == sense.valency
            for s in library.senses(lemma, source.deck.language_tag)
        ):
            objections.append(selected.role + ":unqualified_selected_sense_or_valency")
        stage_ids[selected.role] = analysis.stage.stage_id
        components[selected.role] = {
            "identity": asdict(source.identities[entry.identity]),
            "entry": entry,
            "sense": sense,
            "form": form,
        }
    construction = analyze_construction(root, pattern=pattern, components=stage_ids, versions=proof_versions)
    construction_check = construction.checks[0]
    evidence.extend(construction_check.evidence)
    objections.extend(construction_check.objections)
    constraints = tuple(dict.fromkeys(evidence))
    if construction_check.result != "pass" or objections:
        return {}, constraints, ";".join(objections) or "unsupported_construction"
    # Match the structured assertion, including source identities and contextual input.
    expected = decode_json(canonical({"pattern": pattern, "components": components}))
    return {"construction": expected}, constraints, "independently_attested_constructed_reading"
