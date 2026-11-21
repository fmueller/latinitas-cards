"""Finite source propositions and independently checked prompt/answer projections.

The renderer is not called here. Source applicability, whole utterance meaning,
assertion inventory and displayed projection are separate obligations. Unlisted
sentences remain unresolved even when both providers supply the same sentence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .acceptance_records import Assertion, canonical, decode_json
from .claim_review import Evidence
from .html_text import source_html_to_text
from .latin_library import ProcessingUnavailable, attested_paradigms
from .lexical_evidence import BennettLexicon
from .meaning_language import (
    CLAUSES,
    COMPATIBLE_LOVE,
    CONTEXTS,
    LOVE,
    PHRASES,
    ROLE_CONFLICT,
    ROLES,
    VERSION,
    meaning_references,
)

if TYPE_CHECKING:
    from .phrase_inputs import EntryV2
    from .pipeline_consumers import Rendered, Source

type MeaningRelation = Literal["equivalent", "compatible_alternatives", "contradiction", "ambiguous", "unresolved"]


def _json(value: str) -> object:
    try:
        return decode_json(value)
    except ValueError:
        return None


def _translations(value: str, allowed: tuple[str, ...]) -> bool:
    parts = value.split(" / ")
    return bool(parts) and len(parts) == len(set(parts)) and all(p in allowed for p in parts)


def _translation_relation(
    assertion: Assertion, sense: str, allowed: tuple[str, ...], compatible: tuple[str, ...], conflicts: tuple[str, ...]
) -> MeaningRelation:
    if assertion.selected_sense != sense:
        return "contradiction"
    parts = tuple(assertion.value.split(" / "))
    if len(parts) != len(set(parts)) or parts[1:] != assertion.alternatives:
        return "unresolved"
    relations: set[MeaningRelation] = set()
    for part in parts:
        if part in allowed:
            relations.add("equivalent")
        elif part in compatible:
            relations.add("compatible_alternatives")
        elif part in conflicts:
            relations.add("contradiction")
        else:
            return "unresolved"
    return next(iter(relations)) if len(relations) == 1 else "ambiguous"


def _qualified_entry(source: Source, entry: EntryV2, library: BennettLexicon, language: str) -> bool:
    from .pipeline_consumers import _form

    lemma = source_html_to_text(
        source.records[entry.identity].fields.get(source.deck.fields.lexical_entry_field, "")
    ).strip()
    senses = library.senses(lemma, language)
    if not all(
        any(s.key == given.key and s.meaning == given.meaning and s.valency == given.valency for s in senses)
        for given in entry.senses
    ):
        return False
    anchors = [a for p in attested_paradigms() if p.lemma == lemma and p.pos == entry.pos for a in p.anchors]
    for form in entry.forms:
        try:
            _form(source, entry, form.key)
        except ValueError:
            return False
        choices = [r.features for r in library.analyze(form.text) if r.lemma == lemma and r.pos == entry.pos]
        choices.extend(a.features for a in anchors if a.surface == form.text)
        if not any(
            (entry.gender is None if entry.pos != "noun" else entry.gender == f.gender)
            and all(getattr(f, k) == v for k, v in form.features.model_dump(exclude_none=True).items())
            for f in choices
        ):
            return False
    return True


def meaning_checks(
    source: Source, card: Rendered, assertions: tuple[Assertion, ...], structural: dict[str, object]
) -> tuple[dict[str, MeaningRelation], tuple[Evidence, ...]]:
    """Return qualified assertion verdicts, never broaden an unsupported recipe.

    Only equivalent establishes a claim. Other finite dispositions preserve known
    conflicts/compatible readings; unlisted wording remains unresolved.
    """
    from .pipeline_consumers import FormObjective, _entry, _form

    language = card.binding.language
    objective = card.objective
    try:
        references = meaning_references()
    except ProcessingUnavailable:
        return {}, ()
    if isinstance(objective, FormObjective):
        if objective.recipe != "form_translation" or not objective.form or objective.context or objective.anchors:
            return {}, references
    elif objective.recipe != "phrase_translation" or "construction" not in structural:
        return {}, references
    try:
        library = BennettLexicon()
    except ProcessingUnavailable:
        return {}, ()
    source_value: object = None
    scope = ""
    sense_key = ""
    surface = ""
    if isinstance(objective, FormObjective):
        assert objective.form is not None
        entry, sense = _entry(source, objective.entry, objective.sense)
        selected = _form(source, entry, objective.form)
        lemma = source_html_to_text(
            source.records[entry.identity].fields.get(source.deck.fields.lexical_entry_field, "")
        ).strip()
        if not _qualified_entry(source, entry, library, language):
            return {}, references
        if lemma != "amō" or entry.pos != "verb" or entry.gender is not None or len(entry.senses) != 1:
            return {}, references
        if not any(r.lemma == lemma and r.features == selected.features for r in library.analyze(selected.text)):
            return {}, references
        f = selected.features
        if (f.tense, f.mood, f.voice) != ("present", "indicative", "active") or f.person is None:
            return {}, references
        index = int(f.person) - 1 + (3 if f.number == "plural" else 0)
        allowed = LOVE[language][index]
        compatible: tuple[str, ...] = (COMPATIBLE_LOVE[language][index],)
        conflicts = tuple(
            text
            for cell in LOVE[language]
            for text in cell
            if text not in allowed and not (language == "en" and f.person == "2" and text == "you love")
        )
        scope, sense_key, surface = "present-active", sense.key, selected.text
        source_value = decode_json(canonical({"entry": entry, "sense": sense}))
        structural = {
            **structural,
            "source": source_value,
            "target": decode_json(
                canonical(
                    {
                        "surface": surface,
                        "features": f,
                        "source_role": selected.source_role,
                        "anchors": [],
                        "rule": None,
                    }
                )
            ),
        }
    else:
        pattern = next(p for p in source.data.patterns if p.key == objective.selection.pattern)
        if objective.reference.contexts != (CONTEXTS[pattern.family][language],):
            return {}, references
        cases: dict[str, dict[str, tuple[str, str, str, str | None]]] = {
            "preposition-noun": {"object": ("porta", "portā", "gate", "ablative")},
            "noun-genitive": {
                "head": ("porta", "porta", "gate", "nominative"),
                "dependent": ("hortus", "hortī", "garden", "genitive"),
            },
            "adjective-noun": {
                "head": ("hortus", "hortus", "garden", "nominative"),
                "modifier": ("bonus", "bonus", "good", "nominative"),
            },
            "verb-noun": {
                "subject": ("puer", "puer", "boy", "nominative"),
                "object": ("porta", "portam", "gate", "accusative"),
                "verb": ("amō", "amat", "love", None),
            },
        }
        fixed = cases[pattern.family]
        if set(fixed) != {b.role for b in objective.selection.bindings}:
            return {}, references
        if pattern.function_words != (("cum",) if pattern.family == "preposition-noun" else ()):
            return {}, references
        senses_by_role = {}
        surfaces = {}
        for b in objective.selection.bindings:
            entry, sense = _entry(source, b.entry, b.sense)
            if not _qualified_entry(source, entry, library, language):
                return {}, references
            form = _form(source, entry, b.form)
            lemma = source_html_to_text(
                source.records[entry.identity].fields.get(source.deck.fields.lexical_entry_field, "")
            ).strip()
            if (lemma, form.text, sense.key, form.features.case) != fixed[b.role]:
                return {}, references
            if form.features.number != "singular" or (
                b.role == "verb"
                and (form.features.person, form.features.tense, form.features.mood, form.features.voice)
                != ("3", "present", "indicative", "active")
            ):
                return {}, references
            senses_by_role[b.role], surfaces[b.role] = sense.key, form.text
        scope = pattern.family
        allowed = PHRASES[scope][language]
        compatible = ()
        conflicts = ROLE_CONFLICT[language] if scope == "verb-noun" else ()
        sense_key = canonical(senses_by_role).decode()
        order = {
            "preposition-noun": ("object",),
            "noun-genitive": ("head", "dependent"),
            "adjective-noun": ("head", "modifier"),
            "verb-noun": ("subject", "object", "verb"),
        }[scope]
        surface = " ".join((*pattern.function_words, *(surfaces[r] for r in order)))

    references = (
        *references,
        Evidence(
            "source-grounded-meaning",
            surface,
            VERSION,
            canonical(
                {
                    "scope": scope,
                    "language": language,
                    "selected_sense": sense_key,
                    "utterances": allowed,
                    "compatible_utterances": compatible,
                    "conflicting_utterances": conflicts,
                    "clause_proposition": scope,
                    "interpretation": "isolated finite predicate"
                    if isinstance(objective, FormObjective)
                    else objective.reference.contexts,
                }
            ).decode(),
        ),
    )
    verdicts: dict[str, bool] = {}
    translation_relations: dict[str, MeaningRelation] = {}
    by_id = {a.id: a for a in assertions}
    required = {a.id for a in card.claims.assertions}
    kinds = {a.id: a.kind for a in card.claims.assertions}
    for a in assertions:
        if a.id not in kinds or a.kind != kinds[a.id]:
            verdicts[a.id] = False
            continue
        if a.id in structural:
            verdicts[a.id] = (
                a.kind == "label"
                and a.selected_sense is None
                and not a.alternatives
                and _json(a.value) == structural[a.id]
            )
        elif a.kind == "translation":
            translation_relations[a.id] = _translation_relation(a, sense_key, allowed, compatible, conflicts)
            verdicts[a.id] = translation_relations[a.id] == "equivalent"
        elif a.id.startswith("roles-"):
            verdicts[a.id] = (
                a.kind == "explanation"
                and not a.alternatives
                and _json(a.value) == {r: values[language] for r, values in ROLES[scope].items()}
            )
        elif a.kind == "explanation":
            clause = "bounded" if a.id.startswith("caveat-") else scope
            verdicts[a.id] = not a.alternatives and a.value in CLAUSES[clause][language]

    rendered = by_id.get("rendered")
    if rendered is not None:
        projected = _json(rendered.value)
        valid = (
            set(by_id) == required
            and all(verdicts.get(i, False) for i in required - {"rendered"})
            and rendered.kind == "label"
            and not rendered.alternatives
            and isinstance(projected, dict)
            and set(projected) == {"prompt", "answer"}
        )
        if valid and isinstance(projected, dict):
            answer = projected["answer"]
            prompt = projected["prompt"]
            if isinstance(objective, FormObjective):
                instruction = "Übersetze." if language == "de" else "Translate."
                guidance = (
                    "Referenz als Orientierung; Selbstbeurteilung."
                    if language == "de"
                    else "Reference guidance; self-assessment."
                )
                valid = prompt == surface + "\n" + instruction and isinstance(answer, str)
                if valid and isinstance(answer, str):
                    lines = answer.split("\n")
                    clauses = [a for a in card.claims.assertions if a.kind == "explanation"]
                    valid = len(lines) == 2 + len(clauses) and _translations(lines[0], allowed) and lines[1] == guidance
                    valid = valid and all(
                        text in CLAUSES["bounded" if a.id.startswith("caveat-") else scope][language]
                        for text, a in zip(lines[2:], clauses, strict=False)
                    )
            else:
                instruction = (
                    "Übersetze und vergleiche anschließend mit der Referenz."
                    if language == "de"
                    else "Translate, then compare with the reference."
                )
                guidance = (
                    "Selbsteinschätzung: Vergleiche den Sinn, nicht den genauen Wortlaut."
                    if language == "de"
                    else "Self-assessment: compare the meaning, not the exact wording."
                )
                valid = prompt == surface + "\n" + instruction + "\n" + objective.reference.contexts[0] and isinstance(
                    answer, str
                )
                if valid and isinstance(answer, str):
                    lines = answer.split("\n")
                    count = len(objective.reference.readings)
                    valid = len(lines) == 2 * count + 1 and lines[-1] == guidance
                    for index in range(count):
                        prefix = ("Lesart " if language == "de" else "Reading ") + str(index + 1) + ": "
                        valid = valid and lines[2 * index].startswith(prefix)
                        valid = valid and _translations(lines[2 * index][len(prefix) :], allowed)
                        valid = valid and lines[2 * index + 1] in CLAUSES[scope][language]
        verdicts["rendered"] = bool(valid)
    results: dict[str, MeaningRelation] = {
        key: "equivalent" if value else "unresolved" for key, value in verdicts.items()
    }
    results.update(translation_relations)
    return results, references
