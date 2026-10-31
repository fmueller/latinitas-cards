"""latin-rank/1 factors, deterministic ordering and kept-first greedy fill."""

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from .cli_contract import Code, ContractError
from .corpus_exercises import Candidate, exercise_candidates
from .corpus_facts import FactView, Morphology, PassageFacts
from .corpus_model import latin_fold
from .corpus_profile import CorpusProfile
from .corpus_selection import CandidateEntry, DeckLink, Factors, Reason, SelectedEntry, Selection
from .identity import ResolvedSourceIdentity

SINGLE_WORD = {"corpus_lexical_recognition", "corpus_guided_analysis"}


def matching_lemma(lemma: str) -> str:
    return latin_fold(re.sub(r" \([0-9]+\)$", "", lemma))


@dataclass
class Ranked:
    candidate: Candidate
    entry: CandidateEntry
    work: tuple[str, str] | None
    order: tuple[int | str, ...]
    words: Counter[str]
    lemma: str | None


def _rank(
    candidate: Candidate,
    facts: PassageFacts,
    profile: CorpusProfile,
    reading: set[str],
    known: Mapping[str, tuple[ResolvedSourceIdentity, ...]],
    stored: SelectedEntry | None,
) -> Ranked:
    corpus_id = candidate.passage_id.partition(":")[0]
    work = (corpus_id, facts.passage.work)
    words: list[int] = []
    target_index: int | None = None
    forms: Counter[str] = Counter()
    lemma = None
    factors = contributions = None
    score = None
    reasons: list[Reason] = list(candidate.reasons)
    unit_index = 1_000_000
    if candidate.unit is not None and facts.segmentation is not None:
        if candidate.unit["kind"] == "sentence":
            unit_index = int(candidate.unit["index"])
            start, end = facts.sentences[unit_index].span
        else:
            start, end = 0, len(facts.passage.text)
        words = [
            i
            for i, token in enumerate(facts.segmentation.tokens)
            if token.kind == "word" and start <= token.span[0] and token.span[1] <= end
        ]
        lemmas: dict[int, str] = {}
        for position, index in enumerate(words):
            token = facts.segmentation.tokens[index]
            value = facts.morphology(index).value
            if isinstance(value, Morphology):
                lemmas[index] = value.lemma
            if candidate.span is not None and token.span[0] == candidate.span[0]:
                target_index = position
                if candidate.recipe in SINGLE_WORD:
                    lemma = latin_fold(value.lemma) if isinstance(value, Morphology) else None
            if candidate.recipe != "corpus_phrase_translation" or (
                candidate.span is not None and candidate.span[0] <= token.span[0] and token.span[1] <= candidate.span[1]
            ):
                forms[latin_fold(facts.passage.text[token.span[0] : token.span[1]])] += 1
        single_target = words[target_index] if candidate.recipe in SINGLE_WORD and target_index is not None else None
        unknown = sum(
            index != single_target and (index not in lemmas or matching_lemma(lemmas[index]) not in known)
            for index in words
        )
        in_reading = int(candidate.passage_id in reading)
        filtered: Reason | None = None
        if candidate.filtered:
            filtered = candidate.filtered
        elif not profile.limits.min_tokens <= len(words) <= profile.limits.max_tokens:
            filtered = "filtered:length"
        elif profile.limits.max_unknown is not None and unknown > profile.limits.max_unknown:
            filtered = "filtered:unknown"
        elif profile.reading_only and not in_reading:
            filtered = "filtered:reading"
        if filtered:
            reasons = [filtered, *reasons]
        if stored is not None and stored.context_digest != candidate.context_digest:
            reasons = ["context_changed", *reasons]
        if not reasons:
            targets = {latin_fold(value) for value in profile.targets}
            target_lemma = matching_lemma(lemmas[single_target]) if single_target in lemmas else None
            target = (
                int(target_lemma in targets)
                if candidate.recipe in SINGLE_WORD
                else min(3, len({matching_lemma(value) for value in lemmas.values()} & targets))
            )
            factors = Factors(
                reading=in_reading,
                target=target,
                novelty=int(target_lemma not in known) if candidate.recipe in SINGLE_WORD else 0,
                length=-abs(len(words) - profile.limits.ideal_tokens),
                unknown=-unknown,
                coverage=100 * len(lemmas) // len(words),
            )
            weights = profile.weights.model_dump()
            values = {key: value * weights[key] for key, value in factors.model_dump().items()}
            contributions = Factors.model_validate(values)
            score = sum(values.values())
    entry = CandidateEntry.model_validate(
        {
            "object_key": candidate.object_key,
            "latinitas_id": candidate.latinitas_id,
            "recipe": candidate.recipe,
            "passage_id": candidate.passage_id,
            "unit": candidate.unit,
            "target": candidate.target,
            "annotation_id": candidate.annotation_id,
            "eligible": not reasons,
            "reasons": reasons,
            "factors": factors,
            "contributions": contributions,
            "score": score,
            "outcome": reasons[0] if reasons else "selected",
            "deck_links": [
                DeckLink(identity=identity.value, scope=identity.scope)
                for identity in known.get(matching_lemma(lemma), ())
            ]
            if lemma is not None
            else [],
        }
    )
    if candidate.annotation_id is not None:
        order: tuple[int | str, ...] = (2, candidate.annotation_id)
    elif reasons:
        order = (1, candidate.latinitas_id or "")
    else:
        assert score is not None
        order = (
            0,
            -score,
            corpus_id,
            facts.passage.order,
            unit_index,
            target_index if target_index is not None else 1_000_000,
            candidate.latinitas_id or "",
        )
    return Ranked(candidate, entry, work, order, forms, lemma)


def rank_selection(
    profile: CorpusProfile,
    views: dict[str, FactView],
    selection: Selection,
    reading: set[str],
    known: Mapping[str, tuple[ResolvedSourceIdentity, ...]],
    confirm: set[str],
) -> tuple[Selection, list[CandidateEntry]]:
    """Keep every selected identity, then apply exclusions to remaining eligible candidates."""
    stored = {entry.latinitas_id: entry for entry in selection.selected}
    occurrences = [(view, candidate) for view in views.values() for candidate in exercise_candidates(view, profile)]
    candidates = {
        candidate.latinitas_id: candidate for _, candidate in occurrences if candidate.latinitas_id is not None
    }
    for identity in sorted(confirm):
        if identity not in stored or identity not in candidates:
            raise ContractError(Code.UNKNOWN_EXERCISE, f"No current selected exercise to confirm: {identity}")
    changed = False
    for identity in sorted(confirm):
        current = candidates[identity].context_digest
        assert current is not None
        if stored[identity].context_digest != current:
            stored[identity] = stored[identity].model_copy(
                update={"context_digest": current, "run": selection.runs + 1}
            )
            changed = True
    ranked = [
        _rank(
            candidate,
            view.passages[candidate.passage_id],
            profile,
            reading,
            known,
            stored.get(candidate.latinitas_id or ""),
        )
        for view, candidate in occurrences
    ]
    found = {row.candidate.latinitas_id for row in ranked}
    for identity, old in stored.items():
        if identity in found:
            continue
        candidate = Candidate(
            recipe=old.recipe,
            passage_id=old.passage_id,
            unit=old.unit.model_dump(),
            target=old.target.model_dump() if old.target else None,
            span=None,
            reasons=(),
            object_key=old.object_key,
            latinitas_id=identity,
            context_digest=old.context_digest,
        )
        entry = CandidateEntry.model_validate(
            old.model_dump(exclude={"context_digest", "run"})
            | {
                "eligible": False,
                "reasons": ["target_missing"],
                "factors": None,
                "contributions": None,
                "score": None,
                "outcome": "kept",
            }
        )
        passage = views.get(old.passage_id.partition(":")[0])
        facts = passage.passages.get(old.passage_id) if passage else None
        work = (old.passage_id.partition(":")[0], facts.passage.work) if facts else None
        ranked.append(Ranked(candidate, entry, work, (1, identity), Counter(), None))
    ranked.sort(key=lambda row: row.order)
    kept = [row for row in ranked if row.candidate.latinitas_id in stored]
    recipe_counts = Counter(row.entry.recipe for row in kept)
    work_counts = Counter(row.work for row in kept)
    recipe_limits = {entry.recipe: entry.budget for entry in profile.recipes}
    deselected = {entry.latinitas_id for entry in selection.deselected}
    added = 0
    for row in kept:
        row.entry.outcome = "kept"
    for row in ranked:
        entry, candidate = row.entry, row.candidate
        if candidate.latinitas_id in stored or not entry.eligible:
            continue
        reason: Reason | None = None
        if candidate.latinitas_id in deselected:
            reason = "deselected"
        elif recipe_counts[entry.recipe] >= recipe_limits[entry.recipe]:
            reason = "budget:recipe"
        elif work_counts[row.work] >= profile.budgets.per_work:
            reason = "budget:work"
        elif added >= profile.budgets.per_run:
            reason = "budget:run"
        elif any(
            other.entry.recipe == entry.recipe
            and other.words
            and 5 * sum((row.words & other.words).values()) >= 4 * sum((row.words | other.words).values())
            for other in kept
        ):
            reason = "near_duplicate"
        elif entry.recipe == "corpus_lexical_recognition" and any(
            other.entry.recipe == entry.recipe and row.lemma is not None and row.lemma == other.lemma for other in kept
        ):
            reason = "per_lemma"
        if reason:
            entry.outcome = reason
            continue
        assert candidate.latinitas_id is not None
        stored[candidate.latinitas_id] = SelectedEntry.model_validate(
            {
                "latinitas_id": candidate.latinitas_id,
                "object_key": candidate.object_key,
                "recipe": candidate.recipe,
                "passage_id": candidate.passage_id,
                "unit": candidate.unit,
                "target": candidate.target,
                "context_digest": candidate.context_digest,
                "run": selection.runs + 1,
            }
        )
        entry.outcome = "selected"
        kept.append(row)
        recipe_counts[entry.recipe] += 1
        work_counts[row.work] += 1
        added += 1
    return selection.model_copy(
        update={
            "runs": selection.runs + int(changed or added > 0),
            "selected": sorted(stored.values(), key=lambda entry: entry.latinitas_id),
        }
    ), [row.entry for row in ranked]
