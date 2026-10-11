"""Generate bounded proposals from explicit saved bindings, never infer forms."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import Field

from .claim_sidecar import claim_report
from .phrase_inputs import (
    Binding,
    Key,
    Limits,
    Model,
    PhraseWorkspace,
    Selection,
    SelectionRequest,
    bind_selections,
    canonical,
)


class BoundSource(Model):
    value: str = Field(min_length=1, max_length=2048)
    scope: str | None = Field(max_length=2048)


class SavedBinding(Binding):
    role: Key = Field(max_length=256)
    entry: str = Field(min_length=1, max_length=2048)
    sense: Key = Field(max_length=256)
    form: Key = Field(max_length=256)
    source: BoundSource


class SavedSelection(Model):
    pattern: Key = Field(max_length=256)
    bindings: tuple[SavedBinding, ...] = Field(min_length=1, max_length=3)
    source_identity: str = Field(min_length=1, max_length=256)
    source_scope: Literal["phrase-input-v1"]
    object_key: str = Field(min_length=1, max_length=256)


class SavedSelections(Model):
    version: Literal[1]
    limits: Limits
    selections: tuple[SavedSelection, ...] = Field(min_length=1, max_length=1000)


def load_saved_selections(path: Path) -> SavedSelections:
    """Bound bytes before parsing, including when metadata changes during reading."""
    maximum = 4 * 1024 * 1024
    with path.open("rb") as stream:
        contents = stream.read(maximum + 1)
    if len(contents) > maximum:
        raise ValueError("Saved selections exceed 4 MiB; narrow choices or remove excess whitespace.")
    return SavedSelections.model_validate_json(contents)


def generate_proposals(work: PhraseWorkspace, saved: SavedSelections) -> dict[str, object]:
    """One explicit selection is one attempt; even rejected choices consume budget.

    Sorting the bounded saved list is not a lexical/form Cartesian traversal.
    No unselected forms, senses or vocabulary are visited or selected as variants.
    """
    entries = {entry.identity: entry for entry in work.data.entries}
    patterns = {pattern.key: pattern for pattern in work.data.patterns}
    ordered = sorted(
        (s.model_copy(update={"bindings": tuple(sorted(s.bindings, key=lambda b: b.role))}) for s in saved.selections),
        key=lambda s: canonical(s.model_dump(mode="json")),
    )
    candidates: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    variants: Counter[str] = Counter()
    seen: set[str] = set()
    attempts = ordered[: saved.limits.max_exercises]
    for item, choice in enumerate(attempts):
        selection = Selection(
            pattern=choice.pattern,
            bindings=tuple(Binding(role=b.role, entry=b.entry, sense=b.sense, form=b.form) for b in choice.bindings),
        )
        rebound, diagnostics = bind_selections(
            work, SelectionRequest(version=1, limits=Limits(max_exercises=1), selections=(selection,))
        )
        reasons = [asdict(d) for d in diagnostics]
        if not reasons and rebound["selections"] != [choice.model_dump(mode="json")]:
            reasons.append({"code": "stale_selection", "message": "Saved source/key differs; rerun phrases select."})
        if not reasons and choice.object_key in seen:
            continue
        group = canonical(
            {
                "pattern": choice.pattern,
                "bindings": [
                    {"role": b.role, "source": b.source.model_dump(), "sense": b.sense} for b in choice.bindings
                ],
            }
        )
        if not reasons and variants[group] >= saved.limits.variants_per_binding:
            reasons.append({"code": "variant_limit", "message": "Raise variants_per_binding or narrow saved choices."})
        if reasons:
            skipped.append(
                {
                    "item": item,
                    "selection": choice.model_dump(mode="json"),
                    "diagnostics": reasons,
                    "action": "Repair the named binding/input and review current claims, then rerun phrases select.",
                }
            )
            continue
        seen.add(choice.object_key)
        variants[group] += 1
        pattern = patterns[choice.pattern]
        constituents: list[dict[str, object]] = []
        forms: dict[str, str] = {}
        claim_keys = {(f"pattern:{pattern.key}", "pattern")}
        for binding in choice.bindings:
            entry = entries[binding.entry]
            sense = next(s for s in entry.senses if s.key == binding.sense)
            form = next(f for f in entry.forms if f.key == binding.form)
            forms[binding.role] = form.text
            constituents.append(
                {
                    "role": binding.role,
                    "source": binding.source.model_dump(),
                    "entry": binding.entry,
                    "sense": sense.model_dump(mode="json"),
                    "form": form.model_dump(mode="json"),
                    "evidence": [asdict(e) for e in entry.evidence],
                }
            )
            claim_keys.update((entry.identity, f) for f in (f"sense:{sense.key}", f"form:{form.key}"))
            if entry.pos == "noun":
                claim_keys.add((entry.identity, "gender"))
        role_order = {
            "preposition-noun": ("object",),
            "noun-genitive": ("head", "dependent"),
            "adjective-noun": ("head", "modifier"),
            "verb-noun": ("subject", "object", "verb"),
        }[pattern.family]
        candidates.append(
            choice.model_dump(mode="json")
            | {
                "candidate_key": choice.object_key,
                "text": " ".join([*pattern.function_words, *(forms[r] for r in role_order if r in forms)]),
                "family": pattern.family,
                "pattern_version": work.data.version,
                "function_words": list(pattern.function_words),
                "pattern_evidence": [asdict(e) for e in pattern.evidence],
                "constituents": constituents,
                "review_state": "proposal",
                "input_claims": [claim_report(c) for c in work.state.claims if (c.identity, c.feature) in claim_keys],
            }
        )
    unvisited = len(ordered) - len(attempts)
    return {
        "version": 1,
        "limits": saved.limits.model_dump(),
        "candidates": candidates,
        "skipped": skipped,
        "visited": len(attempts),
        "unvisited": unvisited,
        "limit_reason": "Traversal budget reached; narrow selections or raise max_exercises." if unvisited else None,
        "supported_subset": "Explicit reviewed forms only; nominative third-person noun clauses. "
        "No inferred paradigms, naturalness approval, recipes or export.",
    }
