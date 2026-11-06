"""Bounded source evidence, not linguistic analysis or answer selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .html_text import source_html_to_text
from .profile import PrincipalPartLayout


@dataclass(frozen=True, slots=True)
class SourceHint:
    position: int
    raw: str
    text: str


@dataclass(frozen=True, slots=True)
class SourceExtraction:
    raw: str
    roles: tuple[str, ...]
    status: Literal["supported", "ambiguous", "unsupported", "incomplete"]
    candidates: tuple[tuple[str, ...], ...] | None
    hints: tuple[SourceHint, ...] = ()
    rules: tuple[str, ...] = ()


def separator_rules(layout: PrincipalPartLayout) -> tuple[str, ...]:
    names = {
        ",": "literal comma",
        ";": "literal semicolon",
        " — ": "literal spaced em dash",
        " / ": "literal spaced slash",
    }
    return tuple(names.get(separator, f"literal {separator!r}") for separator in layout.separators)


def mask_parenthetical_notes(value: str) -> str:
    """Blank annotations without moving the raw source's role boundaries."""
    depth = 0
    masked: list[str] = []
    for char in value:
        if char == "(":
            depth += 1
        masked.append(" " if depth or char == ")" else char)
        if char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parenthetical note")
    if depth:
        raise ValueError("Unbalanced parenthetical note")
    return "".join(masked)


def extract_segments(raw: str, segments: tuple[str, ...], layout: PrincipalPartLayout) -> SourceExtraction:
    """Retain ordered alternatives and only the explicitly confirmed poet. hint.

    Unconfirmed pipe evidence stays unresolved too; opting in controls whether
    it can be split into candidates, not whether it can become a study fact.
    """
    rules = list(separator_rules(layout))
    candidates: list[tuple[str, ...]] = []
    hints: list[SourceHint] = []
    for position, segment in enumerate(segments, start=1):
        try:
            masked_text = mask_parenthetical_notes(source_html_to_text(segment))
        except ValueError:
            return SourceExtraction(raw, layout.roles, "unsupported", None, rules=("unbalanced parenthetical note",))
        text = "\n".join(" ".join(line.split()) for line in masked_text.split("\n") if line.strip())
        lines = text.split("\n")
        if len(lines) > 1 and layout.trailing_poet_hint:
            raw_lines = re.split(r"<br\s*/?>|\n", segment, flags=re.IGNORECASE)
            if (
                len(raw_lines) != 2
                or re.sub(r"[^\S\n]+", " ", mask_parenthetical_notes(source_html_to_text(raw_lines[0]))).strip()
                != lines[0]
            ):
                return SourceExtraction(
                    raw,
                    layout.roles,
                    "unsupported",
                    None,
                    rules=(*rules, "unconfirmed hint boundary", "withhold assignment"),
                )
            hint_raw = raw_lines[1].strip()
            hint_text = source_html_to_text(hint_raw)
            hints.append(SourceHint(position, hint_raw, hint_text))
            if position != 3 or layout.roles[position - 1] != "perfect_1s" or hint_text != "poet.":
                return SourceExtraction(
                    raw,
                    layout.roles,
                    "unsupported",
                    None,
                    tuple(hints),
                    ("retain conflicting hint", "withhold assignment"),
                )
            text = lines[0]
        if "|" in text:
            if not layout.pipe_alternatives:
                candidates.append((text,))
                continue
            alternatives = tuple(candidate.strip() for candidate in text.split("|"))
            if not all(alternatives):
                return SourceExtraction(
                    raw, layout.roles, "unsupported", None, tuple(hints), (*rules, "reject empty alternative")
                )
            candidates.append(alternatives)
        else:
            candidates.append((text,) if text else ())
    has_alternatives = any(len(slot) > 1 for slot in candidates)
    has_omission = any(not slot for slot in candidates)
    if hints:
        rules.extend(("safe HTML line boundary", "confirmed trailing poet. hint"))
    elif "<" in raw:
        rules.append("safe HTML text")
    if "&" in raw:
        rules.append("decode entities once")
    if layout.pipe_alternatives and has_alternatives:
        rules.append("confirmed pipe alternatives within each slot")
    if not hints or has_alternatives or has_omission:
        rules.append("explicit omission" if has_omission else "trim")
    if layout.pipe_alternatives and has_alternatives:
        rules.append("preserve alternative order")
    if len(layout.roles) == 3:
        rules.append("confirmed three-role order")
    return SourceExtraction(raw, layout.roles, "supported", tuple(candidates), tuple(hints), tuple(rules))
