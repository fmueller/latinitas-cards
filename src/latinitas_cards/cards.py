"""The conditional sibling-card contract for learning-object notes.

This module is the authoritative card contract that T-030 extends onto the
T-029 note schema: the exact supported semantic keys (recipe plus confirmed
role), their documented template slots, the per-card note fields, and the
eligibility rules.  The exporter and the reference templates consume this
contract; nothing else may invent slots, renumber ordinals, or re-evaluate
eligibility.  Slot ordinals, template names, and field names are frozen: a new
capability appends a new slot and never repurposes an existing one, and profile
enablement changes eligibility only, never slot order or note identity.
"""

from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Sequence
from dataclasses import dataclass

from .identity import derive_card_semantic_key
from .principal_parts import ParsedPrincipalParts, PrincipalPartValue
from .principal_relationships import PrincipalPartComparison
from .profile import MorphologySettings
from .profile_setup import encode_unsafe_controls

TEMPLATE_REGISTRY_VERSION = 1

COMPLETION_RECIPE = "principal_part_completion"
RECOGNITION_RECIPE = "principal_part_recognition"
SUPPORTED_RECIPES: tuple[str, ...] = (COMPLETION_RECIPE, RECOGNITION_RECIPE)

#: Registry enumeration order is the frozen slot order.  It deliberately
#: follows the canonical Latin principal-part order rather than any profile's
#: confirmed layout order, so reordering or extending a profile layout never
#: renumbers template ordinals.
_REGISTRY_ROLE_ORDER: tuple[str, ...] = (
    "present_1s",
    "present_infinitive",
    "perfect_1s",
    "perfect_passive_participle",
    "supine",
)
_RECIPE_LABELS = {
    COMPLETION_RECIPE: "Completion",
    RECOGNITION_RECIPE: "Recognition",
}
_ROLE_LABELS = {
    "present_1s": "Present",
    "present_infinitive": "Infinitive",
    "perfect_1s": "Perfect",
    "perfect_passive_participle": "PPP",
    "supine": "Supine",
}
_ROLE_DISPLAY_LABELS = {
    "present_infinitive": "Infinitiv",
    "present_1s": "Präsens, 1. Person Singular",
    "perfect_1s": "Perfekt, 1. Person Singular",
    "perfect_passive_participle": "Partizip Perfekt Passiv (PPP)",
    "supine": "Supinum",
}

GUARD_VALUE = "1"
_BLANK_MARKER = "____"


@dataclass(frozen=True, slots=True)
class CardSlot:
    """One frozen template slot: semantic key, template identity, and fields."""

    semantic_key: str
    recipe: str
    role: str
    template_name: str
    ordinal: int
    enabled_field: str
    prompt_field: str
    answer_field: str


def _build_registry() -> tuple[CardSlot, ...]:
    slots: list[CardSlot] = []
    ordinal = 0
    for recipe in SUPPORTED_RECIPES:
        recipe_label = _RECIPE_LABELS[recipe]
        for role in _REGISTRY_ROLE_ORDER:
            role_label = _ROLE_LABELS[role]
            slots.append(
                CardSlot(
                    semantic_key=derive_card_semantic_key(recipe, role),
                    recipe=recipe,
                    role=role,
                    template_name=f"{recipe_label} {role_label}",
                    ordinal=ordinal,
                    enabled_field=f"{recipe_label}{role_label}Enabled",
                    prompt_field=f"{recipe_label}{role_label}Prompt",
                    answer_field=f"{recipe_label}{role_label}Answer",
                )
            )
            ordinal += 1
    return tuple(slots)


TEMPLATE_REGISTRY: tuple[CardSlot, ...] = _build_registry()

_REGISTRY_SLOTS_BY_KEY = {slot.semantic_key: slot for slot in TEMPLATE_REGISTRY}

TEMPLATE_REGISTRY_DIGEST = (
    "card-registry-sha256:"
    + hashlib.sha256(
        json.dumps(
            [
                {
                    "key": slot.semantic_key,
                    "recipe": slot.recipe,
                    "role": slot.role,
                    "template": slot.template_name,
                    "ordinal": slot.ordinal,
                }
                for slot in TEMPLATE_REGISTRY
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
)


def slot_for_key(semantic_key: str) -> CardSlot | None:
    """Return the frozen slot for a semantic key, or ``None`` when unsupported."""

    return _REGISTRY_SLOTS_BY_KEY.get(semantic_key)


def slot_for_recipe_role(recipe: str, role: str) -> CardSlot | None:
    """Return the frozen slot for one recipe and role, or ``None``."""

    return slot_for_key(derive_card_semantic_key(recipe, role))


def role_display_label(role: str) -> str:
    """Return the human-readable semantic label of a confirmed role."""

    return _ROLE_DISPLAY_LABELS.get(role, role)


@dataclass(frozen=True, slots=True)
class RenderedCard:
    """One registry slot with its eligibility and wholly-guarded front data.

    ``guard`` is ``"1"`` exactly when the card is eligible, and the prompt and
    answer carry no content for ineligible cards, so a front template of the
    guarded form ``{{#Enabled}}{{Prompt}}{{/Enabled}}`` can never render a
    blank or static-label-only card.
    """

    slot: CardSlot
    eligible: bool
    guard: str
    prompt: str
    answer: str


def card_is_eligible(
    recipe: str,
    part: PrincipalPartValue,
    parsed: ParsedPrincipalParts,
    *,
    selected_recipes: Sequence[str],
) -> bool:
    """Evaluate one card's eligibility from already-normalized semantic values.

    Both recipes require the recipe to be selected, the role to have a frozen
    slot, and the target role to carry a present, non-omitted answer.
    Recognition additionally requires the lemma, which the parser guarantees
    for every parsed entry.  Completion additionally requires meaningful
    remaining context: at least one other present role in the series.
    """

    if recipe not in selected_recipes or slot_for_recipe_role(recipe, part.role) is None:
        return False
    if part.is_omitted or part.unresolved or not (part.display or "").strip():
        return False
    if recipe == RECOGNITION_RECIPE:
        return bool(parsed.lexical_entry.strip())
    if recipe == COMPLETION_RECIPE:
        return any(other is not part and not other.is_omitted and not other.unresolved for other in parsed.parts)
    return False


def render_cards(
    parsed: ParsedPrincipalParts,
    *,
    selected_recipes: Sequence[str],
    meaning: str = "",
    comparison: PrincipalPartComparison | None = None,
    morphology: MorphologySettings | None = None,
) -> tuple[RenderedCard, ...]:
    """Render every registry slot of one learning object in registry order.

    Eligibility consumes the parser's single normalization result; this module
    escapes display text for rendering but never normalizes or re-decodes it.
    An absent optional gloss is simply omitted, never invented.
    """

    gloss = _escape_text(meaning) if meaning.strip() else ""
    settings = morphology or MorphologySettings()
    cards: list[RenderedCard] = []
    for slot in TEMPLATE_REGISTRY:
        part = parsed.by_role.get(slot.role)
        if part is None or not card_is_eligible(slot.recipe, part, parsed, selected_recipes=selected_recipes):
            cards.append(RenderedCard(slot=slot, eligible=False, guard="", prompt="", answer=""))
            continue
        target = _escape_text(part.display or "")
        if slot.recipe == COMPLETION_RECIPE:
            prompt = _completion_prompt(parsed, part, gloss)
            answer = target
        else:
            prompt = _recognition_prompt(parsed, part, gloss)
            answer = _escape_text(role_display_label(slot.role))
        if comparison is not None:
            answer += _render_comparison(comparison, slot.role, settings, parsed.lexical_entry)
        cards.append(RenderedCard(slot=slot, eligible=True, guard=GUARD_VALUE, prompt=prompt, answer=answer))
    return tuple(cards)


def _render_segmentation(value: str) -> str:
    """Style only the reviewed pipe convention; never derive a linguistic split."""
    pieces = tuple(piece.strip() for piece in value.split("|"))
    if len(pieces) not in (2, 3) or not all(pieces):
        return _escape_text(value)
    classes = ("stem", "ending") if len(pieces) == 2 else ("stem", "marker", "ending")
    return " | ".join(
        f'<span class="morphology-{kind}">{_escape_text(piece)}</span>'
        for kind, piece in zip(classes, pieces, strict=True)
    )


def _render_comparison(
    comparison: PrincipalPartComparison, tested_role: str, settings: MorphologySettings, lemma: str
) -> str:
    lines = []
    related = []
    focused = "Analyse zurückgehalten; einzelne Belege prüfen."
    for role in comparison.roles:
        available = role.status == "present"
        value = (
            _escape_text(role.form or "")
            if available
            else ("— (Nicht vorhanden)" if role.status == "absent" else "— (Zurückgehalten)")
        )
        analysis = " · ".join(
            text
            for text in (
                _render_segmentation(role.segmentation) if available and role.segmentation else "",
                _escape_text(role.explanation) if available and role.explanation else "",
            )
            if text
        )
        if role.role == tested_role:
            focused = analysis or "Analyse zurückgehalten; einzelne Belege prüfen."
        elif available and role.segmentation:
            related.append(f"{_escape_text(role_display_label(role.role))}: {_render_segmentation(role.segmentation)}")
        lines.append(
            f'<div class="morphology-role"><strong>{_escape_text(role_display_label(role.role))}:</strong> {value}'
            + (f" · {analysis}" if analysis else f" · {_escape_text(role.reason)}")
            + "</div>"
        )
    body = "".join(lines)
    further = (
        "<details><summary>Stammformen vergleichen</summary>" + body + "</details>"
        if settings.comparison == "disclosure"
        else '<section class="morphology-static"><h3>Stammformen vergleichen</h3>' + body + "</section>"
    )
    return (
        f'<section class="morphology-v1 morphology-{settings.theme} morphology-{settings.appearance}">'
        + f'<div>Lemma: <span class="morphology-lemma">{_escape_text(lemma)}</span></div>'
        + f'<div class="tested-form-explanation">{focused}</div>'
        + '<div class="related-stems">'
        + ("Verwandte Stämme: " + " · ".join(related) if related else "Weitere Stämme: Analyse zurückgehalten.")
        + "</div>"
        + further
        + "</section>"
    )


def _completion_prompt(parsed: ParsedPrincipalParts, target: PrincipalPartValue, gloss: str) -> str:
    lines = []
    for part in parsed.parts:
        value = _BLANK_MARKER if part is target else ("—" if part.is_omitted else _escape_text(part.display or ""))
        if part.unresolved:
            value = "— (unresolved source evidence)"
        lines.append(f"<strong>{_escape_text(role_display_label(part.role))}:</strong> {value}")
    if gloss:
        lines.append(gloss)
    return "<br>".join(lines)


def _recognition_prompt(parsed: ParsedPrincipalParts, target: PrincipalPartValue, gloss: str) -> str:
    lines = [
        f"<strong>{_escape_text(target.display or '')}</strong>",
        _escape_text(parsed.lexical_entry),
    ]
    if gloss:
        lines.append(gloss)
    return "<br>".join(lines)


def _escape_text(value: str) -> str:
    """Escape semantic display text once and keep it on a single HTML line.

    Line boundaries in normalized display text become ``<br>`` so card fields
    stay physically single-line like every other rendered note field; the text
    itself is never re-normalized or re-decoded.
    """

    return html.escape(encode_unsafe_controls(value), quote=True).replace("\n", "<br>")


__all__ = [
    "COMPLETION_RECIPE",
    "RECOGNITION_RECIPE",
    "RenderedCard",
    "SUPPORTED_RECIPES",
    "TEMPLATE_REGISTRY",
    "TEMPLATE_REGISTRY_DIGEST",
    "TEMPLATE_REGISTRY_VERSION",
    "card_is_eligible",
    "render_cards",
    "role_display_label",
    "slot_for_key",
    "slot_for_recipe_role",
]
