"""The manual reference note-type setup derived from the authoritative contract.

T-033 consumes the T-029/T-030 generated-schema and slot contract; this module
never redefines a field or slot.  The reference field list, the guarded card
templates, and the shared styling are derived from the authoritative note
schema and the frozen template registry, so the published manual setup cannot
drift from what the exporter emits.  This is manual reference material for the
docs, not automated note-type provisioning (that is v0.6.0 roadmap work).
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import TEMPLATE_REGISTRY, TEMPLATE_REGISTRY_DIGEST, TEMPLATE_REGISTRY_VERSION, CardSlot
from .notes import AUTHORITATIVE_NOTE_FIELDS, NOTE_SCHEMA_VERSION

#: The note type covers every authoritative field except transport metadata:
#: `Tags` stays the special CSV tags column, never a regular note field.
REFERENCE_NOTE_TYPE_FIELDS: tuple[str, ...] = tuple(
    field.name for field in AUTHORITATIVE_NOTE_FIELDS if field.ownership != "transport"
)


@dataclass(frozen=True, slots=True)
class ReferenceCardTemplate:
    """One copyable card template of the reference note type."""

    slot: CardSlot
    front: str
    back: str


def _front(slot: CardSlot) -> str:
    return f"{{{{#{slot.enabled_field}}}}}{{{{{slot.prompt_field}}}}}{{{{/{slot.enabled_field}}}}}"


def _back(slot: CardSlot) -> str:
    return (
        f"{{{{#{slot.enabled_field}}}}}\n"
        f'<div class="latinitas-answer">{{{{{slot.answer_field}}}}}</div>\n'
        '<div class="latinitas-context">{{Lemma}} &#183; {{Principal Parts}}'
        "{{#Meaning}} &#183; {{Meaning}}{{/Meaning}}</div>\n"
        '{{#Personal Notes}}<div class="latinitas-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}\n'
        f"{{{{/{slot.enabled_field}}}}}"
    )


REFERENCE_CARD_TEMPLATES: tuple[ReferenceCardTemplate, ...] = tuple(
    ReferenceCardTemplate(slot=slot, front=_front(slot), back=_back(slot)) for slot in TEMPLATE_REGISTRY
)

REFERENCE_STYLE_VERSION = 2

REFERENCE_CARD_CSS = """\
/* Latinitas reference styling v2; morphology markup v1. Manual setup only. */
.card {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 22px;
  text-align: center;
  color: #1a1a1a;
  background-color: #ffffff;
}

.latinitas-answer {
  font-size: 30px;
  font-weight: bold;
}

.latinitas-context {
  margin-top: 0.6em;
  font-size: 18px;
  color: #444444;
}

.latinitas-personal-notes {
  margin-top: 1em;
  padding-top: 0.5em;
  border-top: 1px solid #cccccc;
  font-size: 16px;
  font-style: italic;
  color: #666666;
  text-align: left;
}

.morphology-v1 {
  --stem: #356357;
  --marker: #785523;
  --ending: #5d5079;
  margin: 0.8em auto 0;
  padding: 1em;
  max-width: 42em;
  font-size: 18px;
  line-height: 1.5;
  font-weight: normal;
  text-align: left;
  color: #222222;
  background: #f7f7f4;
  border: 1px solid #bdbdb7;
  border-radius: 0.3em;
  overflow-wrap: anywhere;
}
.morphology-v1.morphology-dark {
  --stem: #9bc9b9;
  --marker: #e1bf85;
  --ending: #c4b5e1;
  color: #eeeeee;
  background: #222526;
  border-color: #666666;
}
.morphology-v1.morphology-monochrome {
  --stem: currentColor;
  --marker: currentColor;
  --ending: currentColor;
}
.morphology-lemma { font-style: italic; }
.morphology-stem { color: var(--stem); font-weight: bold; }
.morphology-marker { color: var(--marker); border-bottom: 1px dotted; }
.morphology-ending { color: var(--ending); text-decoration: underline; }
.tested-form-explanation, .related-stems { margin-top: 0.5em; }
.morphology-v1 summary, .morphology-static h3 {
  margin: 0.8em 0 0.3em;
  font-size: 1em;
  font-weight: bold;
}
.morphology-v1 summary { cursor: pointer; }
.morphology-role { margin: 0.4em 0; }
"""


def reference_setup_markdown() -> str:
    """Render the copyable reference setup block for the published docs."""

    lines = [
        f"Reference setup for note schema `{NOTE_SCHEMA_VERSION}` and template registry "
        f"`v{TEMPLATE_REGISTRY_VERSION}` (digest `{TEMPLATE_REGISTRY_DIGEST}`).",
        "This block is generated from the authoritative contract; a test fails if this",
        "published copy drifts from it. Do not hand-edit inside the markers.",
        f"Reference styling `v{REFERENCE_STYLE_VERSION}` supports morphology markup `v1`.",
        "CSV carries managed markup, never installs CSS or provisions/rebinds templates.",
        "Existing destinations with a different CSS/template digest are blocked for",
        "separately reviewed manual setup; do not rewrite scheduled templates.",
        "",
        f"### Reference fields ({len(REFERENCE_NOTE_TYPE_FIELDS)}, exact order)",
        "",
        "Create exactly these fields, in this order, on the dedicated note type.",
        "`LatinitasID` must stay the first field: Anki matches notes for update by the",
        "first field. `Personal Notes` stays last. `Tags` is deliberately absent: it is",
        "the special CSV transport column, never a note-type field.",
        "",
        "```text",
        *REFERENCE_NOTE_TYPE_FIELDS,
        "```",
        "",
        f"### Card templates ({len(REFERENCE_CARD_TEMPLATES)}, frozen registry order)",
        "",
        "Create one card template per entry below with exactly this name, front, and",
        "back. Every front is wholly guarded by its per-card `Enabled` field, so a form",
        "the exporter marks ineligible renders an empty front and creates no card",
        "instead of a blank or label-only card. Every back is guarded the same way and",
        "renders the managed answer, shared managed context (`Lemma`, `Principal Parts`,",
        "optional `Meaning`), and the note-level, user-owned `Personal Notes`.",
        "",
    ]
    for template in REFERENCE_CARD_TEMPLATES:
        slot = template.slot
        lines.extend(
            [
                f"#### `{slot.template_name}` &#8212; ordinal {slot.ordinal} &#8212; `{slot.semantic_key}`",
                "",
                "Front:",
                "",
                "```html",
                template.front,
                "```",
                "",
                "Back:",
                "",
                "```html",
                template.back,
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "### Shared styling (paste once into the note type's Styling)",
            "",
            "```css",
            REFERENCE_CARD_CSS.rstrip("\n"),
            "```",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "REFERENCE_CARD_CSS",
    "REFERENCE_CARD_TEMPLATES",
    "REFERENCE_NOTE_TYPE_FIELDS",
    "ReferenceCardTemplate",
    "reference_setup_markdown",
]
