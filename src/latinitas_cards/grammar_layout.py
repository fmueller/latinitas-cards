"""Explicit append-only grammar layout proposal; never installs or migrates Anki."""

from dataclasses import asdict

from .acceptance_records import digest
from .cards import CardSlot
from .phrase_recipes import PHRASE_LAYOUT, PHRASE_LAYOUT_FIELDS
from .reference_templates import REFERENCE_CARD_CSS, ReferenceCardTemplate

FORM_RECIPES = (
    "form_translation",
    "form_recognition",
    "ppp_translation",
    "ppp_recognition",
    "verb_production",
    "noun_declension",
    "adjective_declension",
    "form_cloze",
)


def _template(recipe: str, ordinal: int) -> ReferenceCardTemplate:
    slot = CardSlot(
        semantic_key=recipe + ":standalone",
        recipe=recipe,
        role="standalone",
        template_name=recipe,
        ordinal=ordinal,
        enabled_field=f"Grammar{ordinal}Enabled",
        prompt_field=f"Grammar{ordinal}Prompt",
        answer_field=f"Grammar{ordinal}Answer",
    )
    return ReferenceCardTemplate(
        slot=slot,
        front=f"{{{{#{slot.enabled_field}}}}}{{{{{slot.prompt_field}}}}}{{{{/{slot.enabled_field}}}}}",
        back=f"{{{{#{slot.enabled_field}}}}}{{{{{slot.answer_field}}}}}{{{{/{slot.enabled_field}}}}}",
    )


GRAMMAR_LAYOUT = (
    *PHRASE_LAYOUT,
    *(_template(r, i) for i, r in enumerate((*FORM_RECIPES, "phrase_form_cloze"), len(PHRASE_LAYOUT))),
)
GRAMMAR_FIELDS = (
    *PHRASE_LAYOUT_FIELDS,
    *(
        name
        for t in GRAMMAR_LAYOUT[len(PHRASE_LAYOUT) :]
        for name in (t.slot.enabled_field, t.slot.prompt_field, t.slot.answer_field)
    ),
    "CardKey",
)
COLUMNS = (*GRAMMAR_FIELDS, "Tags")
LAYOUT_SHA256 = digest(
    {
        "id": "grammar-v1",
        "version": 1,
        "templates": [asdict(t) for t in GRAMMAR_LAYOUT],
        "fields": GRAMMAR_FIELDS,
        "css": REFERENCE_CARD_CSS,
    }
)
