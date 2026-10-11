"""Known offline layouts; recognizing fields never authorizes a schema migration."""

from dataclasses import dataclass

from .cards import TEMPLATE_REGISTRY, TEMPLATE_REGISTRY_DIGEST, TEMPLATE_REGISTRY_VERSION, CardSlot
from .form_parsing import PARSING_BACK, PARSING_FIELDS, PARSING_FRONT, PARSING_SLOT
from .notes import CSV_EXPORT_FIELD_NAMES, NOTE_SCHEMA_VERSION
from .phrase_recipes import PHRASE_LAYOUT, PHRASE_LAYOUT_FIELDS, PHRASE_SCHEMA_VERSION
from .reference_templates import REFERENCE_CARD_CSS, REFERENCE_CARD_TEMPLATES, REFERENCE_NOTE_TYPE_FIELDS


@dataclass(frozen=True)
class ManagedLayout:
    version: str
    fields: tuple[str, ...]
    columns: tuple[str, ...]
    slots: tuple[CardSlot, ...]
    templates: tuple[tuple[str, str], ...]
    registry_version: int
    registry_digest: str
    css: str = REFERENCE_CARD_CSS

    @property
    def managed_fields(self) -> tuple[str, ...]:
        return tuple(name for name in self.fields if name != "Personal Notes")

    @property
    def payload_fields(self) -> frozenset[str]:
        return frozenset(name for slot in self.slots for name in (slot.prompt_field, slot.answer_field))


def known_layouts() -> tuple[ManagedLayout, ...]:
    """Exact versioned contracts, with the old layout retained unchanged."""
    from .destination_state import _digest

    return (
        ManagedLayout(
            NOTE_SCHEMA_VERSION,
            REFERENCE_NOTE_TYPE_FIELDS,
            CSV_EXPORT_FIELD_NAMES,
            TEMPLATE_REGISTRY,
            tuple((t.front, t.back) for t in REFERENCE_CARD_TEMPLATES),
            TEMPLATE_REGISTRY_VERSION,
            TEMPLATE_REGISTRY_DIGEST,
        ),
        ManagedLayout(
            PHRASE_SCHEMA_VERSION,
            PHRASE_LAYOUT_FIELDS,
            (
                *CSV_EXPORT_FIELD_NAMES,
                *(name for name in PHRASE_LAYOUT_FIELDS if name not in REFERENCE_NOTE_TYPE_FIELDS),
            ),
            tuple(t.slot for t in PHRASE_LAYOUT),
            tuple((t.front, t.back) for t in PHRASE_LAYOUT),
            1,
            "phrase-registry-sha256:" + _digest([t.slot.semantic_key for t in PHRASE_LAYOUT]),
        ),
        ManagedLayout(
            "parsing-v1",
            PARSING_FIELDS,
            (*PARSING_FIELDS[:-1], "Tags"),
            (PARSING_SLOT,),
            ((PARSING_FRONT, PARSING_BACK),),
            1,
            "parsing-registry-sha256:" + _digest(PARSING_SLOT.semantic_key),
        ),
    )


def layout_version(version: str) -> ManagedLayout:
    for layout in known_layouts():
        if layout.version == version:
            return layout
    raise ValueError("unknown managed layout; reviewed manual setup required")


def layout_fields(fields: object) -> ManagedLayout:
    for layout in known_layouts():
        if isinstance(fields, dict) and set(fields) == set(layout.managed_fields):
            return layout
    raise ValueError("incomplete managed fields or user-owned field")
