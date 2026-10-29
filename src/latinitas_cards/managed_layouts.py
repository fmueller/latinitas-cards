"""Known offline layouts; recognizing fields never authorizes a schema migration."""

from dataclasses import dataclass

from .cards import TEMPLATE_REGISTRY, TEMPLATE_REGISTRY_DIGEST, TEMPLATE_REGISTRY_VERSION, CardSlot
from .corpus_notes import (
    CORPUS_CARD_CSS,
    CORPUS_COLUMNS,
    CORPUS_CONTENT_FIELDS,
    CORPUS_FIELDS,
    CORPUS_SCHEMA_VERSION,
    CORPUS_SLOTS,
    CORPUS_TEMPLATES,
)
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
    # Non-payload fields a managed update may change, in planning, approval and application alike.
    content_fields: frozenset[str] = frozenset(("Lemma", "Principal Parts", "Meaning", "Generator", "Profile"))

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
        ManagedLayout(
            CORPUS_SCHEMA_VERSION,
            CORPUS_FIELDS,
            CORPUS_COLUMNS,
            CORPUS_SLOTS,
            tuple((t.front, t.back) for t in CORPUS_TEMPLATES),
            1,
            "corpus-registry-sha256:" + _digest([slot.semantic_key for slot in CORPUS_SLOTS]),
            CORPUS_CARD_CSS,
            CORPUS_CONTENT_FIELDS,
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
