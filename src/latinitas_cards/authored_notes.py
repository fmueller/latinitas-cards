"""Stable, ownership-aware note types for already reconciled authored items."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from types import MappingProxyType

from .authored_identity import IdentifiedAuthoredItem
from .authored_import import FormItem, QaItem, VocabItem
from .notes import GenerationMetadata, NoteFieldSpec

AUTHORED_NOTE_SCHEMA_VERSION = "authored-1"

_PROVENANCE_FIELDS = (
    "Document",
    "Section",
    "Reference",
    "Source Key",
    "Source Kind",
    "Language",
    "Note Schema",
    "Generator",
    "Profile",
)
_BACK_SUFFIX = (
    '{{#Reference}}<div class="reference">{{Reference}}</div>{{/Reference}}\n'
    '{{#Personal Notes}}<div class="personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}'
)


@dataclass(frozen=True)
class AuthoredNoteType:
    """One fixed note type with exactly one card template."""

    name: str
    content_fields: tuple[tuple[str, str], ...]
    front_template: str
    back_template: str

    @property
    def fields(self) -> tuple[NoteFieldSpec, ...]:
        managed_names = (
            "LatinitasID",
            *(field for field, _ in self.content_fields),
            *_PROVENANCE_FIELDS,
        )
        managed_fields = tuple(NoteFieldSpec(name, "managed", True) for name in managed_names)
        return (*managed_fields, NoteFieldSpec("Personal Notes", "personal", False))

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)


AUTHORED_NOTE_TYPES: Mapping[str, AuthoredNoteType] = MappingProxyType(
    {
        "vocab": AuthoredNoteType(
            "Latinitas Authored Vocabulary",
            (("Lemma", "lemma"), ("Dictionary Form", "dictionary_form"), ("Meaning", "meaning")),
            '<div class="lemma">{{Lemma}}</div>\n'
            '{{#Dictionary Form}}<div class="dictionary-form">{{Dictionary Form}}</div>{{/Dictionary Form}}',
            '{{FrontSide}}\n<hr id="answer">\n<div class="meaning">{{Meaning}}</div>\n' + _BACK_SUFFIX,
        ),
        "form": AuthoredNoteType(
            "Latinitas Authored Form",
            (
                ("Text Form", "text_form"),
                ("Base Form", "base_form"),
                ("Analysis", "analysis"),
                ("Translation", "translation"),
                ("Context", "context"),
            ),
            '<div class="text-form">{{Text Form}}</div>\n'
            '{{#Context}}<div class="context">{{Context}}</div>{{/Context}}',
            '{{FrontSide}}\n<hr id="answer">\n<div class="base-form">{{Base Form}}</div>\n'
            '<div class="analysis">{{Analysis}}</div>\n<div class="translation">{{Translation}}</div>\n' + _BACK_SUFFIX,
        ),
        "qa": AuthoredNoteType(
            "Latinitas Authored QA",
            (("Question", "question"), ("Answer", "answer")),
            '<div class="question">{{Question}}</div>',
            '{{FrontSide}}\n<hr id="answer">\n<div class="answer">{{Answer}}</div>\n' + _BACK_SUFFIX,
        ),
    }
)


@dataclass(frozen=True)
class RenderedAuthoredNote:
    """Managed HTML-safe fields; personal notes are never offered for import."""

    latinitas_id: str
    note_type: AuthoredNoteType
    managed_fields: tuple[tuple[str, str], ...]
    tags: tuple[str, ...]

    def to_export_fields(self) -> tuple[tuple[str, str], ...]:
        """Tags travel separately as Anki metadata, not as a regular field."""
        return self.managed_fields


def render_authored_note(note: IdentifiedAuthoredItem, metadata: GenerationMetadata) -> RenderedAuthoredNote:
    """Render an item obtained through reconciliation's require_valid().

    Selection, CSV transport, and import policy are owned by later pipeline stages.
    Authored schema versioning is independent of the v0.1.0 sibling-card schema;
    the shared metadata supplies only the generator version and profile digest.
    """
    item = note.item
    schema = AUTHORED_NOTE_TYPES[item.kind]
    # The discriminated union is validated at the import boundary.
    assert isinstance(item, VocabItem | FormItem | QaItem)
    values = {
        "LatinitasID": note.latinitas_id,
        **{field: getattr(item, attribute) or "" for field, attribute in schema.content_fields},
        "Document": item.provenance.document,
        "Section": item.provenance.section,
        "Reference": item.provenance.reference or "",
        "Source Key": item.key,
        "Source Kind": item.kind,
        "Language": item.language_tag,
        "Note Schema": AUTHORED_NOTE_SCHEMA_VERSION,
        "Generator": metadata.generator,
        "Profile": metadata.profile_digest,
    }
    fields = tuple(
        (field.name, escape(values[field.name]).replace("\n", "<br>")) for field in schema.fields if field.exported
    )
    return RenderedAuthoredNote(note.latinitas_id, schema, fields, item.tags)


def reference_authored_note_types() -> str:
    """Generate the copyable reference from the same authoritative contract."""
    sections = []
    for kind, schema in AUTHORED_NOTE_TYPES.items():
        sections.append(
            f"## `{kind}` — {schema.name}\n\n"
            "Create these regular fields in this exact order:\n\n```text\n"
            + "\n".join(schema.field_names)
            + "\n```\n\nCreate exactly one card template, named `Recognition`.\n\n"
            + "Front:\n\n```html\n"
            + schema.front_template
            + "\n```\n\n"
            + "Back:\n\n```html\n"
            + schema.back_template
            + "\n```"
        )
    return "\n\n".join(sections) + "\n"
