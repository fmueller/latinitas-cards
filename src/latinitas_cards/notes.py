"""Ownership-aware generated-note contracts.

The authoritative note schema declares every field of the generated learning
object note together with its ownership category: regular managed fields,
user-owned Personal Notes, and special transport metadata (Tags is not a
regular note field).  The Anki note type comprises ``GENERATED_NOTE_FIELD_NAMES``
with ``LatinitasID`` first, while ``CSV_EXPORT_FIELD_NAMES`` and the tags-column
directive are derived from the same schema so generated import data never
offers the user-owned field for accidental overwrite.

The conditional sibling-card slots of :mod:`latinitas_cards.cards` extend this
same schema with their frozen guard/prompt/answer fields; they are regular
managed fields appended after the provenance fields and before Personal Notes,
so existing slots and the Tags column position are never repurposed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .cards import TEMPLATE_REGISTRY, RenderedCard
from .identity import derive_latinitas_id

NOTE_SCHEMA_VERSION = "3"
GENERATOR_VERSION = "3"

FieldOwnership = Literal["managed", "personal", "transport"]


@dataclass(frozen=True, slots=True)
class NoteFieldSpec:
    """One authoritative note field and its ownership category."""

    name: str
    ownership: FieldOwnership
    exported: bool


_BASE_NOTE_FIELDS: tuple[NoteFieldSpec, ...] = (
    NoteFieldSpec("LatinitasID", "managed", True),
    NoteFieldSpec("Lemma", "managed", True),
    NoteFieldSpec("Principal Parts", "managed", True),
    NoteFieldSpec("Meaning", "managed", True),
    NoteFieldSpec("Tags", "transport", True),
    NoteFieldSpec("Source ID", "managed", True),
    NoteFieldSpec("Source Scope", "managed", True),
    NoteFieldSpec("Source Kind", "managed", True),
    NoteFieldSpec("Source Location", "managed", True),
    NoteFieldSpec("Source Path", "managed", True),
    NoteFieldSpec("Note Schema", "managed", True),
    NoteFieldSpec("Generator", "managed", True),
    NoteFieldSpec("Profile", "managed", True),
)

_CARD_NOTE_FIELDS: tuple[NoteFieldSpec, ...] = tuple(
    NoteFieldSpec(name, "managed", True)
    for slot in TEMPLATE_REGISTRY
    for name in (slot.enabled_field, slot.prompt_field, slot.answer_field)
)

AUTHORITATIVE_NOTE_FIELDS: tuple[NoteFieldSpec, ...] = (
    *_BASE_NOTE_FIELDS,
    *_CARD_NOTE_FIELDS,
    NoteFieldSpec("Personal Notes", "personal", False),
)

GENERATED_NOTE_FIELD_NAMES = tuple(field.name for field in AUTHORITATIVE_NOTE_FIELDS)
CSV_EXPORT_FIELD_NAMES = tuple(field.name for field in AUTHORITATIVE_NOTE_FIELDS if field.exported)
TAGS_CSV_COLUMN = CSV_EXPORT_FIELD_NAMES.index("Tags") + 1


@dataclass(frozen=True, slots=True)
class ManagedNoteContent:
    """Structured knowledge fields that a future generation run may replace."""

    lemma: str
    principal_parts: str
    meaning: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.lemma.strip():
            raise ValueError("managed note lemma must be non-empty")
        if not self.principal_parts.strip():
            raise ValueError("managed note principal parts must be non-empty")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("managed note tags must not be empty")


@dataclass(frozen=True, slots=True)
class GeneratedNoteProvenance:
    """Source relationship data that excludes transport-local Anki IDs."""

    source_kind: str
    location: str
    source_path: str | None = None
    source_identity: str | None = None
    source_scope: str | None = None

    def __post_init__(self) -> None:
        if not self.source_kind.strip() or not self.location.strip():
            raise ValueError("generated note provenance requires source kind and location")


@dataclass(frozen=True, slots=True)
class GenerationMetadata:
    """Descriptive generation metadata that never feeds note identity."""

    profile_digest: str = ""
    note_schema: str = NOTE_SCHEMA_VERSION
    generator: str = GENERATOR_VERSION

    def __post_init__(self) -> None:
        if not self.profile_digest.strip():
            raise ValueError("generation metadata requires the effective profile digest")
        if not self.note_schema.strip() or not self.generator.strip():
            raise ValueError("generation metadata versions must be non-empty")


@dataclass(frozen=True, slots=True)
class GeneratedNote:
    """One coherent learning-object note with explicit field ownership."""

    latinitas_id: str
    object_key: str
    content: ManagedNoteContent
    provenance: GeneratedNoteProvenance
    metadata: GenerationMetadata
    cards: tuple[RenderedCard, ...] = ()
    personal_notes: str = ""

    def __post_init__(self) -> None:
        source_identity = self.provenance.source_identity
        if source_identity is None or not source_identity.strip():
            raise ValueError("generated note provenance requires a stable source identity")
        if not self.object_key.strip():
            raise ValueError("generated note object key must be non-empty")
        expected = derive_latinitas_id(
            source_identity,
            self.object_key,
            source_scope=self.provenance.source_scope,
        )
        if self.latinitas_id != expected:
            raise ValueError("LatinitasID does not match the note's immutable logical identity")

    @property
    def card_keys(self) -> tuple[str, ...]:
        """Return the eligible card semantic keys in registry slot order."""

        return tuple(card.slot.semantic_key for card in self.cards if card.eligible)

    @classmethod
    def create(
        cls,
        *,
        source_identity: str,
        provenance: GeneratedNoteProvenance,
        object_key: str,
        metadata: GenerationMetadata,
        content: ManagedNoteContent,
        cards: tuple[RenderedCard, ...] = (),
        personal_notes: str = "",
        source_scope: str | None = None,
    ) -> GeneratedNote:
        """Create a note while deriving its identity from immutable inputs."""

        if provenance.source_identity is not None and provenance.source_identity != source_identity:
            raise ValueError("provenance source identity does not match the generated note source identity")
        if provenance.source_scope != source_scope:
            raise ValueError("provenance source scope does not match the generated note source scope")
        resolved_provenance = replace(provenance, source_identity=source_identity, source_scope=source_scope)
        return cls(
            latinitas_id=derive_latinitas_id(source_identity, object_key, source_scope=source_scope),
            object_key=object_key,
            content=content,
            provenance=resolved_provenance,
            metadata=metadata,
            cards=tuple(cards),
            personal_notes=personal_notes,
        )

    def with_managed_content(self, content: ManagedNoteContent) -> GeneratedNote:
        """Return an updated note while retaining its identity and personal notes."""

        return replace(self, content=content)

    def with_metadata(self, metadata: GenerationMetadata) -> GeneratedNote:
        """Update descriptive generation metadata without changing note identity."""

        return replace(self, metadata=metadata)

    def with_cards(self, cards: tuple[RenderedCard, ...]) -> GeneratedNote:
        """Update the rendered cards without changing note identity."""

        return replace(self, cards=tuple(cards))

    def to_anki_fields(self) -> tuple[tuple[str, str], ...]:
        """Return deterministic Anki text-import fields with identity first."""

        provenance = self.provenance
        metadata = self.metadata
        values = {
            "LatinitasID": self.latinitas_id,
            "Lemma": self.content.lemma,
            "Principal Parts": self.content.principal_parts,
            "Meaning": self.content.meaning,
            "Tags": " ".join(self.content.tags),
            "Source ID": provenance.source_identity or "",
            "Source Scope": provenance.source_scope or "",
            "Source Kind": provenance.source_kind,
            "Source Location": provenance.location,
            "Source Path": provenance.source_path or "",
            "Note Schema": metadata.note_schema,
            "Generator": metadata.generator,
            "Profile": metadata.profile_digest,
            "Personal Notes": self.personal_notes,
        }
        for card in self.cards:
            values[card.slot.enabled_field] = card.guard
            values[card.slot.prompt_field] = card.prompt
            values[card.slot.answer_field] = card.answer
        return tuple((name, values.get(name, "")) for name in GENERATED_NOTE_FIELD_NAMES)


__all__ = [
    "AUTHORITATIVE_NOTE_FIELDS",
    "CSV_EXPORT_FIELD_NAMES",
    "GENERATED_NOTE_FIELD_NAMES",
    "GeneratedNote",
    "GeneratedNoteProvenance",
    "GenerationMetadata",
    "GENERATOR_VERSION",
    "ManagedNoteContent",
    "NOTE_SCHEMA_VERSION",
    "NoteFieldSpec",
    "TAGS_CSV_COLUMN",
]
