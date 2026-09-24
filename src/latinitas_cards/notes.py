"""Ownership-aware generated-note contracts.

The contract keeps the immutable logical identity, managed fields, provenance,
recipe metadata, and user-owned personal notes in separate typed values.  The
Anki import projection puts ``LatinitasID`` first for repeat imports.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .identity import derive_latinitas_id


@dataclass(frozen=True, slots=True)
class ManagedNoteContent:
    """Fields that a future generation run may replace."""

    prompt: str
    answer: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.prompt.strip() or not self.answer.strip():
            raise ValueError("managed note prompt and answer must be non-empty")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("managed note tags must not be empty")


@dataclass(frozen=True, slots=True)
class GeneratedNoteProvenance:
    """Source relationship data that excludes transport-local Anki IDs."""

    source_kind: str
    location: str
    source_path: str | None = None
    source_identity: str | None = None

    def __post_init__(self) -> None:
        if not self.source_kind.strip() or not self.location.strip():
            raise ValueError("generated note provenance requires source kind and location")


@dataclass(frozen=True, slots=True)
class RecipeMetadata:
    """Recipe identity and descriptive generation metadata."""

    recipe_identity: str
    exercise_key: str
    recipe_version: str = "1"

    def __post_init__(self) -> None:
        if not self.recipe_identity.strip() or not self.exercise_key.strip() or not self.recipe_version.strip():
            raise ValueError("recipe metadata values must be non-empty")


@dataclass(frozen=True, slots=True)
class GeneratedNote:
    """One generated note with explicit field ownership boundaries."""

    latinitas_id: str
    content: ManagedNoteContent
    provenance: GeneratedNoteProvenance
    recipe: RecipeMetadata
    personal_notes: str = ""

    def __post_init__(self) -> None:
        source_identity = self.provenance.source_identity
        if source_identity is None or not source_identity.strip():
            raise ValueError("generated note provenance requires a stable source identity")
        expected = derive_latinitas_id(source_identity, self.recipe.recipe_identity, self.recipe.exercise_key)
        if self.latinitas_id != expected:
            raise ValueError("LatinitasID does not match the note's immutable logical identity")

    @classmethod
    def create(
        cls,
        *,
        source_identity: str,
        provenance: GeneratedNoteProvenance,
        recipe: RecipeMetadata,
        content: ManagedNoteContent,
        personal_notes: str = "",
    ) -> GeneratedNote:
        """Create a note while deriving its identity from immutable inputs."""

        if provenance.source_identity is not None and provenance.source_identity != source_identity:
            raise ValueError("provenance source identity does not match the generated note source identity")
        resolved_provenance = replace(provenance, source_identity=source_identity)
        return cls(
            latinitas_id=derive_latinitas_id(source_identity, recipe.recipe_identity, recipe.exercise_key),
            content=content,
            provenance=resolved_provenance,
            recipe=recipe,
            personal_notes=personal_notes,
        )

    def with_managed_content(self, content: ManagedNoteContent) -> GeneratedNote:
        """Return an updated note while retaining its identity and personal notes."""

        return replace(self, content=content)

    def with_recipe_metadata(self, recipe: RecipeMetadata) -> GeneratedNote:
        """Update descriptive recipe metadata without changing logical exercise identity."""

        return replace(self, recipe=recipe)

    def to_anki_fields(self) -> tuple[tuple[str, str], ...]:
        """Return deterministic Anki text-import fields with identity first."""

        provenance = self.provenance
        recipe = self.recipe
        return (
            ("LatinitasID", self.latinitas_id),
            ("Prompt", self.content.prompt),
            ("Answer", self.content.answer),
            ("Tags", " ".join(self.content.tags)),
            ("Source ID", provenance.source_identity or ""),
            ("Source Kind", provenance.source_kind),
            ("Source Location", provenance.location),
            ("Source Path", provenance.source_path or ""),
            ("Recipe", recipe.recipe_identity),
            ("Exercise Key", recipe.exercise_key),
            ("Recipe Version", recipe.recipe_version),
            ("Personal Notes", self.personal_notes),
        )


__all__ = [
    "GeneratedNote",
    "GeneratedNoteProvenance",
    "ManagedNoteContent",
    "RecipeMetadata",
]
