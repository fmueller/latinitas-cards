"""Authored selection crosses the shared Anki CSV and recoverable-write boundary."""

from pathlib import Path

from .authored_notes import AUTHORED_NOTE_TYPES, render_authored_note
from .authored_preview import AuthoredPreviewResult
from .notes import GenerationMetadata
from .preview_export import commit_csv_artifacts, serialize_anki_csv


def write_authored_csv(result: AuthoredPreviewResult, source: Path, output_dir: Path, deck: str) -> tuple[Path, ...]:
    """Validate the whole selection before touching any destination; omit empty kinds."""
    selected = result.require_valid_selection()
    if not selected:
        return ()
    if not deck.strip():
        raise ValueError("Target deck must be nonempty.")
    if not output_dir.is_dir():
        raise ValueError("The output directory must already exist; no output was written.")
    metadata = GenerationMetadata(profile_digest="authored")
    payloads = []
    for kind, schema in AUTHORED_NOTE_TYPES.items():
        notes = tuple(render_authored_note(note, metadata) for note in selected if note.item.kind == kind)
        if not notes:
            continue
        destination = output_dir / f"{kind}.csv"
        if destination.resolve() == source.resolve() or (destination.exists() and destination.samefile(source)):
            raise ValueError("The output must not overwrite the authored input.")
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError("CSV destinations must be regular files, not symlinks or directories.")
        columns = tuple(field.name for field in schema.fields if field.exported) + ("Tags",)
        payload = serialize_anki_csv(
            schema.name,
            deck,
            columns,
            (tuple(value for _, value in note.to_export_fields()) + (" ".join(note.tags),) for note in notes),
        )
        payloads.append((destination, payload))
    commit_csv_artifacts(tuple(payloads))
    return tuple(path for path, _ in payloads)
