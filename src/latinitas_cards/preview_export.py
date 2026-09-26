"""Typed preview and deterministic CSV output for generated principal-part notes.

The module owns source/profile/manifest preparation and file serialization.  CLI
modules render the returned result separately, so terminal output is not part of
the generation or export contract.
"""

from __future__ import annotations

import csv
import html
import io
import os
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .generation import PrincipalPartGenerationResult, generate_principal_part_study_cards
from .manifest import (
    CsvIdentityManifest,
    ManifestReviewItem,
    reconcile_csv_manifest,
)
from .notes import CSV_EXPORT_FIELD_NAMES, GENERATED_NOTE_FIELD_NAMES, GeneratedNote
from .profile import DeckProfile
from .profile_setup import encode_unsafe_controls
from .sources import read_source_records


class PrincipalPartExportError(ValueError):
    """Raised when a generated CSV cannot be prepared or written safely."""


@dataclass(frozen=True, slots=True)
class PrincipalPartExportResult:
    """Structured preview/export data shared by preview and file output."""

    source_path: Path
    profile: DeckProfile
    generation: PrincipalPartGenerationResult
    profile_path: Path | None = None
    manifest_path: Path | None = None
    manifest_reviews: tuple[ManifestReviewItem, ...] = ()
    candidate_manifest: CsvIdentityManifest | None = None

    @property
    def generated_count(self) -> int:
        return self.generation.generated_count

    @property
    def skipped_count(self) -> int:
        return self.generation.skipped_count

    @property
    def ambiguous_count(self) -> int:
        generation_ambiguities = sum(1 for skip in self.generation.skips if skip.status == "ambiguous")
        return generation_ambiguities + len(self.manifest_reviews)


def prepare_principal_part_export(
    source_path: str | Path,
    profile: DeckProfile,
    *,
    profile_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    approved_reuse: Mapping[int, str] | None = None,
    approved_allocations: Mapping[int, str | None] | Iterable[int] | None = None,
    approved_removals: Iterable[str] | None = None,
) -> PrincipalPartExportResult:
    """Read immutable input and build a typed generation result without writing files."""

    source = Path(source_path)
    profile_file = None if profile_path is None else Path(profile_path)
    manifest_file = None if manifest_path is None else Path(manifest_path)

    if profile.source_identity.strategy == "manifest":
        if source.suffix.lower() != ".csv":
            raise PrincipalPartExportError("manifest source identity is supported only for CSV input")
        manifest_file = manifest_file or Path(f"{source}.latinitas.json")
        _reject_input_aliases(source, profile_file, manifest_file)
        records = read_source_records(source)
        saved_manifest = _load_manifest(manifest_file)
        reconciliation = reconcile_csv_manifest(
            records,
            saved_manifest,
            approved_reuse=approved_reuse,
            approved_allocations=approved_allocations,
            approved_removals=approved_removals,
        )
        assignments = reconciliation.identities_by_row
        assigned_rows = tuple(sorted(assignments))
        assigned_records = tuple(records[index] for index in assigned_rows)
        identities = tuple(assignments[index] for index in assigned_rows)
        generation = generate_principal_part_study_cards(
            assigned_records,
            profile,
            manifest_identities=identities,
        )
        return PrincipalPartExportResult(
            source_path=source,
            profile=profile,
            generation=generation,
            profile_path=profile_file,
            manifest_path=manifest_file,
            manifest_reviews=reconciliation.reviews,
            candidate_manifest=reconciliation.manifest,
        )

    _reject_input_aliases(source, profile_file, manifest_file)
    if manifest_file is not None or approved_reuse or approved_allocations or approved_removals:
        raise PrincipalPartExportError(
            "manifest approvals are valid only for a profile using the manifest source-identity strategy"
        )

    source_id_field = profile.source_identity.field if profile.source_identity.strategy == "source_id_field" else None
    records = read_source_records(source, source_id_field=source_id_field)
    generation = generate_principal_part_study_cards(records, profile)
    return PrincipalPartExportResult(
        source_path=source,
        profile=profile,
        generation=generation,
        profile_path=profile_file,
    )


def deterministic_csv_bytes(result: PrincipalPartExportResult) -> bytes:
    """Serialize a complete typed result as deterministic UTF-8 Anki text import."""

    if result.manifest_reviews:
        raise PrincipalPartExportError(
            "Cannot export before explicit identity review resolves all manifest review items."
        )

    notes = result.generation.notes
    note_ids = [note.latinitas_id for note in notes]
    if len(note_ids) != len(set(note_ids)):
        raise PrincipalPartExportError("Cannot export duplicate logical LatinitasID values.")

    for value_name, value in (
        ("generated note type", result.profile.generated_note_type),
        ("target deck", result.profile.target_deck),
    ):
        _validate_import_metadata(value_name, value)

    output = io.StringIO(newline="")
    output.write("#separator:Comma\n")
    output.write("#html:true\n")
    output.write(f"#notetype:{result.profile.generated_note_type}\n")
    output.write(f"#deck:{result.profile.target_deck}\n")
    output.write("#tags column:4\n")
    output.write(f"#columns:{','.join(CSV_EXPORT_FIELD_NAMES)}\n")
    writer = csv.writer(output, delimiter=",", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    for note in notes:
        writer.writerow(_note_values(note))
    return output.getvalue().encode("utf-8")


def write_principal_part_csv(
    result: PrincipalPartExportResult,
    output_path: str | Path,
    *,
    profile_path: str | Path | None = None,
) -> None:
    """Commit generated CSV and its identity manifest as one recoverable pair."""

    destination = Path(output_path)
    profile_override = None if profile_path is None else Path(profile_path)
    if (
        profile_override is not None
        and result.profile_path is not None
        and not _same_path(profile_override, result.profile_path)
    ):
        raise PrincipalPartExportError("The profile path override must match the prepared profile path.")
    protected_paths = (
        result.source_path,
        result.profile_path,
        profile_override,
        result.manifest_path,
    )
    if any(path is not None and _same_path(destination, path) for path in protected_paths):
        raise PrincipalPartExportError("The output path must not overwrite the input, profile, or manifest.")
    _validate_regular_file_destination("output", destination)
    if result.candidate_manifest is not None and result.manifest_path is not None:
        _validate_regular_file_destination("manifest", result.manifest_path)
    if not destination.parent.is_dir():
        raise PrincipalPartExportError("The output directory does not exist; no output was written.")
    if (
        result.candidate_manifest is not None
        and result.manifest_path is not None
        and not result.manifest_path.parent.is_dir()
    ):
        raise PrincipalPartExportError("The manifest directory does not exist; no output was written.")

    payload = deterministic_csv_bytes(result)
    manifest_destination = result.manifest_path if result.candidate_manifest is not None else None
    staged_output: Path | None = None
    staged_manifest: Path | None = None
    output_backup: Path | None = None
    manifest_backup: Path | None = None
    output_committed = False
    manifest_committed = False
    preserve_output_backup = False
    preserve_manifest_backup = False
    output_existed = destination.exists()
    manifest_existed = manifest_destination is not None and manifest_destination.exists()
    try:
        staged_output = _stage_bytes(destination, payload)
        if result.candidate_manifest is not None and manifest_destination is not None:
            staged_manifest = _stage_bytes(
                manifest_destination,
                result.candidate_manifest.to_json().encode("utf-8"),
            )

        output_backup = _move_existing_to_backup(destination)
        if manifest_destination is not None:
            manifest_backup = _move_existing_to_backup(manifest_destination)

        os.replace(staged_output, destination)
        staged_output = None
        output_committed = True
        if staged_manifest is not None and manifest_destination is not None:
            os.replace(staged_manifest, manifest_destination)
            staged_manifest = None
            manifest_committed = True
    except OSError as error:
        output_restored = _restore_after_failed_commit(destination, output_backup, output_existed, output_committed)
        preserve_output_backup = output_backup is not None and not output_restored
        manifest_restored = True
        if manifest_destination is not None:
            manifest_restored = _restore_after_failed_commit(
                manifest_destination,
                manifest_backup,
                bool(manifest_existed),
                manifest_committed,
            )
            preserve_manifest_backup = manifest_backup is not None and not manifest_restored
        retained_backups = tuple(
            str(path)
            for path, preserve in (
                (output_backup, preserve_output_backup),
                (manifest_backup, preserve_manifest_backup),
            )
            if path is not None and preserve
        )
        affected_destinations = tuple(
            str(path)
            for path, restored in (
                (destination, output_restored),
                (manifest_destination, manifest_restored),
            )
            if path is not None and not restored
        )
        recovery_details = []
        if affected_destinations:
            recovery_details.append("Recovery is required")
        if retained_backups:
            recovery_details.append("backups were retained")
        if affected_destinations:
            recovery_details.append("affected destinations to check: " + ", ".join(affected_destinations))
        if retained_backups:
            recovery_details.append("backup locations: " + ", ".join(retained_backups))
        if recovery_details:
            message = (
                "The CSV and identity manifest could not be committed safely. " + "; ".join(recovery_details) + "."
            )
        else:
            message = (
                "No output or manifest was changed because the CSV and identity manifest could not be committed safely."
            )
        raise PrincipalPartExportError(message) from error
    finally:
        _remove_temporary_path(staged_output)
        _remove_temporary_path(staged_manifest)
        if not preserve_output_backup:
            _remove_temporary_path(output_backup)
        if not preserve_manifest_backup:
            _remove_temporary_path(manifest_backup)


def _load_manifest(path: Path) -> CsvIdentityManifest | None:
    if not path.exists():
        return None
    try:
        return CsvIdentityManifest.load(path)
    except (OSError, ValueError) as error:
        raise PrincipalPartExportError(f"Could not read identity manifest '{path.name}'.") from error


def _reject_input_aliases(source: Path, profile: Path | None, manifest: Path | None) -> None:
    paths = (source, profile, manifest)
    existing = [path for path in paths if path is not None]
    for index, first in enumerate(existing):
        for second in existing[index + 1 :]:
            if _same_path(first, second):
                raise PrincipalPartExportError("The input, profile, and manifest paths must be distinct.")


def _same_path(first: Path, second: Path) -> bool:
    try:
        if first.resolve() == second.resolve():
            return True
        return first.exists() and second.exists() and os.path.samefile(first, second)
    except OSError:
        return False


def _validate_import_metadata(name: str, value: str) -> None:
    if encode_unsafe_controls(value, preserve_line_breaks=False) != value:
        raise PrincipalPartExportError(f"Configured {name} contains a control character.")


def _validate_regular_file_destination(label: str, destination: Path) -> None:
    if destination.exists() and not destination.is_file():
        raise PrincipalPartExportError(f"The {label} output path must be a regular file or not exist.")


def _note_values(note: GeneratedNote) -> tuple[str, ...]:
    fields = dict(note.to_anki_fields())
    if tuple(fields) != GENERATED_NOTE_FIELD_NAMES:
        raise PrincipalPartExportError("Generated note fields do not match the stable export field order.")
    return tuple(_export_field_value(name, fields[name]) for name in CSV_EXPORT_FIELD_NAMES)


def _export_field_value(name: str, value: str) -> str:
    encoded = encode_unsafe_controls(value, preserve_line_breaks=name not in _HTML_ESCAPED_METADATA_FIELDS)
    return html.escape(encoded, quote=True) if name in _HTML_ESCAPED_METADATA_FIELDS else encoded


_HTML_ESCAPED_METADATA_FIELDS = frozenset(
    {
        "LatinitasID",
        "Source ID",
        "Source Kind",
        "Source Location",
        "Source Path",
        "Recipe",
        "Exercise Key",
        "Recipe Version",
    }
)


def _stage_bytes(destination: Path, payload: bytes) -> Path:
    temporary_name: str | None = None
    descriptor = -1
    keep_temporary = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        keep_temporary = True
        return Path(temporary_name)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not keep_temporary and temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)


def _move_existing_to_backup(destination: Path) -> Path | None:
    if not destination.exists():
        return None
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.backup.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    os.unlink(temporary_name)
    os.replace(destination, temporary_name)
    return Path(temporary_name)


def _restore_after_failed_commit(
    destination: Path,
    backup: Path | None,
    existed_before: bool,
    committed: bool,
) -> bool:
    try:
        if backup is not None:
            if destination.exists():
                destination.unlink()
            os.replace(backup, destination)
        elif committed and not existed_before:
            destination.unlink()
    except OSError:
        return False
    return True


def _remove_temporary_path(path: Path | None) -> None:
    if path is not None:
        with suppress(FileNotFoundError):
            path.unlink()


__all__ = [
    "PrincipalPartExportError",
    "PrincipalPartExportResult",
    "deterministic_csv_bytes",
    "prepare_principal_part_export",
    "write_principal_part_csv",
]
