"""Typed preview and deterministic CSV output for generated learning-object notes.

The module owns source/profile/identity-state preparation and file
serialization.  CLI modules render the returned result separately, so terminal
output is not part of the generation or export contract.  CSV sources carry a
persisted, unique source scope: an uninitialized read-only preview reports that
scope confirmation is required, and the first explicitly approved export
allocates the scope and commits it with the source assignments and the output
as one recoverable pair.
"""

from __future__ import annotations

import csv
import html
import io
import os
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

from .checkpoint import (
    GLOBAL_SOURCE_SCOPE,
    CardEligibilityReview,
    CheckpointError,
    PriorExportCheckpoint,
    advance_checkpoint,
    compare_export_with_checkpoint,
    export_fingerprint,
)
from .generation import LearningObjectGenerationResult, generate_learning_object_notes
from .legacy_transition import plan_legacy_transition
from .manifest import (
    CsvIdentityManifest,
    ManifestReviewItem,
    allocate_source_scope,
    reconcile_csv_manifest,
)
from .notes import CSV_EXPORT_FIELD_NAMES, GENERATED_NOTE_FIELD_NAMES, GeneratedNote
from .profile import DeckProfile
from .profile_setup import encode_unsafe_controls
from .sources import CanonicalSourceRecord, read_source_records

CHECKPOINT_SUFFIX = ".latinitas-cards.json"


class PrincipalPartExportError(ValueError):
    """Raised when a generated CSV cannot be prepared or written safely."""


@dataclass(frozen=True, slots=True)
class PrincipalPartExportResult:
    """Structured preview/export data shared by preview and file output."""

    source_path: Path
    profile: DeckProfile
    generation: LearningObjectGenerationResult
    profile_path: Path | None = None
    manifest_path: Path | None = None
    manifest_reviews: tuple[ManifestReviewItem, ...] = ()
    candidate_manifest: CsvIdentityManifest | None = None
    loaded_manifest: CsvIdentityManifest | None = None
    source_scope: str | None = None
    scope_pending: bool = False
    source_entry_count: int = 0
    checkpoint_path: Path | None = None
    loaded_checkpoint: PriorExportCheckpoint | None = None
    checkpoint_scope_binding: str | None = None
    withheld_note_ids: frozenset[str] = frozenset()
    card_eligibility_reviews: tuple[CardEligibilityReview, ...] = ()
    checkpoint_pending: bool = False
    fresh_import_approved: bool = False
    approved_prior_checkpoint: PriorExportCheckpoint | None = None

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

    @property
    def object_count(self) -> int:
        """Learning objects that were confirmed and generated as notes."""

        return len(self.generation.notes)

    @property
    def zero_card_notes(self) -> tuple[GeneratedNote, ...]:
        """Domain-valid objects that currently have no eligible card."""

        return tuple(note for note in self.generation.notes if not note.card_keys)

    @property
    def zero_card_note_count(self) -> int:
        return len(self.zero_card_notes)

    @property
    def exportable_notes(self) -> tuple[GeneratedNote, ...]:
        """Notes that are safe to serialize: eligible cards and no withheld row."""

        return tuple(
            note for note in self.generation.notes if note.card_keys and note.latinitas_id not in self.withheld_note_ids
        )

    @property
    def exported_note_count(self) -> int:
        return len(self.exportable_notes)

    @property
    def card_count(self) -> int:
        """Eligible cards across the notes that would be exported."""

        return sum(len(note.card_keys) for note in self.exportable_notes)


def prepare_principal_part_export(
    source_path: str | Path,
    profile: DeckProfile,
    *,
    profile_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    approve_new_scope: bool = False,
    approved_reuse: Mapping[int, str] | None = None,
    approved_allocations: Mapping[int, str | None] | Iterable[int] | None = None,
    approved_removals: Iterable[str] | None = None,
    checkpoint_path: str | Path | None = None,
    approve_fresh_import: bool = False,
    legacy_note_types: Iterable[str] = (),
) -> PrincipalPartExportResult:
    """Read immutable input and build a typed generation result without writing files."""

    plan_legacy_transition(profile, (), legacy_note_types=legacy_note_types)
    source = Path(source_path)
    profile_file = None if profile_path is None else Path(profile_path)
    manifest_file = None if manifest_path is None else Path(manifest_path)

    strategy = profile.source_identity.strategy
    if strategy == "manifest":
        if source.suffix.lower() != ".csv":
            raise PrincipalPartExportError("manifest source identity is supported only for CSV input")
        manifest_file = manifest_file or Path(f"{source}.latinitas.json")
        checkpoint_file = _resolve_checkpoint_path(source, checkpoint_path)
        _reject_input_aliases(source, profile_file, manifest_file, checkpoint_file)
        return _prepare_scoped_csv(
            source,
            profile,
            profile_file=profile_file,
            state_file=manifest_file,
            checkpoint_file=checkpoint_file,
            reconcile_rows=True,
            approve_new_scope=approve_new_scope,
            approved_reuse=approved_reuse,
            approved_allocations=approved_allocations,
            approved_removals=approved_removals,
            approve_fresh_import=approve_fresh_import,
        )

    package_checkpoint_file = Path(f"{source}{CHECKPOINT_SUFFIX}")
    _reject_input_aliases(source, profile_file, manifest_file, package_checkpoint_file)
    uses_csv_state = source.suffix.lower() == ".csv" and strategy == "source_id_field"
    if not uses_csv_state and (
        manifest_file is not None
        or checkpoint_path is not None
        or approve_new_scope
        or approved_reuse
        or approved_allocations
        or approved_removals
    ):
        raise PrincipalPartExportError(
            "identity state and approvals are valid only for a CSV source using the manifest "
            "or explicit source-ID identity strategy"
        )
    if uses_csv_state:
        if approved_reuse or approved_allocations or approved_removals:
            raise PrincipalPartExportError(
                "Row approvals are manifest identity review options; they are not valid for a CSV source "
                "using the explicit source-ID identity strategy."
            )
        manifest_file = manifest_file or Path(f"{source}.latinitas.json")
        checkpoint_file = _resolve_checkpoint_path(source, checkpoint_path)
        _reject_input_aliases(source, profile_file, manifest_file, checkpoint_file)
        return _prepare_scoped_csv(
            source,
            profile,
            profile_file=profile_file,
            state_file=manifest_file,
            checkpoint_file=checkpoint_file,
            reconcile_rows=False,
            approve_new_scope=approve_new_scope,
            approve_fresh_import=approve_fresh_import,
        )

    source_id_field = profile.source_identity.field if strategy == "source_id_field" else None
    records = read_source_records(source, source_id_field=source_id_field)
    generation = generate_learning_object_notes(records, profile)
    package_result = PrincipalPartExportResult(
        source_path=source,
        profile=profile,
        generation=generation,
        profile_path=profile_file,
        source_entry_count=len(records),
    )
    return _apply_prior_export_evidence(
        package_result,
        checkpoint_file=package_checkpoint_file,
        approve_fresh_import=approve_fresh_import,
        missing_checkpoint_reason=(
            "Package sources keep no local record that proves a first export, so a missing prior-export "
            "checkpoint cannot be read as an empty prior card set."
        ),
        scope_binding=GLOBAL_SOURCE_SCOPE,
    )


def _resolve_checkpoint_path(source: Path, override: str | Path | None) -> Path:
    return Path(override) if override is not None else Path(f"{source}{CHECKPOINT_SUFFIX}")


def _prepare_scoped_csv(
    source: Path,
    profile: DeckProfile,
    *,
    profile_file: Path | None,
    state_file: Path,
    checkpoint_file: Path,
    reconcile_rows: bool,
    approve_new_scope: bool,
    approve_fresh_import: bool = False,
    approved_reuse: Mapping[int, str] | None = None,
    approved_allocations: Mapping[int, str | None] | Iterable[int] | None = None,
    approved_removals: Iterable[str] | None = None,
) -> PrincipalPartExportResult:
    """Prepare one CSV source under a persisted, unique identity scope."""

    identity_field = profile.source_identity.field if profile.source_identity.strategy == "source_id_field" else None
    records = read_source_records(source, source_id_field=identity_field)
    saved_state = _load_identity_state(state_file)
    scope_previously_committed = saved_state is not None and saved_state.is_scoped

    if saved_state is None or not saved_state.is_scoped:
        if not approve_new_scope:
            if approved_reuse or approved_allocations or approved_removals:
                raise PrincipalPartExportError(
                    "Row approvals require a committed source scope; approve the scope first with --approve-scope."
                )
            return _scope_pending_result(
                source,
                profile,
                profile_file=profile_file,
                state_file=state_file,
                legacy_state=saved_state,
                source_entry_count=len(records),
            )
        scope = allocate_source_scope()
        if saved_state is None:
            base_state = CsvIdentityManifest.scoped(_record_columns(records), scope)
        else:
            base_state = saved_state.with_scope(scope)
    else:
        scope = saved_state.source_scope
        base_state = saved_state

    if reconcile_rows:
        reconciliation = reconcile_csv_manifest(
            records,
            base_state,
            source_scope=scope,
            approved_reuse=approved_reuse,
            approved_allocations=approved_allocations,
            approved_removals=approved_removals,
        )
        assignments = reconciliation.identities_by_row
        assigned_rows = tuple(sorted(assignments))
        assigned_records = tuple(records[index] for index in assigned_rows)
        identities = tuple(assignments[index] for index in assigned_rows)
        generation = generate_learning_object_notes(
            assigned_records,
            profile,
            manifest_identities=identities,
            source_scope=scope,
        )
        manifest_reviews = reconciliation.reviews
        candidate_manifest = reconciliation.manifest
    else:
        generation = generate_learning_object_notes(records, profile, source_scope=scope)
        manifest_reviews = ()
        candidate_manifest = base_state
    prepared = PrincipalPartExportResult(
        source_path=source,
        profile=profile,
        generation=generation,
        profile_path=profile_file,
        manifest_path=state_file,
        manifest_reviews=manifest_reviews,
        candidate_manifest=candidate_manifest,
        loaded_manifest=saved_state,
        source_scope=scope,
        source_entry_count=len(records),
    )
    return _apply_prior_export_evidence(
        prepared,
        checkpoint_file=checkpoint_file,
        approve_fresh_import=approve_fresh_import,
        missing_checkpoint_reason=(
            "A committed source scope exists without a prior-export checkpoint; earlier exports cannot be ruled out."
            if scope_previously_committed
            else None
        ),
        scope_binding=scope,
    )


def _apply_prior_export_evidence(
    result: PrincipalPartExportResult,
    *,
    checkpoint_file: Path,
    approve_fresh_import: bool,
    missing_checkpoint_reason: str | None,
    scope_binding: str,
) -> PrincipalPartExportResult:
    """Bind retained prior-export evidence to a prepared result.

    Missing, corrupt, or incompatible checkpoint state is a review gate, never
    an assumed empty prior card set: without an explicit fresh-import
    confirmation the outcome stays read-only.  ``missing_checkpoint_reason``
    states why an absent checkpoint cannot prove a first export: a scoped CSV
    source with a committed scope has exported before, and package sources keep
    no local record that could prove one at all.  Compatible evidence withholds
    whole note rows that lost previously exported card keys.
    """

    prior: PriorExportCheckpoint | None = None
    pending_reason: str | None = None
    if checkpoint_file.exists():
        try:
            prior = PriorExportCheckpoint.load(checkpoint_file)
        except (OSError, CheckpointError):
            pending_reason = "The prior-export checkpoint cannot be read and its evidence must not be assumed empty."
        else:
            incompatibility = prior.incompatibility(source_scope=scope_binding)
            if incompatibility is not None:
                pending_reason = f"The prior-export checkpoint is incompatible: {incompatibility}."
    elif missing_checkpoint_reason is not None:
        pending_reason = missing_checkpoint_reason
    if pending_reason is not None:
        if not approve_fresh_import:
            review = CardEligibilityReview(
                kind="checkpoint_confirmation",
                latinitas_id=None,
                source_identity=None,
                card_keys=(),
                message=(
                    pending_reason
                    + " Recover or review the checkpoint, or explicitly approve a fresh import to proceed."
                ),
            )
            return replace(
                result,
                checkpoint_path=checkpoint_file,
                checkpoint_scope_binding=scope_binding,
                checkpoint_pending=True,
                card_eligibility_reviews=(review,),
            )
        return replace(
            result,
            checkpoint_path=checkpoint_file,
            checkpoint_scope_binding=scope_binding,
            checkpoint_pending=True,
            loaded_checkpoint=None,
            fresh_import_approved=True,
            approved_prior_checkpoint=prior,
        )
    withheld_ids, reviews = compare_export_with_checkpoint(prior, result.generation.notes)
    return replace(
        result,
        checkpoint_path=checkpoint_file,
        checkpoint_scope_binding=scope_binding,
        loaded_checkpoint=prior,
        withheld_note_ids=withheld_ids,
        card_eligibility_reviews=reviews,
        fresh_import_approved=approve_fresh_import,
    )


def _scope_pending_result(
    source: Path,
    profile: DeckProfile,
    *,
    profile_file: Path | None,
    state_file: Path,
    legacy_state: CsvIdentityManifest | None,
    source_entry_count: int,
) -> PrincipalPartExportResult:
    if legacy_state is None:
        message = (
            "No source scope is committed for this CSV source yet. A read-only preview does not "
            "mint stable identities; the first explicitly approved export allocates and commits "
            "a unique source scope together with the output."
        )
    else:
        message = (
            "The existing identity state is a legacy unscoped manifest. Its IDs must not be "
            "silently reinterpreted: recover or review the state, or explicitly approve a fresh "
            "start that allocates a new unique source scope."
        )
    return PrincipalPartExportResult(
        source_path=source,
        profile=profile,
        generation=LearningObjectGenerationResult(notes=(), skips=()),
        profile_path=profile_file,
        manifest_path=state_file,
        manifest_reviews=(
            ManifestReviewItem(
                kind="scope_confirmation",
                row_index=None,
                source_identity=None,
                candidate_identities=(),
                message=message,
            ),
        ),
        candidate_manifest=None,
        source_scope=None,
        scope_pending=True,
        source_entry_count=source_entry_count,
    )


def _load_identity_state(path: Path) -> CsvIdentityManifest | None:
    if not path.exists():
        return None
    try:
        return CsvIdentityManifest.load(path)
    except (OSError, ValueError) as error:
        raise PrincipalPartExportError(f"Could not read identity state '{path.name}'.") from error


def _record_columns(records: tuple[CanonicalSourceRecord, ...]) -> tuple[str, ...]:
    if not records:
        raise PrincipalPartExportError("Cannot bootstrap a source scope for an empty CSV source.")
    return tuple(records[0].fields)


def deterministic_csv_bytes(result: PrincipalPartExportResult) -> bytes:
    """Serialize a complete typed result as deterministic UTF-8 Anki text import."""

    if result.scope_pending:
        raise PrincipalPartExportError("Cannot export before an explicitly approved source scope is committed.")
    if result.manifest_reviews:
        raise PrincipalPartExportError(
            "Cannot export before explicit identity review resolves all manifest review items."
        )
    if result.checkpoint_pending and not result.fresh_import_approved:
        raise PrincipalPartExportError(
            "Cannot export without usable prior-export checkpoint evidence: recover or review the "
            "checkpoint, or explicitly approve a fresh import."
        )

    notes = result.exportable_notes
    note_ids = [note.latinitas_id for note in notes]
    if len(note_ids) != len(set(note_ids)):
        raise PrincipalPartExportError("Cannot export duplicate logical LatinitasID values.")

    return serialize_anki_csv(
        result.profile.generated_note_type,
        result.profile.target_deck,
        CSV_EXPORT_FIELD_NAMES,
        (_note_values(note) for note in notes),
    )


def serialize_anki_csv(note_type: str, deck: str, columns: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> bytes:
    """Shared UTF-8 transport boundary; callers supply already rendered managed fields."""
    for value_name, value in (("generated note type", note_type), ("target deck", deck)):
        _validate_import_metadata(value_name, value)
    output = io.StringIO(newline="")
    output.write("#separator:Comma\n")
    output.write("#html:true\n")
    output.write(f"#notetype:{note_type}\n")
    output.write(f"#deck:{deck}\n")
    output.write(f"#tags column:{columns.index('Tags') + 1}\n")
    output.write(f"#columns:{','.join(columns)}\n")
    writer = csv.writer(output, delimiter=",", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        if len(row) != len(columns):
            raise PrincipalPartExportError("CSV row width does not match the export schema.")
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


@dataclass
class _CommitArtifact:
    """One destination committed with the output through the recovery boundary."""

    destination: Path
    payload: bytes
    staged: Path | None = None
    backup: Path | None = None
    existed_before: bool = False


def _candidate_checkpoint(result: PrincipalPartExportResult, payload: bytes) -> PriorExportCheckpoint | None:
    """Build the checkpoint to commit with this output, if state is persisted."""

    if result.checkpoint_path is None or result.checkpoint_scope_binding is None or result.scope_pending:
        return None
    prior = None if result.checkpoint_pending else result.loaded_checkpoint
    return advance_checkpoint(
        prior,
        source_scope=result.checkpoint_scope_binding,
        exported_notes=result.exportable_notes,
        export_fingerprint=export_fingerprint(payload),
    )


def write_principal_part_csv(
    result: PrincipalPartExportResult,
    output_path: str | Path,
    *,
    profile_path: str | Path | None = None,
) -> None:
    """Commit generated CSV, identity state, and checkpoint as one recoverable unit."""

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
        result.checkpoint_path,
    )
    if any(path is not None and _same_path(destination, path) for path in protected_paths):
        raise PrincipalPartExportError("The output path must not overwrite the input, profile, or state.")
    _validate_regular_file_destination("output", destination)
    if result.candidate_manifest is not None and result.manifest_path is not None:
        _validate_regular_file_destination("manifest", result.manifest_path)
    candidate_checkpoint_path = result.checkpoint_path if not result.scope_pending else None
    if candidate_checkpoint_path is not None:
        _validate_regular_file_destination("checkpoint", candidate_checkpoint_path)
    if not destination.parent.is_dir():
        raise PrincipalPartExportError("The output directory does not exist; no output was written.")
    if (
        result.candidate_manifest is not None
        and result.manifest_path is not None
        and not result.manifest_path.parent.is_dir()
    ):
        raise PrincipalPartExportError("The manifest directory does not exist; no output was written.")
    if candidate_checkpoint_path is not None and not candidate_checkpoint_path.parent.is_dir():
        raise PrincipalPartExportError("The checkpoint directory does not exist; no output was written.")

    payload = deterministic_csv_bytes(result)
    candidate_checkpoint = _candidate_checkpoint(result, payload)
    manifest_destination = result.manifest_path if result.candidate_manifest is not None else None
    _reject_changed_identity_state(result, manifest_destination)
    checkpoint_destination = candidate_checkpoint_path if candidate_checkpoint is not None else None
    _reject_changed_checkpoint_state(result, checkpoint_destination)

    artifacts: list[_CommitArtifact] = [_CommitArtifact(destination=destination, payload=payload)]
    if result.candidate_manifest is not None and manifest_destination is not None:
        artifacts.append(
            _CommitArtifact(
                destination=manifest_destination,
                payload=result.candidate_manifest.to_json().encode("utf-8"),
            )
        )
    if candidate_checkpoint is not None and checkpoint_destination is not None:
        artifacts.append(
            _CommitArtifact(destination=checkpoint_destination, payload=candidate_checkpoint.to_json().encode("utf-8"))
        )
    commit_csv_artifacts(tuple((artifact.destination, artifact.payload) for artifact in artifacts))


def commit_csv_artifacts(payloads: tuple[tuple[Path, bytes], ...]) -> None:
    """Stage all payloads before replacement and restore originals on a failed commit."""
    artifacts = [_CommitArtifact(destination=path, payload=payload) for path, payload in payloads]
    preserve_backups = False
    try:
        for artifact in artifacts:
            artifact.existed_before = artifact.destination.exists()
            artifact.staged = _stage_bytes(artifact.destination, artifact.payload)
        for artifact in artifacts:
            artifact.backup = _prepare_backup_slot(artifact.destination)
            if artifact.backup is not None:
                os.replace(artifact.destination, artifact.backup)
        for artifact in artifacts:
            if artifact.staged is None:
                continue
            os.replace(artifact.staged, artifact.destination)
            artifact.staged = None
    except BaseException as error:
        preserve_backups = True
        restored: dict[Path, bool] = {}
        for artifact in artifacts:
            restored[artifact.destination] = _restore_after_failed_commit(
                artifact.destination, artifact.backup, artifact.existed_before
            )
        retained_backups = tuple(
            str(artifact.backup) for artifact in artifacts if artifact.backup is not None and artifact.backup.exists()
        )
        affected_destinations = tuple(
            str(artifact.destination) for artifact in artifacts if not restored[artifact.destination]
        )
        message = _recovery_failure_message(affected_destinations, retained_backups)
        restoration_incomplete = not all(restored.values())
        if isinstance(error, OSError):
            raise PrincipalPartExportError(message) from error
        if isinstance(error, KeyboardInterrupt) and restoration_incomplete:
            raise KeyboardInterrupt(message) from error
        if restoration_incomplete:
            error.add_note(message)
        raise
    finally:
        for artifact in artifacts:
            _remove_temporary_path(artifact.staged)
            if not preserve_backups:
                _remove_temporary_path(artifact.backup)


def _reject_changed_identity_state(
    result: PrincipalPartExportResult,
    manifest_destination: Path | None,
) -> None:
    """Abort before any destructive move when the identity state changed since prepare.

    The check blocks a concurrent committer from silently orphaning an already
    exported scope: a valid-but-different state, or the loss of a state that
    prepare loaded, must be re-prepared and reviewed rather than overwritten.
    Unparseable prior bytes keep the ordinary backup/restore semantics.
    """

    if manifest_destination is None or result.candidate_manifest is None:
        return
    if manifest_destination.exists():
        if result.loaded_manifest is None:
            try:
                CsvIdentityManifest.load(manifest_destination)
            except (OSError, ValueError):
                return
            raise PrincipalPartExportError(
                "The identity state changed since preparation: another export committed this source. "
                "Re-run the preview and export again before writing."
            )
        try:
            current = CsvIdentityManifest.load(manifest_destination)
        except (OSError, ValueError):
            return
        if current != result.loaded_manifest:
            raise PrincipalPartExportError(
                "The identity state changed since preparation: another export committed this source. "
                "Re-run the preview and export again before writing."
            )
    elif result.loaded_manifest is not None:
        raise PrincipalPartExportError(
            "The identity state changed since preparation: the persisted source scope disappeared. "
            "Recover the identity state or explicitly approve a fresh start before writing."
        )


def _reject_changed_checkpoint_state(
    result: PrincipalPartExportResult,
    checkpoint_destination: Path | None,
) -> None:
    """Abort when retained card evidence changed since prepare.

    A parseable checkpoint other than the one prepare loaded, or one that
    appeared for a run that prepared without one, must be re-prepared so its
    evidence is compared rather than silently dropped — including a checkpoint
    bound to another scope, whose retained evidence an overwrite would destroy.
    This also covers an explicitly approved fresh import: its approval covered
    only the state observed at prepare (no file, unreadable bytes, or exactly
    the incompatible checkpoint recorded as the approved prior), not valid or
    different evidence committed afterwards.  Unreadable prior bytes keep the
    ordinary backup/restore semantics.
    """

    if checkpoint_destination is None:
        return
    if checkpoint_destination.exists():
        if result.loaded_checkpoint is None:
            try:
                appeared = PriorExportCheckpoint.load(checkpoint_destination)
            except (OSError, CheckpointError):
                return
            if appeared != result.approved_prior_checkpoint:
                raise PrincipalPartExportError(
                    "The prior-export checkpoint changed since preparation: another export committed this "
                    "source. Re-run the preview and export again before writing."
                )
            return
        try:
            current = PriorExportCheckpoint.load(checkpoint_destination)
        except (OSError, CheckpointError):
            return
        if current != result.loaded_checkpoint:
            raise PrincipalPartExportError(
                "The prior-export checkpoint changed since preparation: another export committed this "
                "source. Re-run the preview and export again before writing."
            )
    elif result.loaded_checkpoint is not None:
        raise PrincipalPartExportError(
            "The prior-export checkpoint disappeared since preparation. Recover the retained export "
            "evidence or explicitly approve a fresh import before writing."
        )


def _reject_input_aliases(
    source: Path,
    profile: Path | None,
    manifest: Path | None,
    checkpoint: Path | None = None,
) -> None:
    paths = (source, profile, manifest, checkpoint)
    existing = [path for path in paths if path is not None]
    for index, first in enumerate(existing):
        for second in existing[index + 1 :]:
            if _same_path(first, second):
                raise PrincipalPartExportError("The input, profile, and state paths must be distinct.")


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
        raise PrincipalPartExportError("Generated note fields do not match the authoritative note schema order.")
    return tuple(_export_field_value(name, fields[name]) for name in CSV_EXPORT_FIELD_NAMES)


def _export_field_value(name: str, value: str) -> str:
    encoded = encode_unsafe_controls(value, preserve_line_breaks=name not in _HTML_ESCAPED_METADATA_FIELDS)
    return html.escape(encoded, quote=True) if name in _HTML_ESCAPED_METADATA_FIELDS else encoded


_HTML_ESCAPED_METADATA_FIELDS = frozenset(
    {
        "LatinitasID",
        "Source ID",
        "Source Scope",
        "Source Kind",
        "Source Location",
        "Source Path",
        "Note Schema",
        "Generator",
        "Profile",
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


def _prepare_backup_slot(destination: Path) -> Path | None:
    if not destination.exists():
        return None
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.backup.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    os.unlink(temporary_name)
    return Path(temporary_name)


def _recovery_failure_message(affected_destinations: tuple[str, ...], retained_backups: tuple[str, ...]) -> str:
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
        return (
            "The output, identity state, and card-evidence checkpoint could not be committed safely. "
            + ("; ".join(recovery_details))
            + "."
        )
    return (
        "No output or committed state was changed because the output, identity state, and card-evidence "
        "checkpoint could not be committed safely."
    )


def _restore_after_failed_commit(destination: Path, backup: Path | None, existed_before: bool) -> bool:
    try:
        if backup is not None:
            if not backup.exists():
                return True
            if destination.exists():
                destination.unlink()
            os.replace(backup, destination)
        elif not existed_before and destination.exists():
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
