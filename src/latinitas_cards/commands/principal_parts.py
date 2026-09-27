"""Terminal boundary for profile-driven principal-part preview and export."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import typer

from ..generation import GenerationSkip
from ..preview_export import (
    PrincipalPartExportResult,
    prepare_principal_part_export,
    write_principal_part_csv,
)
from ..profile import DeckProfile, load_profile, resolve_profile
from ..profile_setup import safe_source_value

_MAX_DIAGNOSTIC_ROWS = 50


def run_principal_part_preview(
    *,
    source_path: Path,
    profile_path: Path,
    manifest_path: Path | None,
    target_deck: str | None,
    generated_note_type: str | None,
    tags: Sequence[str] | None,
    recipes: Sequence[str] | None,
    approved_reuse: Sequence[str],
    approved_allocations: Iterable[int],
    approved_removals: Iterable[str],
    legacy_note_types: Sequence[str] = (),
    limit: int,
) -> None:
    """Prepare and render a profile-driven preview without writing output."""

    if limit < 0:
        raise ValueError("preview limit must not be negative")
    result = _prepare(
        source_path=source_path,
        profile_path=profile_path,
        manifest_path=manifest_path,
        target_deck=target_deck,
        generated_note_type=generated_note_type,
        tags=tags,
        recipes=recipes,
        approved_reuse=approved_reuse,
        approved_allocations=approved_allocations,
        approved_removals=approved_removals,
        legacy_note_types=legacy_note_types,
    )
    render_principal_part_preview(result, limit=limit)


def run_principal_part_export(
    *,
    source_path: Path,
    profile_path: Path,
    output_path: Path,
    manifest_path: Path | None,
    target_deck: str | None,
    generated_note_type: str | None,
    tags: Sequence[str] | None,
    recipes: Sequence[str] | None,
    approved_reuse: Sequence[str],
    approved_allocations: Iterable[int],
    approved_removals: Iterable[str],
    approve_new_scope: bool,
    approve_fresh_import: bool,
    legacy_note_types: Sequence[str] = (),
    limit: int,
) -> None:
    """Render a preview and then write the deterministic CSV output."""

    if limit < 0:
        raise ValueError("preview limit must not be negative")
    result = _prepare(
        source_path=source_path,
        profile_path=profile_path,
        manifest_path=manifest_path,
        target_deck=target_deck,
        generated_note_type=generated_note_type,
        tags=tags,
        recipes=recipes,
        approved_reuse=approved_reuse,
        approved_allocations=approved_allocations,
        approved_removals=approved_removals,
        approve_new_scope=approve_new_scope,
        approve_fresh_import=approve_fresh_import,
        legacy_note_types=legacy_note_types,
    )
    render_principal_part_preview(result, limit=limit)
    write_principal_part_csv(result, output_path, profile_path=profile_path)
    typer.echo(f"Output: {safe_source_value(str(output_path), 'output path')}")


def render_principal_part_preview(result: PrincipalPartExportResult, *, limit: int) -> None:
    """Render bounded, terminal-safe representative notes and structured reasons."""

    typer.echo("Principal-part preview")
    typer.echo(f"Source entries: {result.source_entry_count}")
    typer.echo(f"Objects: {result.object_count}")
    typer.echo(f"Notes: {result.exported_note_count}")
    typer.echo(f"Cards: {result.card_count}")
    typer.echo(f"Zero-eligible notes: {result.zero_card_note_count}")
    typer.echo(f"Skipped: {result.skipped_count}")
    typer.echo(f"Ambiguous: {result.ambiguous_count}")
    if result.scope_pending:
        typer.echo(
            "Source scope confirmation required: "
            + safe_source_value(
                "a read-only preview does not mint stable identities; run generate with --approve-scope "
                "to allocate and commit a unique source scope with the first export.",
                "source scope",
            )
        )

    field_context = _preview_field_context(result)
    for index, note in enumerate(result.generation.notes[:limit], start=1):
        typer.echo(f"Representative note {index}:")
        typer.echo(f"  LatinitasID: {safe_source_value(note.latinitas_id, 'LatinitasID')}")
        typer.echo(f"  Lemma: {safe_source_value(note.content.lemma, field_context, limit=512)}")
        typer.echo(f"  Principal Parts: {safe_source_value(note.content.principal_parts, field_context, limit=512)}")
        typer.echo(f"  Meaning: {safe_source_value(note.content.meaning, field_context, limit=512)}")
        typer.echo(f"  Tags: {safe_source_value(' '.join(note.content.tags), field_context, limit=512)}")
        typer.echo(f"  Card Keys: {safe_source_value(' '.join(note.card_keys), 'card keys', limit=512)}")
        typer.echo(
            "  Provenance: "
            + safe_source_value(
                f"{note.provenance.source_kind} {note.provenance.location} {note.provenance.source_identity or ''}",
                field_context,
            )
        )

    if result.generation.skips:
        typer.echo("Structured skips:")
        for skip in result.generation.skips[:_MAX_DIAGNOSTIC_ROWS]:
            _render_skip(skip, field_context)
        _render_omitted_count("structured skips", len(result.generation.skips))
    if result.manifest_reviews:
        typer.echo("Manifest reviews:")
        for review in result.manifest_reviews[:_MAX_DIAGNOSTIC_ROWS]:
            row = "none" if review.row_index is None else str(review.row_index)
            identity = "none" if review.source_identity is None else review.source_identity
            candidates = ", ".join(review.candidate_identities) or "none"
            typer.echo(
                "  "
                + safe_source_value(
                    f"{review.kind}: row={row}; source={identity}; candidates={candidates}; {review.message}",
                    field_context,
                )
            )
        _render_omitted_count("manifest reviews", len(result.manifest_reviews))
    if result.card_eligibility_reviews:
        typer.echo("Card eligibility reviews:")
        for card_review in result.card_eligibility_reviews[:_MAX_DIAGNOSTIC_ROWS]:
            note_id = "none" if card_review.latinitas_id is None else card_review.latinitas_id
            identity = "none" if card_review.source_identity is None else card_review.source_identity
            keys = ", ".join(card_review.card_keys) or "none"
            typer.echo(
                "  "
                + safe_source_value(
                    f"{card_review.kind}: note={note_id}; source={identity}; affected keys={keys}; "
                    f"{card_review.message}",
                    field_context,
                )
            )
        _render_omitted_count("card eligibility reviews", len(result.card_eligibility_reviews))


def _prepare(
    *,
    source_path: Path,
    profile_path: Path,
    manifest_path: Path | None,
    target_deck: str | None,
    generated_note_type: str | None,
    tags: Sequence[str] | None,
    recipes: Sequence[str] | None,
    approved_reuse: Sequence[str],
    approved_allocations: Iterable[int],
    approved_removals: Iterable[str],
    approve_new_scope: bool = False,
    approve_fresh_import: bool = False,
    legacy_note_types: Sequence[str] = (),
) -> PrincipalPartExportResult:
    profile = _effective_profile(
        profile_path,
        target_deck=target_deck,
        generated_note_type=generated_note_type,
        tags=tags,
        recipes=recipes,
    )
    return prepare_principal_part_export(
        source_path,
        profile,
        profile_path=profile_path,
        manifest_path=manifest_path,
        approve_new_scope=approve_new_scope,
        approved_reuse=_parse_reuse_approvals(approved_reuse),
        approved_allocations=tuple(approved_allocations),
        approved_removals=tuple(approved_removals),
        approve_fresh_import=approve_fresh_import,
        legacy_note_types=legacy_note_types,
    )


def _effective_profile(
    profile_path: Path,
    *,
    target_deck: str | None,
    generated_note_type: str | None,
    tags: Sequence[str] | None,
    recipes: Sequence[str] | None,
) -> DeckProfile:
    profile = load_profile(profile_path)
    overrides: dict[str, object] = {}
    if target_deck is not None:
        overrides["target_deck"] = target_deck
    if generated_note_type is not None:
        overrides["generated_note_type"] = generated_note_type
    if tags is not None:
        overrides["tags"] = tuple(tags)
    if recipes is not None:
        overrides["selected_recipes"] = tuple(recipes)
    return resolve_profile(profile, overrides or None)


def _parse_reuse_approvals(values: Sequence[str]) -> Mapping[int, str]:
    approvals: dict[int, str] = {}
    for raw_value in values:
        row_text, separator, source_identity = raw_value.partition("=")
        if not separator or not source_identity:
            raise ValueError("--approve-reuse values must use ROW=SOURCE_ID")
        try:
            row_index = int(row_text)
        except ValueError as error:
            raise ValueError("--approve-reuse row indexes must be integers") from error
        if row_index in approvals:
            raise ValueError(f"--approve-reuse specifies row {row_index} more than once")
        approvals[row_index] = source_identity
    return approvals


def _preview_field_context(result: PrincipalPartExportResult) -> str:
    fields = [
        result.profile.fields.lexical_entry_field,
        result.profile.fields.principal_parts_field,
    ]
    if result.profile.fields.meaning_field is not None:
        fields.append(result.profile.fields.meaning_field)
    if result.profile.source_identity.field is not None:
        fields.append(result.profile.source_identity.field)
    return ", ".join(fields)


def _render_omitted_count(label: str, count: int) -> None:
    omitted = count - _MAX_DIAGNOSTIC_ROWS
    if omitted > 0:
        typer.echo(f"  ... {omitted} additional {label} omitted.")


def _render_skip(skip: GenerationSkip, field_context: str) -> None:
    source_identity = skip.source_identity or "none"
    source_location = skip.source_location or "none"
    typer.echo(
        "  "
        + safe_source_value(
            f"{skip.status}: {skip.code}; source={source_identity}; location={source_location}; {skip.message}",
            field_context,
        )
    )


__all__ = [
    "render_principal_part_preview",
    "run_principal_part_export",
    "run_principal_part_preview",
]
