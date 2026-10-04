"""Corpus-independent authored JSONL validation, preview, and export commands."""

from pathlib import Path
from typing import Annotated

import typer

from ..authored_export import write_authored_csv
from ..authored_preview import AuthoredFilters, AuthoredPreviewResult, preview_authored_import
from ..profile_setup import encode_unsafe_controls

app = typer.Typer(help="Validate, preview, and export corpus-independent authored notes.")
InputPath = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]
Namespace = Annotated[
    str, typer.Option("--namespace", help="Explicit collection namespace for stable note identities.")
]


def _load(path: Path, namespace: str, filters: AuthoredFilters | None = None) -> AuthoredPreviewResult:
    try:
        return preview_authored_import(path, namespace, filters)
    except (OSError, ValueError) as error:
        typer.echo(f"Authored input error: {encode_unsafe_controls(str(error), preserve_line_breaks=False)}", err=True)
        raise typer.Exit(1) from error


def _report(result: AuthoredPreviewResult, *, show_cards: bool) -> None:
    typer.echo("Whole-file counts (reconciled nonconflicting notes; invalid rows reported below):")
    for label, counts in (
        ("Kind", result.counts_by_kind),
        ("Section", result.counts_by_section),
        ("Reference", result.counts_by_reference),
        ("Status", result.counts_by_status),
    ):
        typer.echo(f"{label}: " + ", ".join(f"{key!r}: {count}" for key, count in counts))
    typer.echo(f"Merged duplicates: {len(result.merged_duplicates)}")
    for lines in result.merged_duplicates:
        typer.echo("  lines " + ", ".join(map(str, lines)))
    typer.echo(f"Filters: {result.filters}")
    if result.errors:
        for issue in result.errors:
            typer.echo(encode_unsafe_controls(str(issue), preserve_line_breaks=False))
        typer.echo(
            f"Invalid: {len(result.errors)} errors; diagnostic matches: {len(result.diagnostic_notes)}; not exportable"
        )
    else:
        typer.echo(f"Valid. Effective selection: {len(result.selected_notes)}")
        if not result.selected_notes:
            typer.echo("Empty selection.")
    if show_cards:
        typer.echo("Diagnostic cards (not exportable):" if result.errors else "Representative cards:")
        for card in result.cards:
            front = encode_unsafe_controls(card.front, preserve_line_breaks=False)
            back = encode_unsafe_controls(card.back, preserve_line_breaks=False)
            typer.echo(f"{card.kind} {card.key!r} [{card.latinitas_id}]\nFront:\n{front}\nBack:\n{back}")
    if result.errors:
        raise typer.Exit(1)


@app.command()
def validate(input_path: InputPath, namespace: Namespace) -> None:
    """Validate and reconcile the whole file, including skipped notes."""
    _report(_load(input_path, namespace), show_cards=False)


@app.command()
def preview(
    input_path: InputPath,
    namespace: Namespace,
    kind: Annotated[list[str] | None, typer.Option(help="Repeat for OR matching within kinds.")] = None,
    section: Annotated[list[str] | None, typer.Option(help="Exact source section; repeat for OR matching.")] = None,
    reference: Annotated[list[str] | None, typer.Option(help="Exact source reference; repeat for OR matching.")] = None,
    tag: Annotated[list[str] | None, typer.Option(help="Merged tag; repeat for OR matching.")] = None,
) -> None:
    """Preview representative cards; combine filter dimensions with AND."""
    filters = AuthoredFilters(tuple(kind or ()), tuple(section or ()), tuple(reference or ()), tuple(tag or ()))
    _report(_load(input_path, namespace, filters), show_cards=True)


@app.command("export")
def export_notes(
    input_path: InputPath,
    namespace: Namespace,
    output_dir: Annotated[Path, typer.Option(help="Existing directory for per-kind CSV files.")],
    deck: Annotated[str, typer.Option(help="Explicit target Anki deck.")],
    kind: Annotated[list[str] | None, typer.Option(help="Repeat for OR matching within kinds.")] = None,
    section: Annotated[list[str] | None, typer.Option(help="Exact source section; repeat for OR matching.")] = None,
    reference: Annotated[list[str] | None, typer.Option(help="Exact source reference; repeat for OR matching.")] = None,
    tag: Annotated[list[str] | None, typer.Option(help="Merged tag; repeat for OR matching.")] = None,
) -> None:
    """Export selected notes; validate all rows before writing any output."""
    filters = AuthoredFilters(tuple(kind or ()), tuple(section or ()), tuple(reference or ()), tuple(tag or ()))
    result = _load(input_path, namespace, filters)
    _report(result, show_cards=False)
    try:
        paths = write_authored_csv(result, input_path, output_dir, deck)
    except (OSError, ValueError) as error:
        typer.echo(f"Authored export error: {encode_unsafe_controls(str(error), preserve_line_breaks=False)}", err=True)
        raise typer.Exit(1) from error
    for path in paths:
        typer.echo(f"Wrote {path}")
