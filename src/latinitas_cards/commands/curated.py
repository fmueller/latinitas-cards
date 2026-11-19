"""Offline curated validation, candidate inspection and explicit review recording."""

import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from ..curated import load_approvals, load_curated, preview_candidates, record_reviews
from ..profile_setup import encode_unsafe_controls

app = typer.Typer(help="Validate and review curated candidates, without export authorization.")
Input = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]
Permissions = Annotated[Path, typer.Option("--permissions", exists=True, dir_okay=False, readable=True)]
Approvals = Annotated[Path | None, typer.Option("--approvals", exists=True, dir_okay=False, readable=True)]


def _error(error: Exception) -> NoReturn:
    typer.echo(f"Curated input error: {encode_unsafe_controls(str(error), preserve_line_breaks=False)}", err=True)
    raise typer.Exit(1) from error


@app.command()
def validate(input_path: Input, permissions: Permissions) -> None:
    """Validate all items and explicit source/extraction permission attestations."""
    try:
        data, _ = load_curated(input_path, permissions)
    except (OSError, ValueError, RecursionError) as error:
        _error(error)
    typer.echo(
        f"Valid: {len(data.items)} candidates; not export authorization. Effective configuration: offline intake v1."
    )


@app.command()
def preview(input_path: Input, permissions: Permissions, approvals: Approvals = None) -> None:
    """Show candidate wording/provenance and independent review status as JSON."""
    try:
        data, permission = load_curated(input_path, permissions)
        report = preview_candidates(data, permission, load_approvals(approvals))
    except (OSError, ValueError, RecursionError) as error:
        _error(error)
    report["effective_configuration"] = {"mode": "offline", "intake_version": 1}
    # ASCII encoding also prevents source-controlled terminal escape sequences.
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2))


@app.command()
def review(
    input_path: Input,
    permissions: Permissions,
    decisions: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(help="New history file; existing files are never overwritten.")],
    approvals: Approvals = None,
) -> None:
    """Record an explicit decision array against current candidates, retaining history."""
    try:
        data, _ = load_curated(input_path, permissions)
        history = load_approvals(approvals)
        record_reviews(data, history, load_approvals(decisions, history), output)
    except (OSError, ValueError, RecursionError) as error:
        _error(error)
    typer.echo("Recorded explicit decisions; not export authorization. Effective configuration: offline intake v1.")
