"""Explicit reviewed contextual parsing preview and fresh/manual CSV export."""

from pathlib import Path
from typing import Annotated

import typer

from ..form_parsing import ParsingInput, generate_parsing, parsing_csv

app = typer.Typer(help="Preview reviewed contextual parsing; separate manual note type, no managed apply.")


def _load(source: Path) -> ParsingInput:
    try:
        return ParsingInput.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


@app.command()
def preview(source: Path) -> None:
    """Emit claim fingerprints, accepted claims, alternatives and eligibility skips."""
    import json

    # ASCII JSON escapes terminal bidi/control characters without losing data.
    typer.echo(json.dumps(generate_parsing(_load(source)), indent=2))


@app.command(name="export")
def export_csv(
    source: Path,
    output: Path,
    approve_fresh_import: Annotated[
        bool,
        typer.Option(
            "--approve-fresh-import",
            help=(
                "Confirm a fresh manual import with new scheduling and separate reference setup; "
                "no scheduling migration or managed apply"
            ),
        ),
    ] = False,
) -> None:
    """Write eligible cards only; requires separate manual reference setup and new scheduling."""
    data = _load(source)
    if not approve_fresh_import:
        raise typer.BadParameter(
            "Requires --approve-fresh-import: separate manual setup, no scheduling migration/apply."
        )
    if output.exists():
        raise typer.BadParameter("Output already exists; choose a new export path.")
    try:
        payload = parsing_csv(data)
        with output.open("xb") as stream:
            stream.write(payload)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo("Fresh contextual CSV only; managed application and template provisioning unsupported.")
