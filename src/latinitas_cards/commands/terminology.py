"""Explicitly persist Latin-default terminology on a legacy profile."""

import json
from pathlib import Path

import typer

from ..profile import DeckProfile
from ..terminology import terminology_report

app = typer.Typer(help="Upgrade legacy profile terminology without changing linguistic review data.")


@app.command()
def upgrade(source: Path, output: Path) -> None:
    """Write a validated, versioned profile to a new path; never overwrite the source."""
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("profile JSON must contain an object")
        profile = DeckProfile.upgrade_terminology(data)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(profile.to_json())
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(terminology_report(profile.morphology.terminology, profile.language_tag))
    typer.echo("Profile upgraded; linguistic reviews unchanged. Changed answers need renewed managed payload approval.")
