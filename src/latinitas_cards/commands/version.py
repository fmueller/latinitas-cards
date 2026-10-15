"""Eager version option, isolated from legacy command implementations."""

from importlib.metadata import version
from typing import Annotated

import typer


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"latinitas-cards {version('latinitas-cards')}")
        raise typer.Exit()


def root(
    show_version: Annotated[bool, typer.Option("--version", callback=_show_version, is_eager=True)] = False,
) -> None:
    pass
