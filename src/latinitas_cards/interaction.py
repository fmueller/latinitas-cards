"""Shared terminal capability and legacy prompt failure contract."""

import sys

import typer


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def prompt_required(*flags: str) -> typer.BadParameter:
    return typer.BadParameter(f"prompt_required: Cannot prompt; supply {' / '.join(flags)}.")
