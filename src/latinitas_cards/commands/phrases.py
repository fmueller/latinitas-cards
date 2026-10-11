"""Inspect and persist reviewed inputs; recipes and phrase export are later work."""

from dataclasses import asdict
from pathlib import Path

import typer

from ..phrase_inputs import SelectionRequest, bind_selections, inspect_inputs, load_inputs, save_selections
from .claims import JsonOption, _emit, _error

app = typer.Typer(help="Inspect/select bounded reviewed lexical inputs; does not generate or export cards.")


@app.command(name="inspect")
def inspect(source: Path, check: bool = False, machine: JsonOption = False) -> None:
    try:
        report = inspect_inputs(load_inputs(source))
        _emit(report, machine)
        if check and report["diagnostics"]:
            raise typer.Exit(1)
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command()
def select(source: Path, request: Path, output: Path, machine: JsonOption = False) -> None:
    """Validate every explicit role/sense/form binding before atomic persistence."""
    try:
        work = load_inputs(source)
        choices = SelectionRequest.model_validate_json(request.read_text(encoding="utf-8"))
        payload, diagnostics = bind_selections(work, choices)
        if diagnostics:
            _emit({"diagnostics": [asdict(d) for d in diagnostics], "saved": False}, machine)
            raise typer.Exit(1)
        save_selections(output, payload, (*work.protected, request))
        _emit({"diagnostics": [], "saved": True, "selection": payload}, machine)
    except (OSError, ValueError) as error:
        _error(error, machine)
