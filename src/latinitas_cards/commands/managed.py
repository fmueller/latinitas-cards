"""JSON review/approval only; managed CSV application belongs to a later task."""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ..destination_state import BoundDestination, _encoded, _object, read_snapshot
from ..managed_plans import approve_plan, compose_plan
from ..profile_setup import encode_unsafe_controls

app = typer.Typer(help="Inspect offline managed plans and explicitly approve selected operations; no application.")
InputPath = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]


def _load(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")))


def _error(exc: Exception) -> None:
    typer.echo(f"Managed error: {encode_unsafe_controls(str(exc), preserve_line_breaks=False)}", err=True)
    raise typer.Exit(1) from exc


@app.command("plan")
def review_plan(request: InputPath) -> None:
    """Read binding, snapshot, baseline, proposals and effective_profile from JSON."""
    try:
        data = _load(request)
        binding = _object(data["binding"])
        snapshot = read_snapshot(data["snapshot"], BoundDestination(**binding))
        result = compose_plan(snapshot, data.get("baseline"), data["proposals"], data["effective_profile"])
        typer.echo(_encoded(result))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _error(exc)


@app.command("approve")
def approve(
    plan: InputPath,
    review: Annotated[str, typer.Option(help="Explicit apply-review record, not generation/profile approval.")],
    operation: Annotated[list[str] | None, typer.Option(help="Exact operation ID; repeat to select a subset.")] = None,
) -> None:
    """Emit an operation-bound approval receipt, not a CSV or application."""
    try:
        typer.echo(_encoded(approve_plan(_load(plan), operation or [], review)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _error(exc)
