"""Offline managed review, CSV handoff and observed-result reconciliation."""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ..destination_state import (
    BoundDestination,
    _object,
    load_state,
    read_snapshot,
    review_reconciliation,
    save_state,
)
from ..managed_application import emit_updates, observe_updates
from ..managed_plans import approve_plan, compose_plan
from ..profile_setup import encode_unsafe_controls

app = typer.Typer(help="Review managed plans, emit approved CSV, and reconcile offline GUI import results.")
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
        typer.echo(json.dumps(result))
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
        typer.echo(json.dumps(approve_plan(_load(plan), operation or [], review)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _error(exc)


@app.command("emit")
def emit(
    request: InputPath,
    state: Annotated[Path, typer.Option(help="Atomic applied-baseline journal.")],
    output: Annotated[Path, typer.Option(help="New CSV artifact; never proof of import.")],
) -> None:
    """Revalidate JSON handoff with fresh snapshot, selected approval and backup."""
    try:
        data = _load(request)
        snapshot = read_snapshot(data["snapshot"], BoundDestination(**_object(data["binding"])))
        result = emit_updates(
            data["plan"],
            data["approval"],
            snapshot,
            state,
            Path(data["backup"]),
            data["recovery"],
            output,
            data["note_type"],
            data["deck"],
        )
        typer.echo(json.dumps(result))
        if result["failed"] or result["emission"]["status"] == "unknown":
            raise typer.Exit(1)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _error(exc)


@app.command("observe")
def observe_result(
    request: InputPath,
    state: Annotated[Path, typer.Option(help="Atomic applied-baseline journal.")],
    interval_confirmed: Annotated[
        bool, typer.Option(help="Attest fresh snapshot/no intervening edits through import.")
    ] = False,
) -> None:
    """Validate observed snapshot (or null/no import) before baseline advancement."""
    try:
        data = _load(request)
        snapshot = (
            read_snapshot(data["snapshot"], BoundDestination(**_object(data["binding"])))
            if data["snapshot"] is not None
            else None
        )
        typer.echo(
            json.dumps(
                observe_updates(state, data["plan_id"], snapshot, data["report"], interval_confirmed=interval_confirmed)
            )
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _error(exc)


@app.command("reconcile")
def reconcile_result(
    request: InputPath, state: Annotated[Path, typer.Option(help="Applied-baseline journal.")]
) -> None:
    """Explicitly review current ownership after partial import or backup restoration.

    Abandons pending effects; new writes require a new plan and selected approval.
    """
    try:
        data = _load(request)
        snapshot = read_snapshot(data["snapshot"], BoundDestination(**_object(data["binding"])))
        result = review_reconciliation(load_state(state), snapshot, data["ownership"], data["review"])
        save_state(state, result)
        typer.echo(
            json.dumps({"status": "reconciled", "version": result["version"], "requires": "replan and renew approval"})
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _error(exc)
