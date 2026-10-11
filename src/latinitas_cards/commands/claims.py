"""Non-interactive claim review and offline card inspection."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from ..claim_sidecar import claim_report, record_batch, sidecar_path
from ..profile_setup import safe_source_value
from ..review_inspection import inspect_artifact, inspect_workspace, workspace

app = typer.Typer(help="List, individually attest and inspect claims; not destination/export approval.")
JsonOption = Annotated[bool, typer.Option("--json", help="Machine-readable output and item diagnostics.")]


def _emit(report: dict[str, object], machine: bool) -> None:
    if machine:
        typer.echo(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            typer.echo(f"{key}:")
            typer.echo(safe_source_value(json.dumps(value, indent=2, ensure_ascii=False), key))


def _error(error: Exception, machine: bool) -> None:
    _emit({"diagnostics": [{"code": "invalid", "item": None, "message": str(error)}]}, machine)
    raise typer.Exit(1)


@app.command(name="list")
def list_claims(source: Path, profile: Path | None = None, machine: JsonOption = False) -> None:
    """List claims; --profile selects principal parts, otherwise contextual JSON."""
    try:
        work = workspace(source, profile)
        _emit(
            {
                "claims": [claim_report(b) for b in work.state.claims],
                "diagnostics": [asdict(d) for d in work.state.diagnostics],
            },
            machine,
        )
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command()
def record(
    source: Path,
    profile: Path | None = None,
    batch: Path | None = None,
    fingerprint: str | None = None,
    status: str | None = None,
    reviewer: str | None = None,
    reason: str | None = None,
    value: str | None = None,
    evidence: Path | None = None,
    machine: JsonOption = False,
) -> None:
    """Record one fingerprint or an explicit JSON array; no approve-all path."""
    try:
        if batch is not None:
            if any(v is not None for v in (fingerprint, status, reviewer, reason, value, evidence)):
                raise ValueError("batch and single-decision options cannot be combined")
            items = json.loads(batch.read_text(encoding="utf-8"))
            if not isinstance(items, list) or not items:
                raise ValueError("batch must be a nonempty JSON array of individual decisions")
        else:
            item = {"fingerprint": fingerprint, "status": status, "reviewer": reviewer, "reason": reason}
            if value is not None:
                item["value"] = value
            if evidence is not None:
                item["evidence"] = json.loads(evidence.read_text(encoding="utf-8"))
            items = [item]
        work = workspace(source, profile)
        diagnostics = record_batch(sidecar_path(source), work.state.bases, items)
        _emit(
            {"diagnostics": [asdict(d) for d in diagnostics], "recorded": len(items) if not diagnostics else 0}, machine
        )
        if diagnostics:
            raise typer.Exit(1)
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="inspect")
def inspect_cards(
    source: Path,
    profile: Path | None = None,
    artifact: Path | None = None,
    check: bool = False,
    machine: JsonOption = False,
) -> None:
    """Inspect rendered fields and backing claims; --check fails closed on drift."""
    try:
        if profile is None and artifact is None and source.suffix.lower() in {".csv", ".apkg", ".colpkg"}:
            report = inspect_artifact(source)
        else:
            report = inspect_workspace(workspace(source, profile), artifact)
        _emit(report, machine)
        if check and report["diagnostics"]:
            raise typer.Exit(1)
    except (OSError, ValueError) as error:
        _error(error, machine)
