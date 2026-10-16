"""Non-interactive claim review and offline card inspection."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from ..claim_sidecar import claim_report, record_batch, sidecar_path
from ..cli_contract import ContractError
from ..corpus_claims import corpus_claim_report, record_corpus_batch
from ..phrase_inputs import load_inputs
from ..phrase_review import phrase_workspace
from ..review_adapters import ReviewState
from ..review_inspection import inspect_artifact, inspect_workspace, workspace

app = typer.Typer(help="List, individually attest and inspect claims; not destination/export approval.")
JsonOption = Annotated[bool, typer.Option("--json", help="Machine-readable output and item diagnostics.")]


def _emit(report: dict[str, object], machine: bool) -> None:
    if machine:
        typer.echo(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            typer.echo(f"{key}:")
            # JSON escaping keeps terminal controls inert without hiding later items/fingerprints.
            typer.echo(json.dumps(value, indent=2, ensure_ascii=True))


def _error(error: Exception, machine: bool, *, exit_code: int = 1) -> None:
    code = getattr(error, "code", "invalid")
    _emit({"diagnostics": [{"code": code, "item": None, "message": str(error)}]}, machine)
    raise typer.Exit(exit_code)


def _corpus_state(
    source: Path,
    annotations: Path,
    profile: Path | None,
    phrase_input: bool,
    selections: Path | None,
    recipes: Path | None,
    artifact: Path | None = None,
) -> dict[str, object]:
    _validate_corpus_options(profile, phrase_input, selections, recipes, artifact)
    return corpus_claim_report(source, annotations)


def _validate_corpus_options(
    profile: Path | None,
    phrase_input: bool,
    selections: Path | None,
    recipes: Path | None,
    artifact: Path | None = None,
) -> None:
    if profile is not None or phrase_input or selections is not None or recipes is not None or artifact is not None:
        raise ValueError("--annotations cannot be combined with deck/profile/phrase/artifact options")


def _state(
    source: Path, profile: Path | None, phrase_input: bool, selections: Path | None, recipes: Path | None
) -> ReviewState:
    if selections is not None or recipes is not None:
        if selections is None or recipes is None or profile is not None or phrase_input:
            raise ValueError("phrase exercises require --selections and --recipes, without --profile/--phrase-input")
        return phrase_workspace(source, selections, recipes)[1].state
    if phrase_input:
        if profile is not None:
            raise ValueError("phrase inputs name their confirmed profile; do not combine --phrase-input and --profile")
        return load_inputs(source).state
    return workspace(source, profile).state


@app.command(name="list")
def list_claims(
    source: Path,
    profile: Path | None = None,
    annotations: Path | None = None,
    phrase_input: bool = False,
    selections: Path | None = None,
    recipes: Path | None = None,
    machine: JsonOption = False,
) -> None:
    """List principal-part, contextual, or --phrase-input file claims."""
    try:
        if annotations is not None:
            _emit(_corpus_state(source, annotations, profile, phrase_input, selections, recipes), machine)
            return
        state = _state(source, profile, phrase_input, selections, recipes)
        _emit(
            {
                "claims": [claim_report(b) for b in state.claims],
                "diagnostics": [asdict(d) for d in state.diagnostics],
            },
            machine,
        )
    except (OSError, ValueError, ContractError) as error:
        _error(error, machine)


@app.command()
def record(
    source: Path,
    profile: Path | None = None,
    annotations: Path | None = None,
    batch: Path | None = None,
    fingerprint: str | None = None,
    status: str | None = None,
    reviewer: str | None = None,
    reason: str | None = None,
    value: str | None = None,
    evidence: Path | None = None,
    selected_sense: str | None = None,
    phrase_input: bool = False,
    selections: Path | None = None,
    recipes: Path | None = None,
    machine: JsonOption = False,
) -> None:
    """Record one fingerprint or an explicit JSON array; no approve-all path."""
    try:
        if annotations is not None:
            if value is not None or evidence is not None or selected_sense is not None:
                _error(
                    ValueError("corpus decisions cannot author values, evidence or selected senses"),
                    machine,
                    exit_code=3,
                )
            _validate_corpus_options(profile, phrase_input, selections, recipes)
        if batch is not None:
            if any(v is not None for v in (fingerprint, status, reviewer, reason, value, evidence, selected_sense)):
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
            if selected_sense is not None:
                item["selected_sense"] = selected_sense
            items = [item]
        if annotations is not None:
            diagnostics = record_corpus_batch(source, annotations, items)
        else:
            state = _state(source, profile, phrase_input, selections, recipes)
            diagnostics = record_batch(sidecar_path(recipes if recipes is not None else source), state.bases, items)
        _emit(
            {"diagnostics": [asdict(d) for d in diagnostics], "recorded": len(items) if not diagnostics else 0}, machine
        )
        if diagnostics:
            raise typer.Exit(1)
    except (OSError, ValueError, ContractError) as error:
        _error(error, machine)


@app.command(name="inspect")
def inspect_cards(
    source: Path,
    profile: Path | None = None,
    annotations: Path | None = None,
    artifact: Path | None = None,
    selections: Path | None = None,
    recipes: Path | None = None,
    check: bool = False,
    machine: JsonOption = False,
) -> None:
    """Inspect rendered fields and backing claims; --check fails closed on drift."""
    try:
        if annotations is not None:
            report = _corpus_state(source, annotations, profile, False, selections, recipes, artifact)
        elif selections is not None or recipes is not None:
            if selections is None or recipes is None or profile is not None:
                raise ValueError("phrase inspection requires --selections and --recipes, without --profile")
            report = inspect_workspace(phrase_workspace(source, selections, recipes)[1], artifact)
        elif profile is None and artifact is None and source.suffix.lower() in {".csv", ".apkg", ".colpkg"}:
            report = inspect_artifact(source)
        else:
            report = inspect_workspace(workspace(source, profile), artifact)
        _emit(report, machine)
        if check and report["diagnostics"]:
            raise typer.Exit(1)
    except (OSError, ValueError, ContractError) as error:
        _error(error, machine)
