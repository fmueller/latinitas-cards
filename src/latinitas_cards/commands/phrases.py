"""Explicit reviewed phrase selections, previews and initial manual CSV export."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from ..claim_sidecar import sidecar_path
from ..destination_state import DestinationSnapshot, load_state, validate_managed_controls
from ..managed_layouts import layout_version
from ..managed_plans import compose_plan
from ..phrase_generation import generate_proposals, load_saved_selections
from ..phrase_inputs import SelectionRequest, bind_selections, inspect_inputs, load_inputs, save_selections
from ..phrase_recipes import load_recipe_plan, preview_plan
from ..phrase_review import phrase_workspace
from ..preview_export import commit_csv_artifacts, serialize_anki_csv
from ..profile import load_profile
from ..reference_templates import REFERENCE_STYLE_VERSION
from .claims import JsonOption, _emit, _error

app = typer.Typer(help="Select and review bounded phrase exercises; export approved selections for manual import.")


@app.command(name="plan")
def managed_plan(
    source: Path, selections: Path, recipes: Path, snapshot: Path, state: Path, machine: JsonOption = False
) -> None:
    """Plan current reviewed payloads and explicit blocked withdrawal effects; never apply."""
    try:
        _, work = phrase_workspace(source, selections, recipes)
        baseline = load_state(state)
        observed = DestinationSnapshot(snapshot.read_text(encoding="utf-8"))
        if baseline["binding"]["schema"]["version"] != "phrase-v1":
            raise ValueError("phrase-v1 manual setup and explicit adoption required; never rebind scheduled templates")
        proposals: list[dict[str, Any]] = [
            {
                "identity": fields["LatinitasID"],
                "fields": dict(fields),
                "status": "approved" if any(fields[s.enabled_field] == "1" for s in work.slots) else "withheld",
                "contributions": {"configured_tags": list(work.profile.tags)},
            }
            for fields in work.fields
        ]
        selected = {p["identity"] for p in proposals}
        proposals.extend(
            {
                "identity": identity,
                "fields": dict(anchor["fields"]),
                "retire": True,
                "status": "missing",
                "contributions": {},
            }
            for identity, anchor in baseline["anchors"].items()
            if identity not in selected and anchor["fields"].get("Source Kind") == "phrase"
        )
        _emit(compose_plan(observed, baseline, proposals, work.profile.model_dump(mode="json")), machine)
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="export")
def export_csv(source: Path, selections: Path, recipes: Path, output: Path, machine: JsonOption = False) -> None:
    """Initial creation only; omission never withdraws a previously imported exercise."""
    try:
        work = load_inputs(source)
        protected = (*work.protected, selections, recipes, sidecar_path(recipes))
        if output.exists() or any(output.resolve() == path.resolve() for path in protected):
            raise ValueError(
                "choose a new CSV path outside source/review files; existing exercises need managed updates"
            )
        _, reviewed = phrase_workspace(source, selections, recipes)
        layout = layout_version("phrase-v1")
        for fields in reviewed.fields:
            validate_managed_controls(fields)
        rows = tuple(
            tuple(" ".join(reviewed.profile.tags) if name == "Tags" else fields[name] for name in layout.columns)
            for fields in reviewed.fields
            if any(fields[slot.enabled_field] == "1" for slot in layout.slots)
        )
        payload = serialize_anki_csv(
            reviewed.profile.generated_note_type, reviewed.profile.target_deck, layout.columns, rows
        )
        commit_csv_artifacts(((output, payload),))
        _emit(
            {
                "exported": len(rows),
                "output": str(output),
                "diagnostics": [asdict(d) for d in reviewed.state.diagnostics],
                "requires": f"reviewed phrase-v1/CSS v{REFERENCE_STYLE_VERSION} manual setup, "
                "manual import and explicit baseline adoption; "
                "not managed creation or withdrawal",
            },
            machine,
        )
    except (OSError, ValueError) as error:
        _error(error, machine)


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
def recipes(source: Path, selections: Path, request: Path, output: Path, machine: JsonOption = False) -> None:
    """Persist only explicitly selected recipes, targets, references and limits."""
    try:
        work = load_inputs(source)
        plan = load_recipe_plan(request)
        report = preview_plan(
            work, load_saved_selections(selections), plan, load_profile(source.parent / work.data.profile)
        )
        if report["skipped"]:
            _emit({"saved": False, "diagnostics": report["skipped"]}, machine)
            raise typer.Exit(1)
        save_selections(output, plan.model_dump(mode="json"), (*work.protected, selections, request))
        _emit({"saved": True, "plan": plan.model_dump(mode="json")}, machine)
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command()
def preview(source: Path, selections: Path, recipes: Path, machine: JsonOption = False) -> None:
    """Show complete proposed prompts/answers, alternatives and withholding reasons."""
    try:
        _emit(phrase_workspace(source, selections, recipes)[0], machine)
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


@app.command()
def generate(source: Path, selections: Path, machine: JsonOption = False) -> None:
    """Preview proposals from saved explicit choices, rechecking current evidence."""
    try:
        saved = load_saved_selections(selections)
        work = load_inputs(source)
        _emit(generate_proposals(work, saved), machine)
    except (OSError, ValueError) as error:
        _error(error, machine)
