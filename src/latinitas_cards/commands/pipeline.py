"""Thin opt-in grammar consumers; orchestration and provider execution are separate."""

from pathlib import Path
from typing import Annotated, Literal

import typer

from ..acceptance_records import canonical
from ..grammar_layout import GRAMMAR_FIELDS, GRAMMAR_LAYOUT, LAYOUT_SHA256
from ..pipeline_consumers import adapt, decide, export, load_run, load_source, preview
from ..pipeline_profile import PipelineProfile, RecipeOverrides, load_pipeline_profile, resolve_recipes
from ..reference_templates import REFERENCE_CARD_CSS
from .claims import JsonOption, _emit, _error

app = typer.Typer(
    help="Prepare, inspect and export evidence-gated grammar cards; never call providers or apply to Anki."
)
ProfileOption = Annotated[Path, typer.Option("--profile")]
RunOption = Annotated[Path, typer.Option("--run")]


def _effective(
    path: Path,
    preset: str | None,
    include: list[str] | None,
    exclude: list[str] | None,
    save: Path | None,
    initial: PipelineProfile | None = None,
) -> PipelineProfile:
    profile = initial or load_pipeline_profile(path)
    if preset is not None:
        profile = PipelineProfile.read_json(
            canonical(
                {**profile.model_dump(), "preset": preset, "recipes": resolve_recipes(preset, profile.recipe_overrides)}
            )
        )
    profile = profile.overridden(RecipeOverrides(include=tuple(include or ()), exclude=tuple(exclude or ())))
    if save is not None:
        if save.exists() or save.is_symlink():
            raise ValueError("save-profile must name a new file; original profiles are never modified implicitly")
        base = path.resolve().parent
        saved = {
            **profile.model_dump(),
            "deck_profile": {
                **profile.deck_profile.model_dump(),
                "path": str((base / profile.deck_profile.path).resolve()),
            },
            "inputs": [{**item.model_dump(), "path": str((base / item.path).resolve())} for item in profile.inputs],
        }
        with save.open("xb") as stream:
            stream.write(canonical(saved))
    return profile


@app.command(name="adapt")
def adapt_command(
    profile: ProfileOption,
    objectives: Annotated[Path, typer.Option("--objectives")],
    output: Annotated[Path, typer.Option("--output")],
    kind: Annotated[Literal["phrase-inputs"], typer.Option("--kind")],
    preset: str | None = None,
    include_recipe: list[str] | None = None,
    exclude_recipe: list[str] | None = None,
    save_profile: Path | None = None,
    machine: JsonOption = False,
) -> None:
    """Persist current explicit objectives and effective configuration, not approvals."""
    try:
        effective = _effective(profile, preset, include_recipe, exclude_recipe, save_profile)
        adapt(profile, objectives, output, effective)
        _emit(
            {"saved": True, "run": str(output), "recipes": effective.recipes, "language": effective.language}, machine
        )
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="preview")
def preview_command(
    run: RunOption,
    profile: Path | None = None,
    preset: str | None = None,
    include_recipe: list[str] | None = None,
    exclude_recipe: list[str] | None = None,
    save_profile: Path | None = None,
    machine: JsonOption = False,
) -> None:
    """Read-only linguistic inspection; saving an explicitly requested profile grants no authority."""
    try:
        saved = load_run(run)
        path = profile or Path(saved.profile_path)
        effective = _effective(
            path, preset, include_recipe, exclude_recipe, save_profile, None if profile else saved.profile
        )
        _emit(preview(run, path, effective_profile=effective), machine)
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="export")
def export_command(
    run: RunOption, output: Annotated[Path, typer.Option("--output")], machine: JsonOption = False
) -> None:
    """Write an initial manual-import CSV after fresh source and complete-card checks."""
    try:
        _emit(export(run, output), machine)
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="stage")
def stage_command(name: Literal["decide"], run: RunOption, machine: JsonOption = False) -> None:
    """Evaluate complete stored application analyses; never create semantic authority."""
    try:
        _emit(decide(run), machine)
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="validate")
def validate_command(profile: ProfileOption, machine: JsonOption = False) -> None:
    try:
        effective = load_pipeline_profile(profile)
        load_source(profile, effective)
        _emit({"valid": True, "recipes": effective.recipes, "language": effective.language}, machine)
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="migration-plan")
def migration_plan(machine: JsonOption = False) -> None:
    """Copyable append-only layout proposal, never an installation or scheduled-template rewrite."""
    _emit(
        {
            "id": "grammar-v1",
            "version": 1,
            "template_sha256": LAYOUT_SHA256,
            "requires": "Create an explicitly named new grammar-v1 note type; existing ordinals stay unchanged. "
            "Manual installation/import only.",
            "fields": GRAMMAR_FIELDS,
            "css": REFERENCE_CARD_CSS,
            "templates": [
                {"ordinal": t.slot.ordinal, "name": t.slot.template_name, "front": t.front, "back": t.back}
                for t in GRAMMAR_LAYOUT
            ],
        },
        machine,
    )
