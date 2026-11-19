"""Thin opt-in grammar consumers; orchestration and provider execution are separate."""

from pathlib import Path
from typing import Annotated, Literal

import typer

from ..acceptance_presentation import ACCEPTANCE_FIELDS, confirm_installation, layout_digest, stylesheet
from ..acceptance_records import canonical
from ..grammar_layout import GRAMMAR_FIELDS, GRAMMAR_LAYOUT, LAYOUT_SHA256
from ..pipeline_consumers import adapt, decide, export, handoff, load_run, load_source, preview
from ..pipeline_orchestration import cancel, execute, prepare, reconcile_call, report
from ..pipeline_profile import PipelineProfile, RecipeOverrides, load_pipeline_profile, resolve_recipes
from ..pipeline_reporting import readable
from ..reference_templates import REFERENCE_CARD_CSS
from .claims import JsonOption, _emit, _error

app = typer.Typer(help="Inspect or orchestrate evidence-gated grammar cards; explicit providers, never apply to Anki.")
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


@app.command(name="handoff")
def handoff_command(run: RunOption, profile: Path | None = None, machine: JsonOption = False) -> None:
    """List unresolved prior exports and manual actions; never observe or apply to Anki."""
    try:
        _emit(handoff(run, profile), machine)
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="export")
def export_command(
    run: RunOption,
    output: Annotated[Path, typer.Option("--output")],
    profile: Path | None = None,
    output_limit: Annotated[int | None, typer.Option(min=1)] = None,
    after_receipt: str | None = None,
    reconcile_receipt: str | None = None,
    machine: JsonOption = False,
) -> None:
    """Export or replay a checked batch; advance only with its explicit receipt ID."""
    try:
        _emit(
            export(
                run,
                output,
                profile_path=profile,
                output_limit=output_limit,
                after_receipt=after_receipt,
                reconcile_receipt=reconcile_receipt,
            ),
            machine,
        )
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="stage")
def stage_command(
    name: Literal["ingest", "analyze", "propose", "counter_check", "decide", "select", "render", "export"],
    run: RunOption,
    profile: Path | None = None,
    allow_hosted: Path | None = None,
    offline: bool = False,
    local_only: bool = False,
    machine: JsonOption = False,
) -> None:
    """Execute one stage; proposal/check stages may call configured providers, never grant acceptance authority."""
    try:
        if (run / "execution.json").exists():
            _emit(
                execute(
                    run,
                    only=name,
                    profile_path=profile,
                    permission_path=allow_hosted,
                    offline=offline,
                    local_only=local_only,
                    disclose=lambda text: typer.echo(text, err=True),
                ),
                machine,
            )
        elif name == "decide":
            _emit(decide(run, profile), machine)
        else:
            raise ValueError("use pipeline run to prepare an orchestrated run")
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="run")
def run_command(
    profile: ProfileOption,
    output: Annotated[Path, typer.Option("--output")],
    objectives: Path | None = None,
    library_resources: Path | None = None,
    allow_hosted: Path | None = None,
    offline: bool = False,
    local_only: bool = False,
    prepare_only: bool = False,
    machine: JsonOption = False,
) -> None:
    """Emit files, report and stage data; prepare-only allocates a run without requests."""
    try:
        root = prepare(profile, output, objectives, library_resources, offline=offline, local_only=local_only)
        _emit(
            execute(
                root,
                only="ingest" if prepare_only else None,
                permission_path=allow_hosted,
                offline=offline,
                local_only=local_only,
                disclose=lambda text: typer.echo(text, err=True),
            ),
            machine,
        )
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="resume")
def resume_command(
    run: RunOption,
    profile: Path | None = None,
    allow_hosted: Path | None = None,
    offline: bool = False,
    local_only: bool = False,
    continue_cancelled: bool = False,
    retry_unknown: bool = False,
    confirm_duplicate_billing: bool = False,
    reconcile_receipt: str | None = None,
    machine: JsonOption = False,
) -> None:
    """Reuse validated successes with persisted network restrictions; relaxing them requires a new run."""
    try:
        _emit(
            execute(
                run,
                profile_path=profile,
                permission_path=allow_hosted,
                offline=offline,
                local_only=local_only,
                continue_cancelled=continue_cancelled,
                retry_unknown=retry_unknown,
                confirm_duplicate_billing=confirm_duplicate_billing,
                reconcile_receipt=reconcile_receipt,
                disclose=lambda text: typer.echo(text, err=True),
            ),
            machine,
        )
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="cancel")
def cancel_command(run: RunOption, machine: JsonOption = False) -> None:
    try:
        _emit(cancel(run), machine)
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="report")
def report_command(run: RunOption, machine: JsonOption = False) -> None:
    """Read the stored execution report without contacting providers."""
    try:
        result = report(run)
        if machine:
            _emit(result, True)
        else:
            typer.echo(readable(result))
    except (OSError, ValueError) as error:
        _error(error, machine)


@app.command(name="reconcile-call")
def reconcile_command(
    run: RunOption,
    call_id: Annotated[str, typer.Option("--call-id")],
    response: Path | None = None,
    abandon: bool = False,
    request_sha256: str | None = None,
    provider_request_id: str | None = None,
    machine: JsonOption = False,
) -> None:
    """Retain an identified retrieved response, or abandon without refunding its reservation."""
    try:
        _emit(
            reconcile_call(
                run,
                call_id,
                response=response,
                abandon=abandon,
                request_sha256=request_sha256,
                provider_request_id=provider_request_id,
            ),
            machine,
        )
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
def migration_plan(
    machine: JsonOption = False, acceptance: bool = False, family: Literal["grammar", "curated"] = "grammar"
) -> None:
    """Copyable append-only layout proposal, never an installation or scheduled-template rewrite."""
    from ..curated_cards import CURATED_FIELDS, CURATED_LAYOUT

    fields = GRAMMAR_FIELDS if family == "grammar" else CURATED_FIELDS
    templates = GRAMMAR_LAYOUT if family == "grammar" else CURATED_LAYOUT
    if family == "curated" and not acceptance:
        raise typer.BadParameter("curated migration requires --acceptance")
    if acceptance:
        fields = (*fields, *(("Personal Notes",) if family == "curated" else ()), *ACCEPTANCE_FIELDS)
    _emit(
        {
            "id": "acceptance-1" if acceptance else "grammar-v1",
            "version": 1,
            "template_sha256": layout_digest(family) if acceptance else LAYOUT_SHA256,
            "requires": (
                "Manual opt-in installation only; CSV headers never install templates. Preserve existing "
                "LatinitasID, CardKey, template ordinals, Personal Notes, user/source tags and scheduling. "
                "Back up and inspect the actual destination before any update. Native preservation is unverified. "
                "After installing this exact family layout, use pipeline confirm-layout --profile FILE --installed "
                "with the matching --family. The profile keeps its three layout keys."
                if acceptance
                else "Create an explicitly named new grammar-v1 note type; existing ordinals stay unchanged. "
                "Manual installation/import only."
            ),
            "fields": fields,
            "css": stylesheet() if acceptance else REFERENCE_CARD_CSS,
            "templates": [
                {"ordinal": t.slot.ordinal, "name": t.slot.template_name, "front": t.front, "back": t.back}
                for t in templates
            ],
        },
        machine,
    )


@app.command(name="confirm-layout")
def confirm_layout(
    profile: ProfileOption,
    installed: bool = False,
    family: Literal["grammar", "curated"] = "grammar",
    machine: JsonOption = False,
) -> None:
    """Record your explicit installation attestation, not native verification or an Anki write."""
    try:
        if not installed:
            raise ValueError("confirm --installed only after manually installing the exact migration-plan")
        path = confirm_installation(profile, family)
        _emit({"confirmation": str(path), "native_verified": False, "applied": False}, machine)
    except (OSError, ValueError) as error:
        _error(error, machine)
