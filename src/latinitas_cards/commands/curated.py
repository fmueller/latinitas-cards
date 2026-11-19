"""Offline curated validation, candidate inspection and explicit review recording."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer

from ..acceptance_records import canonical
from ..curated import Approval, CuratedInput, load_approvals, load_curated, preview_candidates, record_reviews
from ..curated_cards import RecipePlan, Settings, preview_report, render_cards, write_preview
from ..curated_export import Loader, export_curated
from ..pipeline_profile import PipelineProfile, RecipeOverrides, load_pipeline_profile, resolve_recipes
from ..profile_setup import encode_unsafe_controls

app = typer.Typer(help="Inspect and explicitly review curated cards; offline CSV handoff, never Anki writes.")
Input = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]
Permissions = Annotated[Path, typer.Option("--permissions", exists=True, dir_okay=False, readable=True)]
Approvals = Annotated[Path | None, typer.Option("--approvals", exists=True, dir_okay=False, readable=True)]
Plan = Annotated[Path | None, typer.Option("--plan", exists=True, dir_okay=False, readable=True)]
Language = Annotated[Literal["de", "en"] | None, typer.Option("--language")]
Theme = Annotated[Literal["light", "dark"], typer.Option("--theme")]


def _settings(
    profile: Path | None,
    language: Literal["de", "en"] | None,
    preset: str | None,
    include: list[str] | None,
    exclude: list[str] | None,
    save: Path | None,
    theme: Literal["light", "dark"],
    data_language: Literal["de", "en"] | None,
    input_path: Path,
) -> Settings:
    overrides = RecipeOverrides(include=tuple(include or ()), exclude=tuple(exclude or ()))
    if profile is None:
        if save is not None:
            raise ValueError("save-profile requires an explicit pipeline profile")
        selected_language = language or data_language
        if selected_language is None:
            raise ValueError("empty curated input requires --language or --profile")
        recipes = (
            resolve_recipes(preset, overrides)
            if preset
            else tuple(r for r in ("rule_application", *overrides.include) if r not in overrides.exclude)
        )
        return Settings(selected_language, tuple(dict.fromkeys(recipes)), theme)
    effective = load_pipeline_profile(profile)
    references = [
        r
        for r in effective.inputs
        if r.kind == "curated" and (profile.resolve().parent / r.path).resolve() == input_path.resolve()
    ]
    if len(references) != 1 or references[0].sha256 != hashlib.sha256(input_path.read_bytes()).hexdigest():
        raise ValueError("missing or changed curated profile input; confirm the exact input reference")
    if preset is not None or language is not None:
        effective = PipelineProfile.read_json(
            canonical(
                {
                    **effective.model_dump(),
                    "preset": preset if preset is not None else effective.preset,
                    "language": language or effective.language,
                    "recipes": resolve_recipes(
                        preset if preset is not None else effective.preset, effective.recipe_overrides
                    ),
                }
            )
        )
    if effective.preset is None and "rule_application" not in effective.recipe_overrides.exclude:
        effective = effective.overridden(RecipeOverrides(include=("rule_application",), exclude=()))
    effective = effective.overridden(overrides)
    if save is not None:
        base = profile.resolve().parent
        saved = {
            **effective.model_dump(),
            "deck_profile": {
                **effective.deck_profile.model_dump(),
                "path": str((base / effective.deck_profile.path).resolve()),
            },
            "inputs": [{**item.model_dump(), "path": str((base / item.path).resolve())} for item in effective.inputs],
        }
        with save.open("xb") as stream:
            stream.write(canonical(saved))
    return Settings(effective.language, effective.recipes, theme, effective)


def _loader(input_path: Path, permissions: Path, plan: Path, approvals: Path | None) -> Loader:
    def load() -> tuple[CuratedInput, RecipePlan, tuple[Approval, ...]]:
        data, _ = load_curated(input_path, permissions)
        return data, RecipePlan.read_json(plan.read_bytes()), load_approvals(approvals)

    return load


def _error(error: Exception) -> NoReturn:
    typer.echo(f"Curated input error: {encode_unsafe_controls(str(error), preserve_line_breaks=False)}", err=True)
    raise typer.Exit(1) from error


@app.command()
def validate(input_path: Input, permissions: Permissions) -> None:
    """Validate all items and explicit source/extraction permission attestations."""
    try:
        data, _ = load_curated(input_path, permissions)
    except (OSError, ValueError, RecursionError) as error:
        _error(error)
    typer.echo(
        f"Valid: {len(data.items)} candidates; not export authorization. Effective configuration: offline intake v1."
    )


@app.command()
def preview(
    input_path: Input,
    permissions: Permissions,
    approvals: Approvals = None,
    plan: Plan = None,
    language: Language = None,
    profile: Path | None = None,
    preset: str | None = None,
    include_recipe: list[str] | None = None,
    exclude_recipe: list[str] | None = None,
    save_profile: Path | None = None,
    theme: Theme = "light",
    html: Path | None = None,
    production_run: Path | None = None,
) -> None:
    """Inspect intake, or exact learning cards with --plan; never approve by previewing."""
    try:
        data, permission = load_curated(input_path, permissions)
        if plan is None:
            if any((language, profile, preset, include_recipe, exclude_recipe, save_profile, html, production_run)):
                raise ValueError("card configuration requires --plan")
            report = preview_candidates(data, permission, load_approvals(approvals))
            report["effective_configuration"] = {"mode": "offline", "intake_version": 1}
        else:
            settings = _settings(
                profile,
                language,
                preset,
                include_recipe,
                exclude_recipe,
                save_profile,
                theme,
                data.items[0].rule.language if data.items else None,
                input_path,
            )
            parsed, history = RecipePlan.read_json(plan.read_bytes()), load_approvals(approvals)
            report = preview_report(data, parsed, history, settings)
            if production_run is not None:
                from ..pipeline_consumers import preview as production_preview

                if profile is None:
                    raise ValueError("production-run requires the shared pipeline profile")
                try:
                    report["production"] = production_preview(
                        production_run, profile, effective_profile=settings.profile
                    )
                except (ValueError, OSError) as error:
                    report["production"] = {"state": "withheld", "reason": str(error)}
            elif any(r in settings.recipes for r in ("verb_production", "noun_declension", "adjective_declension")):
                report["production"] = {"state": "withheld", "reason": "missing_production_run"}
            if html is not None:
                write_preview(html, data, parsed, history, settings)
    except (OSError, ValueError, RecursionError) as error:
        _error(error)
    # ASCII encoding also prevents source-controlled terminal escape sequences.
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2))


@app.command()
def review(
    input_path: Input,
    permissions: Permissions,
    decisions: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(help="New history file; existing files are never overwritten.")],
    approvals: Approvals = None,
    plan: Plan = None,
    language: Language = None,
    profile: Path | None = None,
    preset: str | None = None,
    include_recipe: list[str] | None = None,
    exclude_recipe: list[str] | None = None,
    theme: Theme = "light",
) -> None:
    """Record an explicit decision array against current candidates, retaining history."""
    try:
        data, _ = load_curated(input_path, permissions)
        history = load_approvals(approvals)
        subjects: dict[tuple[str, str, str], str] | None = None
        if plan is not None:
            if not data.items:
                raise ValueError("no curated items")
            settings = _settings(
                profile,
                language,
                preset,
                include_recipe,
                exclude_recipe,
                None,
                theme,
                data.items[0].rule.language,
                input_path,
            )
            cards, _ = render_cards(data, RecipePlan.read_json(plan.read_bytes()), history, settings)
            subjects = {(c.item_key, c.card_key, settings.language): c.fingerprint for c in cards}
        record_reviews(data, history, load_approvals(decisions, history), output, rendered_subjects=subjects)
    except (OSError, ValueError, RecursionError) as error:
        _error(error)
    if plan is None:
        typer.echo("Recorded explicit decisions; not export authorization. Effective configuration: offline intake v1.")
    else:
        typer.echo(json.dumps({"recorded": True, "effective_configuration": settings.snapshot}))


@app.command(name="export")
def export_command(
    input_path: Input,
    permissions: Permissions,
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    run: Annotated[Path, typer.Option("--run")],
    output: Annotated[Path, typer.Option("--output")],
    approvals: Approvals = None,
    language: Language = None,
    profile: Path | None = None,
    preset: str | None = None,
    include_recipe: list[str] | None = None,
    exclude_recipe: list[str] | None = None,
    save_profile: Path | None = None,
    theme: Theme = "light",
    after_receipt: str | None = None,
    reconcile_receipt: str | None = None,
    production_run: Path | None = None,
    production_output: Path | None = None,
    production_after_receipt: str | None = None,
    production_reconcile_receipt: str | None = None,
) -> None:
    """Export exactly human-approved cards and retain receipts/manual withdrawal actions."""
    try:
        if (production_run is None) != (production_output is None):
            raise ValueError("production-run and production-output are required together")
        if production_run is not None:
            if profile is None or production_output is None:
                raise ValueError("production export requires the shared pipeline profile")
            paths = (run.resolve(), production_run.resolve())
            if any(paths[i].is_relative_to(paths[1 - i]) for i in (0, 1)):
                raise ValueError("curated and production receipt stores must be separate")
            if output.resolve() == production_output.resolve() or any(
                p.resolve().is_relative_to(root) for p in (output, production_output) for root in paths
            ):
                raise ValueError("choose distinct outputs outside both receipt stores")
        load = _loader(input_path, permissions, plan, approvals)
        data, _, _ = load()
        settings = _settings(
            profile,
            language,
            preset,
            include_recipe,
            exclude_recipe,
            save_profile,
            theme,
            data.items[0].rule.language if data.items else None,
            input_path,
        )
        report = export_curated(
            run, output, load, settings, after_receipt=after_receipt, reconcile_receipt=reconcile_receipt
        )
        if production_run is not None and production_output is not None:
            from ..pipeline_consumers import export as production_export

            try:
                report["production"] = production_export(
                    production_run,
                    production_output,
                    profile_path=profile,
                    effective_profile=settings.profile,
                    after_receipt=production_after_receipt,
                    reconcile_receipt=production_reconcile_receipt,
                )
            except (ValueError, OSError) as error:
                # The two note types have separate receipt transactions. Do not
                # describe a completed curated publication as rolled back.
                report["production"] = {
                    "state": "recovery_required",
                    "reason": str(error),
                    "output": str(production_output.resolve()),
                    "run": str(production_run.resolve()),
                    "required_action": (
                        "Diese Ausgabe nicht importieren oder ersetzen. Die unveränderte Anfrage zur "
                        "Belegwiederherstellung wiederholen; die Veröffentlichung kann bereits erfolgt sein."
                        if settings.language == "de"
                        else "Do not import or replace this output. Retry the unchanged request to recover "
                        "its receipt; publication may already have occurred."
                    ),
                }
        elif any(r in settings.recipes for r in ("verb_production", "noun_declension", "adjective_declension")):
            report["production"] = {"state": "withheld", "reason": "missing_production_run", "exported": 0}
    except (OSError, ValueError, RecursionError) as error:
        _error(error)
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2))
