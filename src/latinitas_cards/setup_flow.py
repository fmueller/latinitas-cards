"""Thin CLI entry point for assisted deck-profile setup."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from .profile import ProfileOverrides, ProfileValidationError, load_profile, resolve_profile
from .profile_setup import (
    ProfileSetupError,
    ProfileSetupProposal,
    apply_profile_overrides,
    build_representative_examples,
    profile_source_issues,
    propose_profile_from_inspection,
    safe_source_value,
)
from .sources import CanonicalSourceError, SourceInspection, inspect_source


def setup(
    input: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to a CSV, .apkg, or .colpkg source to inspect",
            exists=True,
            readable=True,
        ),
    ],
    profile: Annotated[
        Path,
        typer.Option(help="Versioned profile path to read or save"),
    ] = Path(".latinitas/profile.json"),
    note_type: Annotated[
        str | None,
        typer.Option(help="Explicit source note type override"),
    ] = None,
    lexical_entry_field: Annotated[
        str | None,
        typer.Option("--lexical-entry-field", "--lexical-field", help="Explicit lexical-entry field override"),
    ] = None,
    principal_parts_field: Annotated[
        str | None,
        typer.Option(help="Explicit principal-parts field override"),
    ] = None,
    meaning_field: Annotated[
        str | None,
        typer.Option(help="Explicit optional meaning or German-gloss field override"),
    ] = None,
    without_meaning_field: Annotated[
        bool,
        typer.Option(help="Reject the optional meaning/gloss field proposal"),
    ] = False,
    separator: Annotated[
        list[str] | None,
        typer.Option("--separator", help="Principal-part separator; repeat for position-specific separators"),
    ] = None,
    role: Annotated[
        list[str] | None,
        typer.Option("--role", help="Named principal-part role; repeat in semantic order"),
    ] = None,
    generated_note_type: Annotated[
        str | None,
        typer.Option(help="Generated note type override"),
    ] = None,
    target_deck: Annotated[
        str | None,
        typer.Option(help="Generated target deck override"),
    ] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Generated tag; repeat for multiple tags"),
    ] = None,
    recipe: Annotated[
        list[str] | None,
        typer.Option("--recipe", help="Confirmed recipe; repeat for multiple recipes"),
    ] = None,
    language_tag: Annotated[
        str | None,
        typer.Option(help="Generated-content language tag override"),
    ] = None,
    source_identity_strategy: Annotated[
        str | None,
        typer.Option(help="Source identity strategy override: note_guid, source_id_field, or manifest"),
    ] = None,
    source_id_field: Annotated[
        str | None,
        typer.Option(help="Stable source-ID field when using source_id_field"),
    ] = None,
    non_interactive: Annotated[
        bool,
        typer.Option(help="Do not prompt; an initial save also requires --confirm"),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option("--confirm", "--yes", help="Explicitly confirm the proposed profile for saving"),
    ] = False,
    reconfigure: Annotated[
        bool,
        typer.Option(help="Inspect and confirm a new proposal even when a profile already exists"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the effective configuration as machine-readable JSON"),
    ] = False,
) -> None:
    """Inspect a source, confirm a reusable profile, and save it safely."""

    if json_output and not non_interactive:
        _error("--json requires --non-interactive so the output remains one machine-readable document.", True)
        raise typer.Exit(code=2)
    if _paths_alias(input, profile):
        _error("The profile path must not alias the input source path.", json_output)
        raise typer.Exit(code=2)

    try:
        inspection = inspect_source(input)
        explicit_overrides = _build_overrides(
            note_type=note_type,
            lexical_entry_field=lexical_entry_field,
            principal_parts_field=principal_parts_field,
            meaning_field=meaning_field,
            without_meaning_field=without_meaning_field,
            separator=separator,
            role=role,
            generated_note_type=generated_note_type,
            target_deck=target_deck,
            tag=tag,
            recipe=recipe,
            language_tag=language_tag,
            source_identity_strategy=source_identity_strategy,
            source_id_field=source_id_field,
        )
    except (CanonicalSourceError, ProfileSetupError, ProfileValidationError, ValueError) as error:
        _error(str(error), json_output)
        raise typer.Exit(code=2) from error

    if profile.exists() and not reconfigure:
        _reuse_profile(
            profile_path=profile,
            inspection=inspection,
            explicit_overrides=explicit_overrides,
            json_output=json_output,
        )
        return

    try:
        proposal = propose_profile_from_inspection(inspection, input, note_type=note_type)
        proposal = apply_profile_overrides(proposal, explicit_overrides)
        if not json_output:
            _print_proposal(proposal, status="proposal")
        if not non_interactive and not confirm:
            proposal = _prompt_for_corrections(proposal)
            if not json_output:
                _print_proposal(proposal, status="corrected")
        issues = profile_source_issues(proposal.profile, inspection)
        if issues:
            raise ProfileSetupError("Profile is not compatible with the inspected source: " + "; ".join(issues))
    except (CanonicalSourceError, ProfileSetupError, ProfileValidationError, ValueError) as error:
        _error(str(error), json_output)
        raise typer.Exit(code=2) from error

    if not confirm and (non_interactive or not typer.confirm("Save this confirmed profile?", default=False)):
        _emit(
            {"status": "cancelled", "profile_path": str(profile)},
            json_output,
            "Profile setup cancelled; no profile was saved.",
        )
        return

    try:
        profile.parent.mkdir(parents=True, exist_ok=True)
        proposal.profile.save(profile)
    except OSError as error:
        _error(f"Could not save profile: {error}", json_output)
        raise typer.Exit(code=2) from error
    payload = _result_payload("saved", profile, proposal)
    _emit(payload, json_output, f"Saved confirmed profile to {profile}.")


def _reuse_profile(
    *,
    profile_path: Path,
    inspection: SourceInspection,
    explicit_overrides: ProfileOverrides,
    json_output: bool,
) -> None:
    try:
        saved_profile = load_profile(profile_path)
        effective_profile = resolve_profile(saved_profile, explicit_overrides)
        issues = profile_source_issues(effective_profile, inspection)
        if issues:
            raise ProfileSetupError(
                "Saved profile is invalid for this source: "
                + "; ".join(issues)
                + ". Re-run with --reconfigure after reviewing the proposal."
            )
        records = tuple(
            record
            for record in inspection.records
            if record.note_type is None or record.note_type == effective_profile.note_type
        )
        examples = build_representative_examples(records, effective_profile)
        proposal = ProfileSetupProposal(
            source_kind=records[0].source_kind,
            record_count=len(records),
            note_types=inspection.note_types,
            fields=tuple(sorted({field for record in records for field in record.fields})),
            profile=effective_profile,
            field_candidates=(),
            examples=examples,
            uncertainties=(
                "This confirmed profile was reused deterministically; setup prompts were not repeated.",
                "Structural principal-part matches do not confirm verb eligibility or the semantic identity "
                "of the PPP.",
            ),
            recipe_suggestions=effective_profile.selected_recipes,
        )
    except (OSError, ProfileValidationError, ProfileSetupError, ValueError) as error:
        _error(str(error), json_output)
        raise typer.Exit(code=2) from error

    if not json_output:
        _print_proposal(proposal, status="reused")
    _emit(
        _result_payload("reused", profile_path, proposal),
        json_output,
        f"Reused confirmed profile from {profile_path}.",
    )


def _build_overrides(
    *,
    note_type: str | None,
    lexical_entry_field: str | None,
    principal_parts_field: str | None,
    meaning_field: str | None,
    without_meaning_field: bool,
    separator: list[str] | None,
    role: list[str] | None,
    generated_note_type: str | None,
    target_deck: str | None,
    tag: list[str] | None,
    recipe: list[str] | None,
    language_tag: str | None,
    source_identity_strategy: str | None,
    source_id_field: str | None,
) -> ProfileOverrides:
    fields: dict[str, str | None] = {}
    if lexical_entry_field is not None:
        fields["lexical_entry_field"] = lexical_entry_field
    if principal_parts_field is not None:
        fields["principal_parts_field"] = principal_parts_field
    if meaning_field is not None:
        fields["meaning_field"] = meaning_field
    if without_meaning_field:
        fields["meaning_field"] = None

    principal_parts: dict[str, tuple[str, ...]] = {}
    if role:
        principal_parts["roles"] = tuple(role)
    if separator:
        principal_parts["separators"] = tuple(separator)

    source_identity: dict[str, str] = {}
    if source_identity_strategy is not None:
        source_identity["strategy"] = source_identity_strategy
    if source_id_field is not None:
        source_identity["strategy"] = "source_id_field"
        source_identity["field"] = source_id_field

    values: dict[str, Any] = {}
    if source_identity:
        values["source_identity"] = source_identity
    if note_type is not None:
        values["note_type"] = note_type
    if fields:
        values["fields"] = fields
    if principal_parts:
        values["principal_parts"] = principal_parts
    for name, value in (
        ("language_tag", language_tag),
        ("generated_note_type", generated_note_type),
        ("target_deck", target_deck),
    ):
        if value is not None:
            values[name] = value
    if tag:
        values["tags"] = tuple(tag)
    if recipe:
        values["selected_recipes"] = tuple(recipe)
    return ProfileOverrides.from_mapping(values)


def _prompt_for_corrections(proposal: ProfileSetupProposal) -> ProfileSetupProposal:
    current = proposal.profile
    note_type = typer.prompt("Source note type", default=current.note_type)
    lexical_field = typer.prompt("Lexical-entry field", default=current.fields.lexical_entry_field)
    principal_field = typer.prompt("Principal-parts field", default=current.fields.principal_parts_field)
    meaning_field = typer.prompt(
        "Optional German gloss/meaning field (blank for none)",
        default="",
        show_default=False,
    )
    roles_text = typer.prompt(
        "Named principal-part roles, comma-separated",
        default=", ".join(current.principal_parts.roles),
    )
    separators_text = typer.prompt(
        "Principal-part separators as a JSON list",
        default=json.dumps(list(current.principal_parts.separators), ensure_ascii=False),
    )
    try:
        separators = json.loads(separators_text)
    except json.JSONDecodeError as error:
        raise ProfileSetupError("Principal-part separators must be a JSON list of strings.") from error
    if not isinstance(separators, list) or not all(isinstance(value, str) for value in separators):
        raise ProfileSetupError("Principal-part separators must be a JSON list of strings.")

    generated_note_type = typer.prompt("Generated note type", default=current.generated_note_type)
    target_deck = typer.prompt("Target deck", default=current.target_deck)
    language_tag = typer.prompt("Generated-content language tag", default=current.language_tag)
    tags_text = typer.prompt("Generated tags, comma-separated", default=", ".join(current.tags))
    recipes_text = typer.prompt("Confirmed recipes, comma-separated", default=", ".join(current.selected_recipes))
    overrides = {
        "note_type": note_type,
        "fields": {
            "lexical_entry_field": lexical_field,
            "principal_parts_field": principal_field,
            "meaning_field": meaning_field or None,
        },
        "principal_parts": {
            "roles": tuple(value.strip() for value in roles_text.split(",")),
            "separators": tuple(separators),
        },
        "generated_note_type": generated_note_type,
        "target_deck": target_deck,
        "language_tag": language_tag,
        "tags": tuple(value.strip() for value in tags_text.split(",")),
        "selected_recipes": tuple(value.strip() for value in recipes_text.split(",")),
    }
    return apply_profile_overrides(proposal, overrides)


def _print_proposal(proposal: ProfileSetupProposal, *, status: str) -> None:
    profile = proposal.profile
    typer.echo(f"Profile {safe_source_value(status, 'status')}: inspected {proposal.record_count} record(s).")
    typer.echo(f"Source kind: {safe_source_value(proposal.source_kind, 'source kind')}")
    typer.echo(f"Note type: {safe_source_value(profile.note_type, 'note type')}")
    typer.echo(
        "Lexical-entry field: "
        + safe_source_value(profile.fields.lexical_entry_field, profile.fields.lexical_entry_field)
    )
    typer.echo(
        "Principal-parts field: "
        + safe_source_value(profile.fields.principal_parts_field, profile.fields.principal_parts_field)
    )
    source_identity = profile.source_identity
    identity_text: str = source_identity.strategy
    if source_identity.field is not None:
        identity_text += f" (field: {safe_source_value(source_identity.field, source_identity.field)})"
    typer.echo(f"Source identity: {identity_text}")
    meaning = profile.fields.meaning_field
    typer.echo(
        "Optional German gloss/meaning field: "
        + (safe_source_value(meaning, meaning) if meaning is not None else "(none)")
    )
    typer.echo(f"Separators: {list(profile.principal_parts.separators)!r}")
    typer.echo(f"Named role order: {', '.join(profile.principal_parts.roles)}")
    typer.echo(f"Generated note type: {safe_source_value(profile.generated_note_type, 'generated note type')}")
    typer.echo(f"Target deck: {safe_source_value(profile.target_deck, 'target deck')}")
    typer.echo(f"Tags: {safe_source_value(', '.join(profile.tags), 'tags')}")
    typer.echo(f"Generated-content language tag: {safe_source_value(profile.language_tag, 'language tag')}")
    typer.echo(f"Compatible recipes: {safe_source_value(', '.join(proposal.recipe_suggestions), 'recipes')}")
    typer.echo("Representative examples:")
    for example in proposal.examples:
        typer.echo(
            f"  {safe_source_value(example.source_location, 'source location')}: "
            f"{example.lexical_entry} -> {example.principal_parts} "
            f"[{example.structural_status}] {example.structural_message}"
        )
    typer.echo("Uncertainty:")
    for uncertainty in proposal.uncertainties:
        typer.echo(f"  - {uncertainty}")


def _result_payload(status: str, profile_path: Path, proposal: ProfileSetupProposal) -> dict[str, Any]:
    effective = proposal.profile.to_machine_readable()
    return {
        "status": status,
        "profile_path": str(profile_path),
        "profile": effective,
        "effective_profile": effective,
        "examples": [example.to_machine_readable() for example in proposal.examples],
        "uncertainties": list(proposal.uncertainties),
        "compatible_recipes": list(proposal.recipe_suggestions),
    }


def _emit(payload: Mapping[str, Any], json_output: bool, human_message: str) -> None:
    if json_output:
        typer.echo(json.dumps(dict(payload), ensure_ascii=False, indent=2))
    else:
        typer.echo(human_message)


def _error(message: str, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps({"status": "error", "message": message}, ensure_ascii=False), err=True)
    else:
        typer.echo(f"Error: {message}", err=True)


def _paths_alias(input_path: Path, profile_path: Path) -> bool:
    try:
        if input_path.resolve() == profile_path.resolve():
            return True
        return profile_path.exists() and os.path.samefile(input_path, profile_path)
    except OSError:
        return False


setup.__module__ = __name__
