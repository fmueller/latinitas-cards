"""Corpus-first card commands; profile validation reads only the profile file."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from ..cli_contract import Diagnostic, run_contract
from ..corpus_output import export_corpus, inspect_corpus, preview_corpus
from ..corpus_profile import load_corpus_profile, profile_digest
from ..corpus_selection import deselect_corpus, select_corpus

app = typer.Typer(help="Generate study cards directly from normalized corpora.")
profile_app = typer.Typer(help="Validate corpus-generation profiles.")
app.add_typer(profile_app, name="profile")


@profile_app.command()
def validate(
    file: Annotated[Path, typer.Argument(help="latinitas-corpus-profile JSON file.")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Check a corpus profile and report its effective values and digest."""

    def body() -> tuple[JsonValue, list[Diagnostic]]:
        profile = load_corpus_profile(file)
        return {
            "profile_id": profile.profile_id,
            "profile_digest": profile_digest(profile),
            "profile": profile.model_dump(mode="json"),
        }, []

    run_contract("corpus-cards profile validate", json_output, body)


@app.command()
def select(
    profile: Annotated[Path, typer.Argument(help="Corpus-generation profile JSON file.")],
    corpus: Annotated[list[Path], typer.Option("--corpus", help="Normalized corpus; repeat for each profile corpus.")],
    selection: Annotated[Path, typer.Option(help="Sticky selection JSON file.")],
    report: Annotated[Path, typer.Option(help="Candidate report JSON file.")],
    known_lemmas: Annotated[Path | None, typer.Option(help="Confirmed deck profile for known lemmas.")] = None,
    known_lemmas_source: Annotated[Path | None, typer.Option(help="Known-lemma source CSV or package.")] = None,
    confirm_context: Annotated[list[str] | None, typer.Option(help="Reviewed selected ID; repeat as needed.")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Rank current candidates and fill free budgets without dropping earlier selections."""
    run_contract(
        "corpus-cards select",
        json_output,
        lambda: select_corpus(
            profile,
            corpus,
            selection,
            report,
            known_profile=known_lemmas,
            known_source=known_lemmas_source,
            confirm_context=confirm_context or (),
        ),
    )


@app.command()
def deselect(
    selection: Annotated[Path, typer.Argument(help="Existing sticky selection JSON file.")],
    exercise: Annotated[list[str], typer.Option("--exercise", help="LatinitasID; repeat as needed.")],
    reason: Annotated[str, typer.Option(help="Human reason for deselection.")],
    restore: Annotated[bool, typer.Option(help="Restore stored entries instead of deselecting.")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Deselect exercises or restore them with their previously reviewed context digest."""
    run_contract(
        "corpus-cards deselect",
        json_output,
        lambda: deselect_corpus(selection, exercise, reason, restore=restore),
    )


@app.command()
def preview(
    profile: Annotated[Path, typer.Argument(help="Corpus-generation profile JSON file.")],
    corpus: Annotated[list[Path], typer.Option("--corpus")],
    selection: Annotated[Path, typer.Option()],
    limit: Annotated[int, typer.Option(min=1, max=1000)] = 20,
    known_lemmas: Annotated[Path | None, typer.Option()] = None,
    known_lemmas_source: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show current selected exercises and their fact provenance without writing files."""
    run_contract(
        "corpus-cards preview",
        json_output,
        lambda: preview_corpus(
            profile, corpus, selection, limit, known_profile=known_lemmas, known_source=known_lemmas_source
        ),
    )


@app.command(name="export")
def export_csv(
    profile: Annotated[Path, typer.Argument(help="Corpus-generation profile JSON file.")],
    corpus: Annotated[list[Path], typer.Option("--corpus")],
    selection: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Create a new CSV of eligible selected exercises; never overwrite an existing file."""
    run_contract("corpus-cards export", json_output, lambda: export_corpus(profile, corpus, selection, output))


@app.command()
def inspect(
    profile: Annotated[Path, typer.Argument(help="Corpus-generation profile JSON file.")],
    corpus: Annotated[list[Path], typer.Option("--corpus")],
    selection: Annotated[Path, typer.Option()],
    check: Annotated[Path | None, typer.Option(help="Compare a CSV with current reviewed fields.")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Inspect selected exercises and optionally check an export, without writing files."""
    run_contract("corpus-cards inspect", json_output, lambda: inspect_corpus(profile, corpus, selection, check))
