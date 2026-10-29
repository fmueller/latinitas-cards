"""Corpus-first card commands; profile validation reads only the profile file."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from ..cli_contract import Diagnostic, run_contract
from ..corpus_profile import load_corpus_profile, profile_digest

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
