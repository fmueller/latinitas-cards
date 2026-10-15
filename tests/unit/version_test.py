import importlib.metadata
import tomllib
from pathlib import Path

from typer.main import get_command
from typer.testing import CliRunner

from latinitas_cards.cli import app


def test_version_matches_metadata_and_project() -> None:
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    version = importlib.metadata.version("latinitas-cards")
    assert version == project["project"]["version"]
    for args in [["--version"], ["--version", "--help"], ["--version", "generate", "--bad-option"]]:
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert result.stdout == f"latinitas-cards {version}\n"


def test_root_help_description_and_legacy_commands_unchanged() -> None:
    assert get_command(app).help == (
        "Build principal-part study cards from Anki decks with a confirmed --profile. "
        "Review authored notes, managed CSV updates, and contextual form parsing; "
        "inspect, split, or annotate sources, or explore experimental USFX clozes."
    )
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in [
        "generate",
        "preview",
        "inspect",
        "split",
        "annotate",
        "cloze",
        "validate",
        "setup",
        "authored",
        "managed",
        "form-parsing",
        "claims",
        "phrases",
        "terminology",
    ]:
        assert command in result.stdout
