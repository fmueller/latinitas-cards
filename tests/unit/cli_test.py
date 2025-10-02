from click.testing import CliRunner
from typer.main import get_command

from latinitas_cards.cli import app


def test_cli_invokes_without_arguments_shows_help():
    runner = CliRunner()
    result = runner.invoke(get_command(app), ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output or "--help" in result.output


def test_cli_help_option():
    runner = CliRunner()
    result = runner.invoke(get_command(app), ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output
