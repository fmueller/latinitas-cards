from pathlib import Path

from click.testing import CliRunner
from typer.main import get_command

from latinitas_cards.cli import app


def test_cli_invokes_without_arguments_shows_help() -> None:
    runner = CliRunner()
    result = runner.invoke(get_command(app), ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output or "--help" in result.output


def test_cli_help_option() -> None:
    runner = CliRunner()
    result = runner.invoke(get_command(app), ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_preview_shows_sample_clozes_without_writing_output(tmp_path: Path) -> None:
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>In principio erat verbum</v><ve/></usfx>",
        encoding="utf-8",
    )

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("Front,Back\nverbum,\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        ["preview", "--input", str(csv_path), "--usfx", str(usfx_path), "--limit", "1"],
    )

    assert result.exit_code == 0
    assert "verbum" in result.output
    assert "{{c1::verbum}}" in result.output


def test_generate_writes_output_csv(tmp_path: Path) -> None:
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>In principio erat verbum</v><ve/></usfx>",
        encoding="utf-8",
    )

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("Front,Back\nverbum,\n", encoding="utf-8")

    output_path = tmp_path / "output.csv"

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "generate",
            "--input",
            str(csv_path),
            "--usfx",
            str(usfx_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    output_text = output_path.read_text(encoding="utf-8")
    assert "VulgataCloze" in output_text
    assert "{{c1::verbum}}" in output_text
