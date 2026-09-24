from pathlib import Path
from typing import cast

import click
import pytest
from click.testing import CliRunner
from typer.main import get_command as _typer_get_command

from latinitas_cards.cli import app
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig


def _profile() -> DeckProfile:
    return DeckProfile.default(
        note_type="CSV source",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="Stable ID"),
        principal_part_roles=("present_infinitive", "present_1s", "perfect_1s", "supine"),
        separators=(",",),
        selected_recipes=("principal_part_recognition",),
        generated_note_type="Latinitas Principal Parts",
        target_deck="Latin::Latinitas::Review",
        tags=("latinitas", "provenance"),
    )


def _write_source(path: Path) -> None:
    path.write_text(
        'Stable ID,Lemma,Forms,German gloss\nentry-1,dīcō,"dīcere, dīcō, dīxī, dictum",sagen\n',
        encoding="utf-8",
    )


def _command() -> click.Command:
    return cast(click.Command, _typer_get_command(app))


def test_profile_preview_renders_counts_prompt_answer_and_provenance_without_output(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    _write_source(source)
    _profile().save(profile_path)

    result = CliRunner().invoke(
        _command(),
        ["preview", "--input", str(source), "--profile", str(profile_path), "--limit", "1"],
    )

    assert result.exit_code == 0
    assert "Generated: 4" in result.stdout
    assert "Skipped: 0" in result.stdout
    assert "Ambiguous: 0" in result.stdout
    assert "Welche Stammform" in result.stdout
    assert "Bedeutung" in result.stdout
    assert "csv row 2" in result.stdout
    assert "Output:" not in result.stdout


def test_profile_generate_renders_preview_before_writing_deterministic_output(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    output = tmp_path / "generated.csv"
    _write_source(source)
    _profile().save(profile_path)
    source_before = source.read_bytes()
    profile_before = profile_path.read_bytes()

    result = CliRunner().invoke(
        _command(),
        [
            "generate",
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.index("Generated: 4") < result.stdout.index("Output:")
    assert output.read_bytes().startswith(b"#separator:Comma\n")
    assert source.read_bytes() == source_before
    assert profile_path.read_bytes() == profile_before


@pytest.mark.parametrize("command_name", ["preview", "generate"])
def test_legacy_usfx_path_is_validated_before_corpus_parsing(tmp_path: Path, command_name: str) -> None:
    source = tmp_path / "source.csv"
    missing_usfx = tmp_path / "missing.usfx.xml"
    _write_source(source)
    arguments = [command_name, "--input", str(source), "--usfx", str(missing_usfx)]
    if command_name == "generate":
        arguments.extend(["--output", str(tmp_path / "generated.csv")])

    result = CliRunner().invoke(_command(), arguments)

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_terminal_preview_escapes_c1_controls_in_source_values(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    source.write_text(
        'Stable ID,Lemma,Forms,German gloss\n"entry-\x9b31m\u2028\u2029",dīcō,"dīcere, dīcō, dīxī, dictum",sagen\n',
        encoding="utf-8",
    )
    _profile().save(profile_path)

    result = CliRunner().invoke(
        _command(),
        ["preview", "--input", str(source), "--profile", str(profile_path), "--limit", "1"],
    )

    assert result.exit_code == 0
    assert "\\x9b" in result.stdout
    assert "\\u2028" in result.stdout
    assert "\\u2029" in result.stdout
    assert "\x9b" not in result.stdout
    assert "\u2028" not in result.stdout
    assert "\u2029" not in result.stdout


def test_terminal_preview_redacts_sensitive_mapped_source_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    source.write_text(
        'Stable ID,Lemma,Forms,Password\nentry-1,dīcō,"dīcere, dīcō, dīxī, dictum",SENTINEL_SECRET_VALUE\n',
        encoding="utf-8",
    )
    DeckProfile.default(
        note_type="CSV source",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="Password",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="Stable ID"),
        principal_part_roles=("present_infinitive", "present_1s", "perfect_1s", "supine"),
        separators=(",",),
        selected_recipes=("principal_part_recognition",),
    ).save(profile_path)

    result = CliRunner().invoke(
        _command(),
        ["preview", "--input", str(source), "--profile", str(profile_path), "--limit", "1"],
    )

    assert result.exit_code == 0
    assert "SENTINEL_SECRET_VALUE" not in result.stdout
    assert "[redacted sensitive source field]" in result.stdout


def test_terminal_preview_bounds_structured_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    rows = ["Stable ID,Lemma,Forms,German gloss"]
    rows.extend(f"entry-{index},dīcō,,sagen" for index in range(60))
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _profile().save(profile_path)

    result = CliRunner().invoke(
        _command(),
        ["preview", "--input", str(source), "--profile", str(profile_path), "--limit", "0"],
    )

    assert result.exit_code == 0
    assert "additional structured skips omitted" in result.stdout
    assert result.stdout.count("missing_principal_parts") <= 50
