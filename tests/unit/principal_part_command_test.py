import csv
import io
import os
import sqlite3
import zipfile
from pathlib import Path
from typing import cast

import click
import pytest
from click.testing import CliRunner
from typer.main import get_command as _typer_get_command

from latinitas_cards.cli import app
from latinitas_cards.preview_export import prepare_principal_part_export, write_principal_part_csv
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


def test_profile_preview_and_generate_render_combined_tags_matching_the_csv(tmp_path: Path) -> None:
    database = tmp_path / "tagged.anki2"
    con = sqlite3.connect(database)
    con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER, tags TEXT, flds TEXT)")
    con.execute("CREATE TABLE notetypes (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("CREATE TABLE fields (ntid INTEGER, ord INTEGER, name TEXT)")
    con.execute("INSERT INTO notetypes (id, name) VALUES (10, 'CSV source')")
    con.executemany(
        "INSERT INTO fields (ntid, ord, name) VALUES (?, ?, ?)",
        ((10, 0, "Lemma"), (10, 1, "Forms"), (10, 2, "German gloss")),
    )
    con.execute(
        "INSERT INTO notes (id, guid, mid, tags, flds) VALUES (1, 'guid-a', 10, 'latin verb::irregular',"
        " 'dīcō\x1fdīcere, dīcō, dīxī, dictum\x1fsagen')"
    )
    con.commit()
    con.close()
    source = tmp_path / "tagged.apkg"
    with zipfile.ZipFile(source, "w") as archive:
        archive.write(database, "collection.anki2")
    profile_path = tmp_path / "profile.json"
    output = tmp_path / "generated.csv"
    DeckProfile.default(
        note_type="CSV source",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="note_guid"),
        principal_part_roles=("present_infinitive", "present_1s", "perfect_1s", "supine"),
        separators=(",",),
        selected_recipes=("principal_part_recognition",),
        generated_note_type="Latinitas Principal Parts",
        target_deck="Latin::Latinitas::Review",
        tags=("latinitas", "latin"),
    ).save(profile_path)

    previewed = CliRunner().invoke(
        _command(),
        ["preview", "--input", str(source), "--profile", str(profile_path), "--limit", "1"],
    )
    generated = CliRunner().invoke(
        _command(),
        ["generate", "--input", str(source), "--profile", str(profile_path), "--output", str(output)],
    )

    expected_tags = "latin verb::irregular latinitas"
    assert previewed.exit_code == 0
    assert f"Tags: {expected_tags}" in previewed.stdout
    assert generated.exit_code == 0
    assert f"Tags: {expected_tags}" in generated.stdout
    rows = list(csv.reader(io.StringIO(output.read_text(encoding="utf-8"))))
    tag_rows = [row for row in rows if row and row[0].startswith("latinitas-v1-")]
    assert tag_rows
    assert all(row[3] == expected_tags for row in tag_rows)


def test_profile_preview_tags_line_matches_the_csv_beyond_the_default_limit(tmp_path: Path) -> None:
    long_parent_tags = " ".join(f"chapter::{number:03d}" for number in range(1, 21))
    database = tmp_path / "long-tags.anki2"
    con = sqlite3.connect(database)
    con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER, tags TEXT, flds TEXT)")
    con.execute("CREATE TABLE notetypes (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("CREATE TABLE fields (ntid INTEGER, ord INTEGER, name TEXT)")
    con.execute("INSERT INTO notetypes (id, name) VALUES (10, 'CSV source')")
    con.executemany(
        "INSERT INTO fields (ntid, ord, name) VALUES (?, ?, ?)",
        ((10, 0, "Lemma"), (10, 1, "Forms"), (10, 2, "German gloss")),
    )
    con.execute(
        "INSERT INTO notes (id, guid, mid, tags, flds) VALUES (1, 'guid-a', 10, ?,"
        " 'dīcō\x1fdīcere, dīcō, dīxī, dictum\x1fsagen')",
        (long_parent_tags,),
    )
    con.commit()
    con.close()
    source = tmp_path / "long-tags.apkg"
    with zipfile.ZipFile(source, "w") as archive:
        archive.write(database, "collection.anki2")
    profile_path = tmp_path / "profile.json"
    output = tmp_path / "generated.csv"
    DeckProfile.default(
        note_type="CSV source",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="note_guid"),
        principal_part_roles=("present_infinitive", "present_1s", "perfect_1s", "supine"),
        separators=(",",),
        selected_recipes=("principal_part_recognition",),
        generated_note_type="Latinitas Principal Parts",
        target_deck="Latin::Latinitas::Review",
        tags=("latinitas",),
    ).save(profile_path)
    expected_tags = f"{long_parent_tags} latinitas"
    assert len(expected_tags) > 160

    previewed = CliRunner().invoke(
        _command(),
        ["preview", "--input", str(source), "--profile", str(profile_path), "--limit", "1"],
    )
    generated = CliRunner().invoke(
        _command(),
        ["generate", "--input", str(source), "--profile", str(profile_path), "--output", str(output)],
    )

    assert previewed.exit_code == 0
    assert f"Tags: {expected_tags}" in previewed.stdout
    assert generated.exit_code == 0
    assert f"Tags: {expected_tags}" in generated.stdout
    rows = list(csv.reader(io.StringIO(output.read_text(encoding="utf-8"))))
    tag_rows = [row for row in rows if row and row[0].startswith("latinitas-v1-")]
    assert tag_rows
    assert all(row[3] == expected_tags for row in tag_rows)


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


def test_terminal_generate_keeps_recovery_status_before_long_destination_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    manifest = Path(f"{source}.latinitas.json")
    output = tmp_path / ("generated-" + "x" * 140 + ".csv")
    profile = DeckProfile.default(
        note_type="CSV source",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="manifest"),
        principal_part_roles=("present_infinitive", "present_1s", "perfect_1s", "supine"),
        separators=(",",),
        selected_recipes=("principal_part_recognition",),
    )
    _write_source(source)
    profile.save(profile_path)
    prepared = prepare_principal_part_export(
        source,
        profile,
        manifest_path=manifest,
        approved_allocations={0},
    )
    write_principal_part_csv(prepared, output)

    original_replace = os.replace
    manifest_failed = False

    def fail_manifest_once(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal manifest_failed
        if not manifest_failed and not isinstance(destination_name, bytes) and Path(destination_name) == manifest:
            manifest_failed = True
            raise OSError("simulated manifest commit failure")
        original_replace(source_name, destination_name)

    original_unlink = Path.unlink

    def fail_output_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path == output:
            raise OSError("simulated output rollback failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", fail_manifest_once)
    monkeypatch.setattr(Path, "unlink", fail_output_unlink)

    result = CliRunner().invoke(
        _command(),
        ["generate", "--input", str(source), "--profile", str(profile_path), "--output", str(output)],
    )

    assert result.exit_code == 2
    assert manifest_failed
    assert output.exists()
    assert "Recovery is required" in result.output
    assert "backups were retained" in result.output
    assert "No output or manifest was changed" not in result.output
