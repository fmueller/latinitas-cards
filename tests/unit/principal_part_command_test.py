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


def _commit_scope_state(source: Path, profile: DeckProfile) -> None:
    from latinitas_cards.preview_export import prepare_principal_part_export, write_principal_part_csv

    prepared = prepare_principal_part_export(
        source,
        profile,
        manifest_path=Path(f"{source}.latinitas.json"),
        approve_new_scope=True,
    )
    write_principal_part_csv(prepared, source.parent / "bootstrap.csv")


def test_profile_preview_requests_csv_scope_confirmation_without_minted_output(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    _write_source(source)
    _profile().save(profile_path)

    result = CliRunner().invoke(
        _command(),
        ["preview", "--input", str(source), "--profile", str(profile_path), "--limit", "1"],
    )

    assert result.exit_code == 0
    assert "Objects: 0" in result.stdout
    assert "Cards: 0" in result.stdout
    assert "Claim sample: 0 generated objects from 1 source entries" in result.stdout
    assert "Accepted reviewed claims: 0/0 bound claim assessments" in result.stdout
    assert "Withheld assessed claims: 0/0 bound claim assessments" in result.stdout
    assert "Withheld roles without claim assessments: 0/0 non-absent comparison roles" in result.stdout
    assert "source scope" in result.stdout.lower()
    assert "scope confirmation" in result.stdout.lower()
    assert "latinitas-v2-" not in result.stdout
    assert "Output:" not in result.stdout
    assert not Path(f"{source}.latinitas.json").exists()


def test_cli_preview_counts_missing_reviews_separately_from_claims(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    _write_source(source)
    _profile().save(profile_path)
    _commit_scope_state(source, _profile())
    result = CliRunner().invoke(
        _command(),
        ["preview", "--input", str(source), "--profile", str(profile_path), "--limit", "0"],
    )
    assert result.exit_code == 0
    assert "Claim sample: 1 generated objects from 1 source entries" in result.stdout
    assert "Accepted reviewed claims: 0/0 bound claim assessments" in result.stdout
    assert "Withheld assessed claims: 0/0 bound claim assessments" in result.stdout
    assert "Withheld roles without claim assessments: 1/4 non-absent comparison roles" in result.stdout
    assert "Cards: 3" in result.stdout
    assert "Generated entries with warnings: 1 (overlaps generated)" in result.stdout
    assert "Wholly skipped entries: 0/1" in result.stdout


def test_read_only_preview_rejects_scope_approval(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    _write_source(source)
    _profile().save(profile_path)

    result = CliRunner().invoke(
        _command(),
        ["preview", "--input", str(source), "--profile", str(profile_path), "--approve-scope"],
    )

    assert result.exit_code != 0
    assert "approve-scope" in click.unstyle(result.output)


def test_generate_requires_scope_approval_before_writing_csv_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    output = tmp_path / "generated.csv"
    _write_source(source)
    _profile().save(profile_path)

    result = CliRunner().invoke(
        _command(),
        ["generate", "--input", str(source), "--profile", str(profile_path), "--output", str(output)],
    )

    assert result.exit_code == 2
    assert "source scope" in result.output.lower()
    assert not output.exists()
    assert not Path(f"{source}.latinitas.json").exists()


def test_generate_rejects_a_generated_note_type_declared_legacy_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    output = tmp_path / "generated.csv"
    _write_source(source)
    _profile().save(profile_path)

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
            "--legacy-note-type",
            "Latinitas Principal Parts",
        ],
    )

    assert result.exit_code == 2
    rendered = click.unstyle(result.output)
    assert "legacy note model" in rendered
    assert "Latinitas Principal Parts" in rendered
    assert not output.exists()
    assert not Path(f"{source}.latinitas.json").exists()


def test_preview_rejects_a_generated_note_type_declared_legacy_without_minting_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    _write_source(source)
    _profile().save(profile_path)

    result = CliRunner().invoke(
        _command(),
        [
            "preview",
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--legacy-note-type",
            "Latinitas Principal Parts",
        ],
    )

    assert result.exit_code == 2
    assert "legacy note model" in click.unstyle(result.output)
    assert not Path(f"{source}.latinitas.json").exists()


@pytest.mark.parametrize("command_name", ["generate", "preview"])
def test_legacy_note_type_option_requires_profile(tmp_path: Path, command_name: str) -> None:
    source = tmp_path / "source.csv"
    _write_source(source)
    arguments = [
        command_name,
        "--input",
        str(source),
        "--legacy-note-type",
        "Latinitas Legacy Exercise",
    ]
    if command_name == "generate":
        arguments.extend(["--output", str(tmp_path / "generated.csv")])

    result = CliRunner().invoke(_command(), arguments)

    assert result.exit_code != 0
    assert "Profile-only options require --profile." in click.unstyle(result.output)


@pytest.mark.parametrize("command_name", ["generate", "preview"])
@pytest.mark.parametrize("declared", ["", "   "])
def test_legacy_note_type_rejects_blank_declarations(tmp_path: Path, command_name: str, declared: str) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    output = tmp_path / "generated.csv"
    _write_source(source)
    _profile().save(profile_path)
    arguments = [
        command_name,
        "--input",
        str(source),
        "--profile",
        str(profile_path),
        "--legacy-note-type",
        declared,
    ]
    if command_name == "generate":
        arguments.extend(["--output", str(output)])

    result = CliRunner().invoke(_command(), arguments)

    assert result.exit_code == 2
    assert "--legacy-note-type requires a non-empty note type name." in click.unstyle(result.output)
    assert not output.exists()
    assert not Path(f"{source}.latinitas.json").exists()
    assert not Path(f"{source}.latinitas-cards.json").exists()


def test_generate_with_scope_approval_commits_state_and_object_output(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    output = tmp_path / "generated.csv"
    state = Path(f"{source}.latinitas.json")
    _write_source(source)
    _profile().save(profile_path)

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
            "--approve-scope",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.index("Notes: 1") < result.stdout.index("Output:")
    assert "LatinitasID: latinitas-v2-" in result.stdout
    assert output.read_bytes().startswith(b"#separator:Comma\n")
    assert state.exists()

    repeat = CliRunner().invoke(
        _command(),
        ["generate", "--input", str(source), "--profile", str(profile_path), "--output", str(output)],
    )
    assert repeat.exit_code == 0
    assert "source scope" not in repeat.stdout.lower()


def _write_csv_source(path: Path, forms: str) -> None:
    path.write_text(
        f'Stable ID,Lemma,Forms,German gloss\nentry-1,dīcō,"{forms}",sagen\n',
        encoding="utf-8",
    )


def test_preview_reports_withheld_rows_and_card_reviews_without_writing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    output = tmp_path / "generated.csv"
    _write_csv_source(source, "dīcere, dīcō, dīxī, dictum")
    _profile().save(profile_path)
    committed = CliRunner().invoke(
        _command(),
        [
            "generate",
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--output",
            str(output),
            "--approve-scope",
        ],
    )
    assert committed.exit_code == 0
    _write_csv_source(source, "dīcere, dīcō, , dictum")

    previewed = CliRunner().invoke(
        _command(),
        ["preview", "--input", str(source), "--profile", str(profile_path), "--limit", "1"],
    )
    withheld = CliRunner().invoke(
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

    assert previewed.exit_code == 0
    assert "Notes: 0" in previewed.stdout
    assert "Zero-eligible notes: 0" in previewed.stdout
    assert "Card eligibility reviews:" in previewed.stdout
    assert "withheld" in previewed.stdout
    assert "perfect_1s" in previewed.stdout
    assert withheld.exit_code == 0
    rows = [row for row in csv.reader(io.StringIO(output.read_text(encoding="utf-8"))) if row]
    assert all(row[0].startswith("#") for row in rows)


def test_generate_requires_fresh_import_approval_for_a_damaged_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    output = tmp_path / "generated.csv"
    checkpoint = Path(f"{source}.latinitas-cards.json")
    _write_csv_source(source, "dīcere, dīcō, dīxī, dictum")
    _profile().save(profile_path)
    first = CliRunner().invoke(
        _command(),
        [
            "generate",
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--output",
            str(output),
            "--approve-scope",
        ],
    )
    assert first.exit_code == 0
    checkpoint.write_text("{damaged", encoding="utf-8")

    gated = CliRunner().invoke(
        _command(),
        ["generate", "--input", str(source), "--profile", str(profile_path), "--output", str(output)],
    )
    fresh = CliRunner().invoke(
        _command(),
        [
            "generate",
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--output",
            str(output),
            "--approve-fresh-import",
        ],
    )

    assert gated.exit_code != 0
    guidance = " ".join(click.unstyle(gated.output).split())
    assert "--approve-fresh-import" in guidance
    assert "new scheduling, not a scheduling migration" in guidance
    assert "replaces retained card-evidence state" in guidance
    assert gated.exception is not None
    assert fresh.exit_code == 0
    assert "Notes: 1" in fresh.stdout
    from latinitas_cards.checkpoint import PriorExportCheckpoint

    restored = PriorExportCheckpoint.load(checkpoint)
    assert restored.source_scope
    assert restored.objects and all(keys for _note_id, keys in restored.objects)


def test_read_only_preview_rejects_fresh_import_approval(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    _write_csv_source(source, "dīcere, dīcō, dīxī, dictum")
    _profile().save(profile_path)

    result = CliRunner().invoke(
        _command(),
        ["preview", "--input", str(source), "--profile", str(profile_path), "--approve-fresh-import"],
    )

    assert result.exit_code != 0
    assert "approve-fresh-import" in click.unstyle(result.output)


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
        [
            "generate",
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--output",
            str(output),
            "--approve-fresh-import",
        ],
    )

    expected_tags = "latin verb::irregular latinitas"
    assert previewed.exit_code == 0
    assert f"Tags: {expected_tags}" in previewed.stdout
    assert "Notes: 1" in previewed.stdout
    assert "Cards: 3" in previewed.stdout
    assert generated.exit_code == 0
    assert f"Tags: {expected_tags}" in generated.stdout
    rows = list(csv.reader(io.StringIO(output.read_text(encoding="utf-8"))))
    tag_rows = [row for row in rows if row and row[0].startswith("latinitas-v2-")]
    assert tag_rows
    assert all(row[4] == expected_tags for row in tag_rows)


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
        [
            "generate",
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--output",
            str(output),
            "--approve-fresh-import",
        ],
    )

    assert previewed.exit_code == 0
    assert f"Tags: {expected_tags}" in previewed.stdout
    assert generated.exit_code == 0
    assert f"Tags: {expected_tags}" in generated.stdout
    rows = list(csv.reader(io.StringIO(output.read_text(encoding="utf-8"))))
    tag_rows = [row for row in rows if row and row[0].startswith("latinitas-v2-")]
    assert tag_rows
    assert all(row[4] == expected_tags for row in tag_rows)


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
            "--approve-scope",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.index("Notes: 1") < result.stdout.index("Output:")
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
    _commit_scope_state(source, _profile())

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
    _commit_scope_state(
        source,
        DeckProfile.default(
            note_type="CSV source",
            lexical_entry_field="Lemma",
            principal_parts_field="Forms",
            meaning_field="Password",
            source_identity=SourceIdentityConfig(strategy="source_id_field", field="Stable ID"),
            principal_part_roles=("present_infinitive", "present_1s", "perfect_1s", "supine"),
            separators=(",",),
            selected_recipes=("principal_part_recognition",),
        ),
    )

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
    _commit_scope_state(source, _profile())

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
        approve_new_scope=True,
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
    assert "No output or committed state was changed" not in result.output
