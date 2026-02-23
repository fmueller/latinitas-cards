import json
import sqlite3
import zipfile
from pathlib import Path

import click
import pandas as pd
import pytest
from click.testing import CliRunner
from typer.main import get_command

import latinitas_cards.cli as cli_mod
from latinitas_cards.cli import (
    _extract_llm_choice,
    _read_apkg_field_rows,
    _select_analysis_candidate,
    app,
    clean_anki_field,
    split_latin_forms,
    strip_anki_field,
)


def _create_anki_db(db_path: Path, notes: list[list[str]], field_names: list[str] | None = None) -> None:
    """Create a minimal Anki SQLite database with the given notes."""
    if field_names is None:
        field_names = ["Front", "Back", "VulgataCloze"]
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, mid INTEGER, flds TEXT)")
    con.execute("CREATE TABLE col (id INTEGER PRIMARY KEY, models TEXT)")
    model_id = 1234567890
    model = {
        str(model_id): {
            "flds": [{"name": n} for n in field_names],
        }
    }
    con.execute("INSERT INTO col (id, models) VALUES (1, ?)", (json.dumps(model),))
    for i, fields in enumerate(notes):
        con.execute(
            "INSERT INTO notes (id, mid, flds) VALUES (?, ?, ?)",
            (i, model_id, "\x1f".join(fields)),
        )
    con.commit()
    con.close()


def _create_colpkg_with_anki2(
    colpkg_path: Path,
    notes: list[list[str]],
    field_names: list[str] | None = None,
) -> None:
    """Create a .colpkg zip containing only collection.anki2."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "collection.anki2"
        _create_anki_db(db_path, notes, field_names=field_names)
        with zipfile.ZipFile(colpkg_path, "w") as zf:
            zf.write(db_path, "collection.anki2")


def _create_colpkg_with_anki21b(
    colpkg_path: Path,
    real_notes: list[list[str]],
    dummy_notes: list[list[str]] | None = None,
    field_names: list[str] | None = None,
) -> None:
    """Create a .colpkg zip with a zstd-compressed collection.anki21b and a dummy collection.anki2."""
    import tempfile

    import zstandard

    with tempfile.TemporaryDirectory() as td:
        # Build the real DB for anki21b
        real_db = Path(td) / "real.db"
        _create_anki_db(real_db, real_notes, field_names=field_names)
        compressed = zstandard.ZstdCompressor().compress(real_db.read_bytes())

        # Build a dummy anki2 (placeholder)
        dummy_db = Path(td) / "collection.anki2"
        _create_anki_db(dummy_db, dummy_notes or [["placeholder", "", ""]], field_names=field_names)

        with zipfile.ZipFile(colpkg_path, "w") as zf:
            zf.write(dummy_db, "collection.anki2")
            zf.writestr("collection.anki21b", compressed)


def _create_modern_anki_db(db_path: Path, notes: list[list[str]], field_names: list[str] | None = None) -> None:
    """Create a modern Anki-like DB with fields/notetypes tables."""
    if field_names is None:
        field_names = ["Latein", "Deutsch", "Konstruktion_Hinweise"]
    con = sqlite3.connect(str(db_path))
    con.execute(
        "CREATE TABLE notes (id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER, mod INTEGER, usn INTEGER, "
        "tags TEXT, flds TEXT, sfld INTEGER, csum INTEGER, flags INTEGER, data TEXT)"
    )
    con.execute(
        "CREATE TABLE cards (id INTEGER PRIMARY KEY, nid INTEGER, did INTEGER, ord INTEGER, mod INTEGER, usn INTEGER, "
        "type INTEGER, queue INTEGER, due INTEGER, ivl INTEGER, factor INTEGER, reps INTEGER, lapses INTEGER, "
        "left INTEGER, odue INTEGER, odid INTEGER, flags INTEGER, data TEXT)"
    )
    con.execute("CREATE TABLE col (id INTEGER PRIMARY KEY, models TEXT)")
    con.execute(
        "CREATE TABLE notetypes (id INTEGER PRIMARY KEY, name TEXT, mtime_secs INTEGER, usn INTEGER, config BLOB)"
    )
    con.execute("CREATE TABLE fields (ntid INTEGER, ord INTEGER, name TEXT, config BLOB)")

    model_id = 1502189247895
    con.execute(
        "INSERT INTO notetypes (id, name, mtime_secs, usn, config) VALUES (?, ?, 0, 0, ?)",
        (model_id, "Latein", b"{}"),
    )
    for i, field_name in enumerate(field_names):
        con.execute(
            "INSERT INTO fields (ntid, ord, name, config) VALUES (?, ?, ?, ?)",
            (model_id, i, field_name, b"{}"),
        )
    con.execute("INSERT INTO col (id, models) VALUES (1, '')")

    for i, note_fields in enumerate(notes, start=1):
        note_id = 1000 + i
        con.execute(
            (
                "INSERT INTO notes (id, guid, mid, mod, usn, tags, flds, sfld, csum, flags, data) "
                "VALUES (?, ?, ?, 0, 0, '', ?, 0, 0, 0, '')"
            ),
            (note_id, f"guid-{i}", model_id, "\x1f".join(note_fields)),
        )
        con.execute(
            (
                "INSERT INTO cards (id, nid, did, ord, mod, usn, type, queue, due, ivl, factor, reps, lapses, "
                "left, odue, odid, flags, data) VALUES (?, ?, 1, 0, 0, 0, 0, 0, ?, 0, 0, 0, 0, 0, 0, 0, 0, '')"
            ),
            (2000 + i, note_id, i),
        )
    con.commit()
    con.close()


def _create_modern_colpkg(colpkg_path: Path, notes: list[list[str]], field_names: list[str] | None = None) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "collection.anki2"
        _create_modern_anki_db(db_path, notes, field_names=field_names)
        with zipfile.ZipFile(colpkg_path, "w") as zf:
            zf.write(db_path, "collection.anki2")


class TestStripAnkiField:
    def test_plain_word_unchanged(self) -> None:
        assert strip_anki_field("dominus") == "dominus"

    def test_strips_macrons(self) -> None:
        assert strip_anki_field("īnsula") == "insula"

    def test_strips_breves(self) -> None:
        assert strip_anki_field("ĭnsula") == "insula"

    def test_strips_multiple_macrons(self) -> None:
        assert strip_anki_field("labōrāre") == "laborare"

    def test_drops_block_level_html(self) -> None:
        assert strip_anki_field("esse<div>(est - sunt)</div>") == "esse"

    def test_drops_block_html_with_entities(self) -> None:
        assert strip_anki_field("esse<div>(est -&nbsp;sunt)</div>") == "esse"

    def test_strips_inline_html(self) -> None:
        assert strip_anki_field("<b>dominus</b>") == "dominus"

    def test_empty_string(self) -> None:
        assert strip_anki_field("") == ""

    def test_combined_html_and_macrons(self) -> None:
        assert strip_anki_field("vocāre<div>(hint)</div>") == "vocare"


class TestReadApkgFieldRows:
    def test_reads_anki2_colpkg(self, tmp_path: Path) -> None:
        colpkg = tmp_path / "deck.colpkg"
        _create_colpkg_with_anki2(colpkg, [["dominus", "lord", ""], ["servus", "slave", ""]])
        rows = _read_apkg_field_rows(colpkg, "Front")
        assert len(rows) == 2
        assert rows[0]["Front"] == "dominus"
        assert rows[1]["Front"] == "servus"

    def test_prefers_anki21b_over_anki2(self, tmp_path: Path) -> None:
        colpkg = tmp_path / "deck.colpkg"
        _create_colpkg_with_anki21b(
            colpkg,
            real_notes=[["insula", "island", ""], ["terra", "land", ""], ["aqua", "water", ""]],
            dummy_notes=[["placeholder", "", ""]],
        )
        rows = _read_apkg_field_rows(colpkg, "Front")
        assert len(rows) == 3
        assert rows[0]["Front"] == "insula"

    def test_reads_apkg_suffix(self, tmp_path: Path) -> None:
        apkg = tmp_path / "deck.apkg"
        _create_colpkg_with_anki2(apkg, [["verbum", "word", ""]])
        rows = _read_apkg_field_rows(apkg, "Front")
        assert len(rows) == 1
        assert rows[0]["Front"] == "verbum"

    def test_load_apkg_suffix_in_preview(self, tmp_path: Path) -> None:
        """The .apkg suffix should be recognized by _load_input_to_dataframe."""
        usfx_path = tmp_path / "sample.usfx.xml"
        usfx_path.write_text(
            "<usfx><book id='GEN'/><c n='1'/><v n='1'>In principio erat verbum</v><ve/></usfx>",
            encoding="utf-8",
        )
        apkg = tmp_path / "deck.apkg"
        _create_colpkg_with_anki2(apkg, [["verbum", "word", ""]])

        runner = CliRunner()
        result = runner.invoke(
            get_command(app),
            ["preview", "--input", str(apkg), "--usfx", str(usfx_path), "--limit", "1"],
        )
        assert result.exit_code == 0
        assert "{{c1::verbum}}" in result.output


def test_preview_matches_macron_words_against_vulgate(tmp_path: Path) -> None:
    """A word with macrons (e.g. 'īnsula') should match plain Vulgate text ('insula')."""
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>Magnus est insula in mari</v><ve/></usfx>",
        encoding="utf-8",
    )

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("Front,Back\n\u012bnsula,island\n", encoding="utf-8")  # īnsula

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        ["preview", "--input", str(csv_path), "--usfx", str(usfx_path), "--limit", "1"],
    )

    assert result.exit_code == 0
    assert "{{c1::insula}}" in result.output


def test_preview_matches_html_field_against_vulgate(tmp_path: Path) -> None:
    """A word wrapped in HTML (e.g. 'esse<div>(est)</div>') should match Vulgate text ('esse')."""
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>Bonum est esse in terra</v><ve/></usfx>",
        encoding="utf-8",
    )

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("Front,Back\nesse<div>(est - sunt)</div>,to be\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        ["preview", "--input", str(csv_path), "--usfx", str(usfx_path), "--limit", "1"],
    )

    assert result.exit_code == 0
    assert "{{c1::esse}}" in result.output


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


def test_cli_short_help_option() -> None:
    runner = CliRunner()
    result = runner.invoke(get_command(app), ["-h"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_command_callbacks_are_split_into_command_modules() -> None:
    click_app = get_command(app)
    assert isinstance(click_app, click.Group)
    command_modules = {name: command.callback.__module__ for name, command in click_app.commands.items()}

    assert command_modules
    assert all(module.startswith("latinitas_cards.commands.") for module in command_modules.values())


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


def test_preview_multi_cloze_per_verse_marks_all_occurrences(tmp_path: Path) -> None:
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>verbum et verbum in principio</v><ve/></usfx>",
        encoding="utf-8",
    )

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("Front,Back\nverbum,\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "preview",
            "--input",
            str(csv_path),
            "--usfx",
            str(usfx_path),
            "--multi-cloze-per-verse",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert result.output.count("{{c1::verbum}}") == 2


def test_preview_expands_word_forms_from_mapping_file(tmp_path: Path) -> None:
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>dixit autem dominus</v><ve/></usfx>",
        encoding="utf-8",
    )

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("Front,Back\ndice,\n", encoding="utf-8")

    forms_path = tmp_path / "forms.txt"
    forms_path.write_text("dice,dico,dicis,dixit\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "preview",
            "--input",
            str(csv_path),
            "--usfx",
            str(usfx_path),
            "--word-forms",
            str(forms_path),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "{{c1::dixit}}" in result.output


def test_preview_supports_lemma_forms_file(tmp_path: Path) -> None:
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>Puer amat sapientiam</v><ve/></usfx>",
        encoding="utf-8",
    )

    apkg_path = tmp_path / "input.apkg"
    _create_colpkg_with_anki2(apkg_path, [["amo", "love", ""], ["terra", "land", ""]])

    lemmas_path = tmp_path / "lemmas.txt"
    lemmas_path.write_text("amo, amas, amat\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "preview",
            "--input",
            str(apkg_path),
            "--usfx",
            str(usfx_path),
            "--lemmas",
            str(lemmas_path),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "{{c1::amat}}" in result.output


def test_preview_supports_ignore_pattern(tmp_path: Path) -> None:
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>Et verbum erat apud Deum</v><ve/></usfx>",
        encoding="utf-8",
    )

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("Front,Back\net,and\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "preview",
            "--input",
            str(csv_path),
            "--usfx",
            str(usfx_path),
            "--ignore-pattern",
            "^et$",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "No clozes generated" in result.output


def test_generate_writes_back_to_apkg(tmp_path: Path) -> None:
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>In principio erat verbum</v><ve/></usfx>",
        encoding="utf-8",
    )

    apkg_path = tmp_path / "input.apkg"
    _create_colpkg_with_anki2(apkg_path, [["verbum", "word", ""]])
    output_path = tmp_path / "output.apkg"

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "generate",
            "--input",
            str(apkg_path),
            "--usfx",
            str(usfx_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    rows = _read_apkg_field_rows(output_path, "VulgataCloze")
    assert len(rows) == 1
    assert "{{c1::verbum}}" in rows[0]["VulgataCloze"]


def test_generate_apkg_requires_existing_output_field(tmp_path: Path) -> None:
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>In principio erat verbum</v><ve/></usfx>",
        encoding="utf-8",
    )

    apkg_path = tmp_path / "input.apkg"
    _create_colpkg_with_anki2(
        apkg_path,
        [["verbum", "word"]],
        field_names=["Front", "Back"],
    )
    output_path = tmp_path / "output.apkg"

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "generate",
            "--input",
            str(apkg_path),
            "--usfx",
            str(usfx_path),
            "--output",
            str(output_path),
            "--new-field",
            "VulgataCloze",
        ],
    )

    assert result.exit_code != 0
    assert result.exception is not None
    assert "Field 'VulgataCloze' not found" in str(result.exception)


def test_validate_passes_for_valid_usfx_and_csv(tmp_path: Path) -> None:
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>In principio erat verbum</v><ve/></usfx>",
        encoding="utf-8",
    )

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("Front,Back\nverbum,word\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        ["validate", "--input", str(csv_path), "--usfx", str(usfx_path)],
    )

    assert result.exit_code == 0
    assert "Validation Report" in result.output
    assert "PASS" in result.output


def test_validate_fails_when_front_column_missing(tmp_path: Path) -> None:
    usfx_path = tmp_path / "sample.usfx.xml"
    usfx_path.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>In principio erat verbum</v><ve/></usfx>",
        encoding="utf-8",
    )

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("Word,Back\nverbum,word\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        ["validate", "--input", str(csv_path), "--usfx", str(usfx_path)],
    )

    assert result.exit_code == 1
    assert "Validation Report" in result.output
    assert "Missing required column 'Front'" in result.output


def test_validate_fails_for_invalid_usfx_structure(tmp_path: Path) -> None:
    usfx_path = tmp_path / "broken.usfx.xml"
    usfx_path.write_text("<usfx><book id='GEN'/><c n='1'/></usfx>", encoding="utf-8")

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("Front,Back\nverbum,word\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        ["validate", "--input", str(csv_path), "--usfx", str(usfx_path)],
    )

    assert result.exit_code == 1
    assert "Validation Report" in result.output
    assert "Could not parse USFX structure" in result.output


def test_read_apkg_field_rows_supports_modern_schema(tmp_path: Path) -> None:
    apkg = tmp_path / "modern.apkg"
    _create_modern_colpkg(
        apkg,
        notes=[
            ["amo", "lieben", "amo, amas, amat"],
            ["video", "sehen", "video, vides, videt"],
        ],
    )
    rows = _read_apkg_field_rows(apkg, "Latein")
    assert [row["Latein"] for row in rows] == ["amo", "video"]


def test_clean_anki_field_preserves_block_content_when_requested() -> None:
    assert clean_anki_field("esse<div>(est - sunt)</div>", truncate_at_block=False) == "esse | (est - sunt)"


def test_split_latin_forms_auto_detects_commas() -> None:
    forms, rule, confidence = split_latin_forms("amo, amas, amat", mode="auto")
    assert forms == ["amo", "amas", "amat"]
    assert rule == "comma"
    assert confidence > 0.9


def test_inspect_lists_notetype_and_fields(tmp_path: Path) -> None:
    apkg = tmp_path / "modern.apkg"
    _create_modern_colpkg(apkg, notes=[["amo", "lieben", "amo, amas, amat"]])
    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "inspect",
            "--input",
            str(apkg),
            "--head",
            "1",
            "--fields",
            "Latein",
            "--fields",
            "Konstruktion_Hinweise",
        ],
    )
    assert result.exit_code == 0
    assert "Anki Note Types" in result.output
    assert "Konstruktion_Hinweise" in result.output


def test_split_command_writes_csv_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text('Latein,Konstruktion_Hinweise\namo,"amo, amas, amat"\n', encoding="utf-8")
    out_path = tmp_path / "split.csv"
    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "split",
            "--input",
            str(csv_path),
            "--output",
            str(out_path),
            "--source-field",
            "Konstruktion_Hinweise",
            "--split-mode",
            "comma",
        ],
    )
    assert result.exit_code == 0
    out_text = out_path.read_text(encoding="utf-8")
    assert "form" in out_text
    assert "amo" in out_text
    assert "amas" in out_text


def test_split_overwrites_source_field_by_default(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text('Latein,Konstruktion_Hinweise\namo,"amo, amas, amat"\n', encoding="utf-8")
    out_path = tmp_path / "split.csv"
    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "split",
            "--input",
            str(csv_path),
            "--output",
            str(out_path),
            "--source-field",
            "Konstruktion_Hinweise",
            "--split-mode",
            "comma",
        ],
    )
    assert result.exit_code == 0
    import csv

    with open(out_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert row["Konstruktion_Hinweise"] == row["form"]


def test_split_with_front_field_overwrites_front_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text('Latein,Konstruktion_Hinweise\namo,"amo, amas, amat"\n', encoding="utf-8")
    out_path = tmp_path / "split.csv"
    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "split",
            "--input",
            str(csv_path),
            "--output",
            str(out_path),
            "--source-field",
            "Konstruktion_Hinweise",
            "--front-field",
            "Latein",
            "--split-mode",
            "comma",
        ],
    )
    assert result.exit_code == 0
    import csv

    with open(out_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert row["Latein"] == row["form"]


def test_split_keep_original_source(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text('Latein,Konstruktion_Hinweise\namo,"amo, amas, amat"\n', encoding="utf-8")
    out_path = tmp_path / "split.csv"
    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "split",
            "--input",
            str(csv_path),
            "--output",
            str(out_path),
            "--source-field",
            "Konstruktion_Hinweise",
            "--keep-original-source",
            "--split-mode",
            "comma",
        ],
    )
    assert result.exit_code == 0
    import csv

    with open(out_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert row["Konstruktion_Hinweise"] == "amo, amas, amat"


def test_split_command_can_rewrite_apkg(tmp_path: Path) -> None:
    apkg = tmp_path / "modern.apkg"
    _create_modern_colpkg(
        apkg,
        notes=[
            ["amo", "lieben", "amo, amas, amat"],
        ],
    )
    out_apkg = tmp_path / "rewritten.apkg"
    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "split",
            "--input",
            str(apkg),
            "--output",
            str(out_apkg),
            "--source-field",
            "Konstruktion_Hinweise",
            "--split-mode",
            "comma",
            "--output-format",
            "apkg",
        ],
    )
    assert result.exit_code == 0
    rows = _read_apkg_field_rows(out_apkg, "Konstruktion_Hinweise")
    assert len(rows) >= 3


def test_annotate_command_uses_annotation_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "split.csv"
    csv_path.write_text("form\namo\n", encoding="utf-8")
    out_path = tmp_path / "annotated.csv"

    def fake_annotate(df: pd.DataFrame, form_column: str, **_: object) -> pd.DataFrame:
        out: pd.DataFrame = df.copy()
        out["lemma"] = ["amo"]
        out["upos"] = ["VERB"]
        out["xpos"] = [""]
        out["morph_features"] = ["Mood=Ind"]
        out["analysis_count"] = [1]
        out["analysis_status"] = ["ok"]
        return out

    monkeypatch.setattr(cli_mod, "annotate_with_cltk", fake_annotate)
    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        ["annotate", "--input", str(csv_path), "--output", str(out_path), "--form-column", "form"],
    )
    assert result.exit_code == 0
    content = out_path.read_text(encoding="utf-8")
    assert "lemma" in content
    assert "VERB" in content


def test_annotate_command_accepts_ollama_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "split.csv"
    csv_path.write_text("form\namo\n", encoding="utf-8")
    out_path = tmp_path / "annotated.csv"

    captured: dict[str, object] = {}

    def fake_annotate(df: pd.DataFrame, form_column: str, **kwargs: object) -> pd.DataFrame:
        captured["kwargs"] = kwargs
        out: pd.DataFrame = df.copy()
        out["lemma"] = ["amo"]
        out["upos"] = ["VERB"]
        out["xpos"] = [""]
        out["morph_features"] = ["Mood=Ind"]
        out["analysis_count"] = [1]
        out["analysis_status"] = ["ok"]
        return out

    monkeypatch.setattr(cli_mod, "annotate_with_cltk", fake_annotate)
    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "annotate",
            "--input",
            str(csv_path),
            "--output",
            str(out_path),
            "--use-llm",
        ],
    )
    assert result.exit_code == 0
    assert "provider=ollama" in result.output
    assert "model=ministral-3:8b" in result.output
    assert isinstance(captured.get("kwargs"), dict)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("use_llm") is True
    assert kwargs.get("llm_provider") == "ollama"
    assert kwargs.get("llm_model") == "ministral-3:8b"


def test_extract_llm_choice_supports_json_and_plain_number() -> None:
    assert _extract_llm_choice('{"choice": 2}', max_choice=3) == 1
    assert _extract_llm_choice("I choose 1", max_choice=2) == 0


def test_select_analysis_candidate_falls_back_when_llm_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [
        {"lemma": "dominus", "upos": "NOUN", "xpos": "", "morph_features": "", "source": "cltk"},
        {"lemma": "domina", "upos": "NOUN", "xpos": "", "morph_features": "", "source": "surface"},
    ]

    def fake_query(*_: object, **__: object) -> int:
        raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(cli_mod, "_query_ollama_choice", fake_query)
    selected, status = _select_analysis_candidate(
        form="domina",
        candidates=candidates,
        use_llm=True,
        llm_provider="ollama",
        llm_model="ministral-3:8b",
        llm_endpoint="http://localhost:11434",
    )
    assert status == "ok-llm-fallback"
    assert selected["lemma"] == "dominus"


def test_select_analysis_candidate_uses_llm_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [
        {"lemma": "dominus", "upos": "NOUN", "xpos": "", "morph_features": "", "source": "cltk"},
        {"lemma": "domina", "upos": "NOUN", "xpos": "", "morph_features": "", "source": "surface"},
    ]

    def fake_query(*_: object, **__: object) -> int:
        return 1

    monkeypatch.setattr(cli_mod, "_query_ollama_choice", fake_query)
    selected, status = _select_analysis_candidate(
        form="domina",
        candidates=candidates,
        use_llm=True,
        llm_provider="ollama",
        llm_model="ministral-3:8b",
        llm_endpoint="http://localhost:11434",
    )
    assert status == "ok-llm"
    assert selected["lemma"] == "domina"


def test_cloze_command_generates_corpus_cloze(tmp_path: Path) -> None:
    input_csv = tmp_path / "forms.csv"
    input_csv.write_text("form\nverbum\n", encoding="utf-8")
    corpus_txt = tmp_path / "corpus.txt"
    corpus_txt.write_text("In principio erat verbum.\n", encoding="utf-8")
    output_csv = tmp_path / "cloze.csv"

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "cloze",
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--corpus",
            str(corpus_txt),
            "--corpus-format",
            "txt",
        ],
    )
    assert result.exit_code == 0
    text = output_csv.read_text(encoding="utf-8")
    assert "CorpusCloze" in text
    assert "{{c1::verbum}}" in text


def test_cloze_non_interactive_ignores_parallel_columns_by_default(tmp_path: Path) -> None:
    input_csv = tmp_path / "forms.csv"
    input_csv.write_text("form\namo\n", encoding="utf-8")
    corpus_csv = tmp_path / "parallel.csv"
    corpus_csv.write_text("la,en,de\namo,loving,lieben\n", encoding="utf-8")
    output_csv = tmp_path / "cloze.csv"

    runner = CliRunner()
    result = runner.invoke(
        get_command(app),
        [
            "cloze",
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--corpus",
            str(corpus_csv),
            "--corpus-format",
            "csv",
        ],
    )
    assert result.exit_code == 0
    text = output_csv.read_text(encoding="utf-8")
    assert "translation_en" not in text
