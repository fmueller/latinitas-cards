import json
import sqlite3
import zipfile
from pathlib import Path

from click.testing import CliRunner
from typer.main import get_command

from latinitas_cards.cli import _read_apkg_field_rows, app, strip_anki_field


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
