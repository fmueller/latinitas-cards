from pathlib import Path

import pytest
from authored_import_test import row, write_rows
from typer.testing import CliRunner

from latinitas_cards.authored_import import AuthoredImportError
from latinitas_cards.authored_preview import AuthoredFilters, preview_authored_import
from latinitas_cards.cli import app


def test_merged_selection_counts_and_cards(tmp_path: Path) -> None:
    source = write_rows(
        tmp_path / "notes.jsonl",
        [
            row(tags=["first"]),
            row(
                tags=["second"],
                provenance={"document": "Vorlesung ä", "section": "  Freier Abschnitt  ", "reference": "R"},
            ),
            row("qa", status="skip"),
            row("form"),
        ],
    )
    result = preview_authored_import(
        source,
        "course",
        AuthoredFilters(kinds=("vocab",), sections=("  Freier Abschnitt  ",), references=("R",), tags=("second",)),
    )
    assert result.counts_by_kind == (("form", 1), ("qa", 1), ("vocab", 1))
    assert result.counts_by_status == (("include", 2), ("skip", 1))
    assert result.counts_by_section == (("  Freier Abschnitt  ", 3),)
    assert dict(result.counts_by_reference) == {None: 2, "R": 1}
    assert result.merged_duplicates == ((1, 2),)
    assert len(result.require_valid_selection()) == 1
    assert len(result.cards) == 1
    assert result.cards[0].front == '<div class="lemma">amō</div>\n'
    assert '<div class="meaning">lieben</div>' in result.cards[0].back
    assert "{{" not in result.cards[0].back


@pytest.mark.parametrize("bad", [row("qa", status="skip", answer=""), row("form", analysis=""), "not json"])
def test_errors_before_filters_withhold_selection(tmp_path: Path, bad: dict[str, object] | str) -> None:
    source = write_rows(
        tmp_path / "notes.jsonl", [row(), bad, row(key="conflict", meaning="a"), row(key="conflict", meaning="b")]
    )
    result = preview_authored_import(source, "course", AuthoredFilters(kinds=("vocab",), tags=("absent",)))
    assert len(result.errors) >= 2
    assert result.selected_notes == ()
    with pytest.raises(AuthoredImportError):
        result.require_valid_selection()
    visible = preview_authored_import(source, "course", AuthoredFilters(kinds=("vocab",)))
    assert visible.cards and not visible.selected_notes
    assert visible.diagnostic_notes


def test_cli_filters_read_only_empty_and_diagnostics(tmp_path: Path) -> None:
    source = write_rows(
        tmp_path / "notes.jsonl",
        [
            row(tags=["a"]),
            row(
                tags=["b"], provenance={"document": "Vorlesung ä", "section": "  Freier Abschnitt  ", "reference": "R"}
            ),
            row("qa", status="skip"),
        ],
    )
    before = source.read_bytes()
    files = set(tmp_path.iterdir())
    runner = CliRunner()
    args = ["authored", "preview", str(source), "--namespace", "course"]
    selected = runner.invoke(
        app, [*args, "--kind", "vocab", "--section", "  Freier Abschnitt  ", "--reference", "R", "--tag", "b"]
    )
    assert selected.exit_code == 0, selected.output
    assert "Effective selection: 1" in selected.output and "Merged duplicates: 1" in selected.output
    assert "amō" in selected.output and "lieben" in selected.output
    empty = runner.invoke(app, [*args, "--kind", "qa"])
    assert empty.exit_code == 0 and "Empty selection" in empty.output
    valid = runner.invoke(app, ["authored", "validate", str(source), "--namespace", "course"])
    assert valid.exit_code == 0 and "Valid" in valid.output
    assert source.read_bytes() == before and set(tmp_path.iterdir()) == files
    assert runner.invoke(app, ["authored", "preview", str(source)]).exit_code != 0


def test_cli_aggregates_invalid_skipped_and_filtered_rows(tmp_path: Path) -> None:
    source = write_rows(
        tmp_path / "notes.jsonl", [row(), row("qa", status="skip", answer=""), row("form", analysis="")]
    )
    before = source.read_bytes()
    files = set(tmp_path.iterdir())
    for command in ("validate", "preview"):
        args = ["authored", command, str(source), "--namespace", "course"]
        if command == "preview":
            args += ["--kind", "vocab"]
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 1, result.output
        assert "line 2" in result.output and "line 3" in result.output
        assert "not exportable" in result.output
        if command == "preview":
            assert "Diagnostic cards" in result.output and "amō" in result.output
        assert source.read_bytes() == before and set(tmp_path.iterdir()) == files


def test_cli_controls_are_inert_but_typed_content_is_preserved(tmp_path: Path) -> None:
    control = "\x1b]52;c;Y2xpcGJvYXJk\x07"
    source = write_rows(tmp_path / "notes.jsonl", [row(meaning=f"before{control}after\rspoof\u202etext")])
    preview = preview_authored_import(source, "course")
    assert control in preview.cards[0].back
    result = CliRunner().invoke(app, ["authored", "preview", str(source), "--namespace", "course"], color=True)
    assert result.exit_code == 0
    assert "\x1b" not in result.output and "\x07" not in result.output and "\u202e" not in result.output
    assert r"\x1b]52;c;Y2xpcGJvYXJk\x07" in result.output
    assert r"\x0dspoof\u202etext" in result.output


@pytest.mark.parametrize("command", ["validate", "preview", "export"])
def test_cli_escapes_issue_fields_and_recovers_without_writes(tmp_path: Path, command: str) -> None:
    field = "bad\x1b]0;OWNED\x07\r\u202e"
    provenance = {"document": "D", "section": "S\x1b\x07", "reference": "R\r"}
    source = write_rows(tmp_path / "notes.jsonl", [row(**{field: "extra"}), row(provenance=provenance)])
    existing = tmp_path / "authored-vocab.csv"
    existing.write_bytes(b"existing output\x00\xff")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    structured = preview_authored_import(source, "course")
    assert field in structured.errors[0].field
    assert structured.diagnostic_notes[0].item.provenance.section == provenance["section"]
    args = ["authored", command, str(source), "--namespace", "course"]
    if command == "export":
        args += ["--output-dir", str(tmp_path), "--deck", "Latin"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    assert "\x1b" not in result.output and "\x07" not in result.output
    assert "\r" not in result.output and "\u202e" not in result.output
    assert r"bad\x1b]0;OWNED\x07\x0d\u202e" in result.output
    assert "line 1" in result.output and "Extra inputs are not permitted" in result.output
    assert "diagnostic matches: 1; not exportable" in result.output
    if command == "preview":
        assert "Front:\n" in result.output and "Back:\n" in result.output and "amō" in result.output
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    ("command", "boundary"),
    [("validate", "input"), ("preview", "input"), ("export", "input"), ("export", "export")],
)
def test_cli_escapes_caught_error_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, boundary: str
) -> None:
    from latinitas_cards.commands import authored

    source = write_rows(tmp_path / "notes.jsonl", [row()])

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("unsafe\x1b]0;OWNED\x07\r\n\t\u202e")

    monkeypatch.setattr(authored, "preview_authored_import" if boundary == "input" else "write_authored_csv", fail)
    args = ["authored", command, str(source), "--namespace", "course"]
    if command == "export":
        args += ["--output-dir", str(tmp_path), "--deck", "Latin"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "\x1b" not in result.output and "\x07" not in result.output
    assert f"Authored {boundary} error: " in result.output
    assert r"unsafe\x1b]0;OWNED\x07\x0d\x0a\x09\u202e" in result.output


@pytest.mark.parametrize("dimension", ["kind", "section", "reference", "tag"])
def test_each_combined_cli_filter_excludes_its_own_decoy(tmp_path: Path, dimension: str) -> None:
    provenance = {"document": "D", "section": "S", "reference": "R"}
    decoy = row("qa" if dimension == "kind" else "vocab", key="decoy", provenance=provenance, tags=["T"])
    if dimension in {"section", "reference"}:
        decoy["provenance"] = {**provenance, dimension: "other"}
    elif dimension == "tag":
        decoy["tags"] = ["other"]
    source = write_rows(tmp_path / "notes.jsonl", [row(key="target", provenance=provenance, tags=["T"]), decoy])
    result = CliRunner().invoke(
        app,
        [
            "authored",
            "preview",
            str(source),
            "--namespace",
            "course",
            "--kind",
            "vocab",
            "--section",
            "S",
            "--reference",
            "R",
            "--tag",
            "T",
        ],
    )
    assert result.exit_code == 0
    assert "Effective selection: 1" in result.output
    assert "'target'" in result.output and "'decoy'" not in result.output


def test_repeated_filters_are_or_and_merged_skip_never_selected(tmp_path: Path) -> None:
    source = write_rows(
        tmp_path / "notes.jsonl",
        [row(tags=["a"]), row(status="skip", tags=["b"]), row("qa", tags=["b"]), row("form", tags=["c"])],
    )
    result = preview_authored_import(source, "course", AuthoredFilters(kinds=("qa", "form"), tags=("b", "c")))
    assert [note.item.kind for note in result.require_valid_selection()] == ["form", "qa"]
    all_notes = preview_authored_import(source, "course")
    assert dict(all_notes.counts_by_status) == {"include": 2, "skip": 1}
    assert all(note.item.kind != "vocab" for note in all_notes.selected_notes)
    assert [card.kind for card in result.cards] == ["form", "qa"]
    assert "amāvērunt" in result.cards[0].front and "sie liebten" in result.cards[0].back
    assert "Was ist ein Ablativ?" in result.cards[1].front and "Ein Kasus." in result.cards[1].back
