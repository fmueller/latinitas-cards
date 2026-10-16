"""Normalized passages are cloze contexts, not raw JSON lines."""

import csv
import json
import runpy
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
from cli_test import get_command
from click.testing import CliRunner

from latinitas_cards import cli
from latinitas_cards.corpus_io import write_corpus
from latinitas_cards.corpus_model import CorpusPassage, latin_fold, passage_hash

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    records = runpy.run_path(str(ROOT / "scripts/regen-corpus-fixtures.py"))["valid_records"]("wsl")
    for index, record in enumerate(records):
        if isinstance(record, CorpusPassage):
            raw = record.model_dump()
            raw["translations"] = [
                {"language": "en", "text": f"English {record.order}", "provenance": "fixture"},
                {"language": "de", "text": f"Deutsch {record.order}", "provenance": "fixture"},
            ]
            records[index] = CorpusPassage.model_validate(raw)
    path = tmp_path / "corpus.jsonl"
    write_corpus(path, records, [])
    return path


@pytest.mark.parametrize("format", ["auto", "jsonl"])
def test_jsonl_context_and_mixed_refs(corpus: Path, tmp_path: Path, format: str) -> None:
    paths = [corpus]
    if format == "auto":
        xml = tmp_path / "other.xml"
        xml.write_text("<usfx><book id='GEN'><c id='2'/><v id='7'>Deus</v></book></usfx>")
        text = tmp_path / "plain.txt"
        text.write_text("Deus\n")
        csv = tmp_path / "parallel.csv"
        csv.write_text("la,en\nDeus,Other English\n")
        paths += [xml, text, csv]
    frame = cli.load_corpora(paths, format, "la")
    assert frame.at[1, "text_norm"] == "māter fīlium amat."
    matches = cli.collect_cloze_matches(frame, "Deus", cli.build_bucket_index(frame), 10, False, ["translation_en"])
    expected = ["poem-a.é.1"] + (["GEN 2:7", "plain 1:1", "parallel 1:1"] if format == "auto" else [])
    assert [row["ref"] for row in matches] == expected
    assert matches[0]["translation_en"] == "English 1"
    rendered = cli.generate_clozes_for_word(frame, "Deus", cli.build_bucket_index(frame), max_examples=10)
    assert all(f"({ref})" in item for ref, item in zip(expected, rendered, strict=True))


@pytest.mark.parametrize(
    ("mode", "langs", "expected"),
    [("include", [], ["en", "de"]), ("ignore", [], []), ("prompt", [], []), ("ignore", ["de"], ["de"])],
)
def test_jsonl_translation_cli(corpus: Path, tmp_path: Path, mode: str, langs: list[str], expected: list[str]) -> None:
    source = tmp_path / "forms.csv"
    source.write_text("form\nDeus\namat\n")
    output = tmp_path / "out.csv"
    argv = ["cloze", "--input", str(source), "--output", str(output), "--corpus", str(corpus), "--parallel-mode", mode]
    for lang in langs:
        argv += ["--translation-lang", lang]
    result = CliRunner().invoke(get_command(cli.app), argv)
    assert result.exit_code == 0, (result.output, result.exception)
    frame = pd.read_csv(output, keep_default_na=False)
    assert [c for c in frame.columns if c.startswith("translation_")] == [f"translation_{lang}" for lang in expected]
    for lang in expected:
        assert frame[f"translation_{lang}"].tolist() == (
            ["English 1", "English 2"] if lang == "en" else ["Deutsch 1", "Deutsch 2"]
        )
    assert "(poem-a.é.1)" in str(frame.at[0, "CorpusCloze"])
    if mode == "prompt":
        assert "non-interactively" in result.stderr


def test_explicit_txt_restores_json_lines(corpus: Path) -> None:
    frame = cli.load_corpora([corpus], "txt", "la")
    assert str(frame.at[0, "text"]).startswith('{"adapter"')
    assert frame.at[0, "book"] == "corpus"
    assert "display_ref" not in frame


def test_golden_csv_literal_contract(tmp_path: Path) -> None:
    directory = ROOT / "tests/fixtures/goldens/cloze-jsonl"
    shutil.copytree(directory / "inputs", tmp_path, dirs_exist_ok=True)
    invoke = runpy.run_path(str(ROOT / "scripts/regen-legacy-goldens.py"))["invoke"]
    invoke(
        [
            "cloze",
            "--input",
            "forms.csv",
            "--output",
            "out.csv",
            "--corpus",
            "corpus.jsonl",
            "--translation-lang",
            "en",
        ],
        tmp_path,
    )
    with (tmp_path / "out.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert (
        rows[0]["CorpusCloze"]
        == "In principio creavit {{c1::Deus}} cælum et terram. <span style='color:#888'>(poem-a.é.1)</span>"
    )
    assert rows[0]["translation_en"] == "English 1"
    assert rows[1]["translation_en"] == "English 2"
    assert "translation_de" not in rows[0]
    assert "corpus 1:" not in rows[0]["CorpusCloze"]
    assert b"jsonl" in invoke(["cloze", "--help"], tmp_path)


@pytest.mark.parametrize("renderer", ["helper", "cli"])
def test_jsonl_citation_is_text_in_html(corpus: Path, tmp_path: Path, renderer: str) -> None:
    citation = '<img src=x onerror=alert(1)> & "é"'
    records = [json.loads(line) for line in corpus.read_text().splitlines()]
    for record in records:
        if record.get("record") == "passage" and record["order"] == 1:
            record["citation"] = {
                "scheme": "custom",
                "parts": [{"unit": "label", "value": citation}],
                "display": citation,
            }
    corpus.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    frame = cli.load_corpora([corpus], "auto", "la")
    bucket = cli.build_bucket_index(frame)
    assert cli.collect_cloze_matches(frame, "Deus", bucket, 2, False, [])[0]["ref"] == citation
    escaped = "&lt;img src=x onerror=alert(1)&gt; &amp; &quot;é&quot;"
    if renderer == "helper":
        rendered = cli.generate_clozes_for_word(frame, "Deus", bucket)[0]
        assert f"({escaped})</span>" in rendered
        return
    source = tmp_path / "forms.csv"
    source.write_text("form\nDeus\n")
    output = tmp_path / "out.csv"
    result = CliRunner().invoke(
        get_command(cli.app),
        [
            "cloze",
            "--input",
            str(source),
            "--output",
            str(output),
            "--corpus",
            str(corpus),
            "--parallel-mode",
            "ignore",
        ],
    )
    assert result.exit_code == 0, result.output
    with output.open() as stream:
        row = next(csv.DictReader(stream))
    assert f"({escaped})</span>" in row["CorpusCloze"]
    assert "<img" not in row["CorpusCloze"]


@pytest.mark.parametrize("renderer", ["helper", "cli"])
@pytest.mark.parametrize("multi", [False, True])
def test_jsonl_passage_is_text_in_html(corpus: Path, tmp_path: Path, renderer: str, multi: bool) -> None:
    text = "<img src=x onerror=alert(1)> Deus & Deus"
    records = [json.loads(line) for line in corpus.read_text().splitlines()]
    for record in records:
        if record.get("record") == "passage" and record["order"] == 1:
            record.update(text=text, norm=latin_fold(text), passage_hash=passage_hash(text))
    corpus.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    frame = cli.load_corpora([corpus], "auto", "la")
    assert frame.at[0, "text"] == text
    assert frame.at[0, "text_norm"] == "<img src=x onerror=alert(1)> deus & deus"
    expected = "&lt;img src=x onerror=alert(1)&gt; {{c1::Deus}} &amp; " + ("{{c1::Deus}}" if multi else "Deus")
    if renderer == "helper":
        rendered = cli.generate_clozes_for_word(
            frame, "Deus", cli.build_bucket_index(frame), multi_cloze_per_verse=multi
        )[0]
    else:
        source = tmp_path / "forms.csv"
        source.write_text("form\nDeus\n")
        output = tmp_path / "out.csv"
        argv = ["cloze", "--input", str(source), "--output", str(output), "--corpus", str(corpus)]
        if multi:
            argv.append("--multi-cloze-per-verse")
        result = CliRunner().invoke(get_command(cli.app), argv)
        assert result.exit_code == 0, result.output
        with output.open() as stream:
            rendered = next(csv.DictReader(stream))["CorpusCloze"]
    assert rendered == expected + " <span style='color:#888'>(poem-a.é.1)</span>"


@pytest.mark.parametrize(
    "header",
    [
        {},
        {"record": "header", "format": "unknown", "version": 1},
        {"record": "header", "format": "latinitas-corpus", "version": 2},
    ],
)
def test_jsonl_gate_no_output(header: dict[str, object], tmp_path: Path) -> None:
    corpus = tmp_path / "bad.jsonl"
    corpus.write_text(json.dumps(header) + "\n")
    output = tmp_path / "out.csv"
    result = CliRunner().invoke(
        get_command(cli.app),
        [
            "cloze",
            "--input",
            str(ROOT / "tests/fixtures/goldens/legacy-vulgate/inputs/forms.csv"),
            "--corpus",
            str(corpus),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 3
    assert any(code in result.stderr for code in ("missing_header", "unsupported_format", "unsupported_version"))
    assert not output.exists()


def test_cloze_jsonl_generator_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    main = runpy.run_path(str(ROOT / "scripts/regen-corpus-fixtures.py"))["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["regen", "--target", "cloze-jsonl"])
    main()
    output = tmp_path / "tests/fixtures/goldens/cloze-jsonl/inputs/corpus.jsonl"
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        main()
    assert output.read_bytes() == original
    output.unlink()
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"protected")
    output.symlink_to(sentinel)
    monkeypatch.setattr(sys, "argv", ["regen", "--target", "cloze-jsonl", "--force"])
    with pytest.raises(SystemExit):
        main()
    assert sentinel.read_bytes() == b"protected"
