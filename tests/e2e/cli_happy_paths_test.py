"""Offline user workflows through fresh CLI processes, without application imports."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path


def invoke(work: Path, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "latinitas_cards", *args],
        cwd=work,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def rows(path: Path) -> list[list[str]]:
    csv_text = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#"))
    return list(csv.reader(io.StringIO(csv_text)))


def test_authored_validate_preview_export_and_repeat(tmp_path: Path) -> None:
    source = tmp_path / "notes ä.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(note, ensure_ascii=False)
            for note in (
                {
                    "schema_version": 1,
                    "kind": "vocab",
                    "key": "lesson:love",
                    "provenance": {"document": "Lektion ä", "section": "1"},
                    "status": "include",
                    "language_tag": "de-DE",
                    "lemma": "amō",
                    "meaning": "lieben",
                },
                {
                    "schema_version": 1,
                    "kind": "qa",
                    "key": "lesson:case",
                    "provenance": {"document": "Lektion ä", "section": "1"},
                    "status": "include",
                    "language_tag": "de-DE",
                    "question": "Was ist ein Ablativ?",
                    "answer": "Ein Kasus.",
                },
                {
                    "schema_version": 1,
                    "kind": "vocab",
                    "key": "lesson:skip",
                    "provenance": {"document": "Lektion ä", "section": "1"},
                    "status": "skip",
                    "language_tag": "de-DE",
                    "lemma": "omit-me",
                    "meaning": "skip-me",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    before = source.read_bytes()
    args = (str(source), "--namespace", "e2e-course")
    assert "Valid. Effective selection: 2" in invoke(tmp_path, "authored", "validate", *args)
    preview = invoke(tmp_path, "authored", "preview", *args)
    assert "amō" in preview and "lieben" in preview and "Ein Kasus." in preview
    output = tmp_path / "export ä"
    output.mkdir()
    export = ("authored", "export", *args, "--output-dir", str(output), "--deck", "Latin::E2E")
    invoke(tmp_path, *export)
    assert {path.name for path in output.iterdir()} == {"vocab.csv", "qa.csv"}
    vocab, qa = rows(output / "vocab.csv"), rows(output / "qa.csv")
    assert len(vocab) == len(qa) == 1
    assert "amō" in vocab[0] and "lieben" in vocab[0]
    assert "Was ist ein Ablativ?" in qa[0] and "Ein Kasus." in qa[0]
    exported = {path.name: path.read_bytes() for path in output.iterdir()}
    assert all(b"#deck:Latin::E2E" in payload for payload in exported.values())
    invoke(tmp_path, *export)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == exported
    assert source.read_bytes() == before


def test_usfx_preview_generate_and_repeat(tmp_path: Path) -> None:
    corpus = tmp_path / "sample.usfx.xml"
    corpus.write_text(
        "<usfx><book id='GEN'/><c n='1'/><v n='1'>In principio erat verbum</v><ve/></usfx>",
        encoding="utf-8",
    )
    source = tmp_path / "words.csv"
    source.write_text("Front,Back\nverbum,\n", encoding="utf-8")
    before = source.read_bytes(), corpus.read_bytes()
    args = ("--input", str(source), "--usfx", str(corpus))
    preview = invoke(tmp_path, "preview", *args)
    assert "{{c1::verbum}}" in preview
    output = tmp_path / "cards.csv"
    generate = ("generate", *args, "--output", str(output))
    invoke(tmp_path, *generate)
    data = rows(output)
    assert data == [
        ["Front", "Back", "VulgataCloze"],
        ["verbum", "", "In principio erat {{c1::verbum}} <span style='color:#888'>(GEN 1:1)</span>"],
    ]
    exported = output.read_bytes()
    invoke(tmp_path, *generate)
    assert output.read_bytes() == exported
    assert (source.read_bytes(), corpus.read_bytes()) == before
