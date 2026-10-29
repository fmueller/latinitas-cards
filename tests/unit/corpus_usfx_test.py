"""Strict USFX normalization through the public CLI, independent of legacy parsing."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from latinitas_cards.cli import app

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/corpus/usfx"


def normalize(source: Path, output: Path, *flags: str) -> Result:
    before = source.read_bytes()
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "normalize",
            str(source),
            "--adapter",
            "usfx",
            "--corpus-id",
            "vulgata-clementina",
            "--output",
            str(output),
            "--json",
            *flags,
        ],
    )
    assert source.read_bytes() == before
    return result


def test_structural_order_identity_and_character_data(tmp_path: Path) -> None:
    source, output = FIXTURES / "structural.usfx.xml", tmp_path / "out.jsonl"
    result = normalize(source, output)
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_bytes().splitlines()]
    assert [(r["work"], r["title"], r["order"]) for r in rows if r["record"] == "work"] == [
        ("ZZZ", "Last first", 1),
        ("é", "Other", 2),
    ]
    passages = [r for r in rows if r["record"] == "passage"]
    assert [r["native_ref"] for r in passages] == ["ZZZ.1%2E2.3a", "ZZZ.1%2E2.%2E", "ZZZ.1%2E2.%252E", "%C3%A9.2.1"]
    assert [r["order"] for r in passages] == [1, 2, 3, 4]
    assert passages[0]["text"] == passages[-1]["text"] == "Māter amat."
    assert passages[0]["passage_hash"] == passages[-1]["passage_hash"]
    assert passages[0]["citation"] == {
        "scheme": "usfx-bcv",
        "display": "ZZZ 1.2:3a",
        "parts": [
            {"unit": "book", "value": "ZZZ"},
            {"unit": "chapter", "value": "1.2"},
            {"unit": "verse", "value": "3a"},
        ],
    }
    assert all(rows[0][key] is None for key in ("edition", "publisher", "url", "license"))
    assert CliRunner().invoke(app, ["corpus", "validate", str(output), "--json"]).exit_code == 0
    assert not output.with_suffix(".manifest.json").exists()


@pytest.mark.parametrize(
    "body,code",
    [
        ('<book id="GEN"><c id="1"/><v id="1"/>a<ve/><v id="1">b</v></book>', "duplicate_source_ref"),
        ('<book id="GEN"><c id="1"/><v id="1"/> <ve/></book>', "empty_passage"),
        ('<book id="GEN"><c id="1"/><v id="1"></v></book>', "empty_passage"),
        ('<book id="GEN"><c id="1"/><v id="1"/>a</book>', "invalid_structure"),
        ('<book id="GEN"/><c id="1"/><v id="1"/>a<v id="2"/>b<ve/>', "invalid_structure"),
        ('<book id="GEN"/><c id="1"/><v id="1"/>a<c id="2"/>', "invalid_structure"),
        ('<book id="GEN"/><c id="1"/><v id="1"/>a<book id="EXO"/>', "invalid_structure"),
        ('<book id="GEN"/><c id="1"/><v id="1"/>a', "invalid_structure"),
        ('<v id="1">a</v>', "invalid_structure"),
        ('<book id="GEN"><v id="1">a</v></book>', "invalid_structure"),
        ('<book id="GEN"><c id="1"/>outside<v id="1">a</v></book>', "invalid_structure"),
        ('<book id="GEN"><c id="1"/><v id="1">a<v id="2">b</v></v></book>', "invalid_structure"),
        ('<book id="GEN"><c id="1"/><v id="1">a</v></book><v id="2">b</v>', "invalid_structure"),
        ('<book id="GEN"><c id="1"/><v id="1">a</v></book><book id="EXO"><v id="2">b</v></book>', "invalid_structure"),
        ('<book id="GEN"><c id="1"/><v id="1">a</v>', "invalid_xml"),
    ],
)
def test_invalid_sources_do_not_replace_output(tmp_path: Path, body: str, code: str) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out.jsonl"
    source.write_text(f"<usfx>{body}</usfx>")
    output.write_bytes(b"previous output\n")
    result = normalize(source, output)
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == code
    assert output.read_bytes() == b"previous output\n"


@pytest.mark.parametrize("fixture,code", [("duplicate", "duplicate_source_ref"), ("outside-text", "invalid_structure")])
def test_rejected_fixtures(tmp_path: Path, fixture: str, code: str) -> None:
    output = tmp_path / "out"
    result = normalize(FIXTURES / f"{fixture}.usfx.xml", output)
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == code
    assert not output.exists()


def test_unknown_counts_and_doctype(tmp_path: Path) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out"
    source.write_text("<usfx><zz/><foo/><zz><bar/></zz><foo/></usfx>")
    result = normalize(source, output)
    assert result.exit_code == 3, result.output
    diagnostic = json.loads(result.stdout)["diagnostics"][0]
    assert diagnostic["code"] == "unknown_element"
    assert diagnostic["message"] == "Unknown elements: bar=1, foo=2, zz=2"
    source.write_text('<!DOCTYPE usfx [<!ENTITY secret SYSTEM "file:///etc/passwd">]><usfx>&secret;</usfx>')
    result = normalize(source, output)
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "unsafe_xml"
    assert not output.exists()


@pytest.mark.parametrize("encoding", ["bogus", "UTF-32"])
@pytest.mark.parametrize("existing", [False, True])
def test_unsupported_xml_encodings_are_input_errors(tmp_path: Path, encoding: str, existing: bool) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out"
    source.write_bytes(
        (
            f'<?xml version="1.0" encoding="{encoding}"?>'
            '<usfx><book id="GEN"><c id="1"/><v id="1">text</v></book></usfx>'
        ).encode()
    )
    if existing:
        output.write_bytes(b"previous output")
    result = normalize(source, output)
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "invalid_xml"
    assert output.read_bytes() == b"previous output" if existing else not output.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_no_work_is_invalid_structure(tmp_path: Path, existing: bool) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out"
    source.write_text("<usfx/>")
    if existing:
        output.write_bytes(b"previous output")
    result = normalize(source, output)
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "invalid_structure"
    assert output.read_bytes() == b"previous output" if existing else not output.exists()


@pytest.mark.parametrize(
    "flags",
    [
        ["--manifest", "unused"],
        ["--approve-reuse", "0=GEN.1.1"],
        ["--approve-allocation", "0"],
        ["--approve-removal", "GEN.1.1"],
        ["--citation-template", "wsl:work.line"],
    ],
)
def test_usfx_rejects_manifest_options(tmp_path: Path, flags: list[str]) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out"
    source.write_text("<usfx/>")
    result = normalize(source, output, *flags)
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "usage_error"
    assert not output.exists()


def test_genesis_snapshot_and_literal_oracles(tmp_path: Path) -> None:
    source, output = FIXTURES / "gen-1-1-5.usfx.xml", tmp_path / "gen.jsonl"
    raw_source = (ROOT / "data/lat-clementine.usfx.xml").read_bytes()
    end = raw_source.index(b"<ve/>", raw_source.index(b'<v id="5"/>')) + len(b"<ve/>")
    assert source.read_bytes() == raw_source[:end] + b"</book></usfx>\n"
    result = normalize(source, output, "--descriptor", str(ROOT / "data/lat-clementine.descriptor.json"))
    assert result.exit_code == 0, result.output
    assert output.read_bytes() == (FIXTURES / "gen-1-1-5.expected.jsonl").read_bytes()
    rows = [json.loads(line) for line in output.read_bytes().splitlines()]
    assert rows[0]["adapter"] == "usfx/1"
    expected_descriptor = {
        "edition": "Biblia Sacra Vulgata Clementina",
        "publisher": "eBible.org",
        "url": "https://ebible.org/find/details.php?id=latVUC",
        "license": "Public Domain",
    }
    descriptor = json.loads((ROOT / "data/lat-clementine.descriptor.json").read_bytes())
    assert {key: descriptor[key] for key in expected_descriptor} == expected_descriptor
    assert {key: rows[0][key] for key in expected_descriptor} == expected_descriptor
    passages = [r for r in rows if r["record"] == "passage"]
    assert [p["passage_id"] for p in passages] == [
        "vulgata-clementina:GEN.1.1",
        "vulgata-clementina:GEN.1.2",
        "vulgata-clementina:GEN.1.3",
        "vulgata-clementina:GEN.1.4",
        "vulgata-clementina:GEN.1.5",
    ]
    assert passages[0]["text"] == "In principio creavit Deus cælum et terram."
    assert passages[0]["passage_hash"] == "sha256:f231be94df2e0e4e7899e12485f82668689fe2d6596d0ab5af2b31e484619e73"
    assert passages[2]["text"] == "Dixitque Deus : Fiat lux. Et facta est lux."
    assert passages[0]["citation"]["display"] == "GEN 1:1"
    assert passages[2]["citation"]["display"] == "GEN 1:3"
    assert rows[2]["title"] == "Genesis"
    assert CliRunner().invoke(app, ["corpus", "validate", str(output), "--json"]).exit_code == 0


def test_provenance_audit() -> None:
    audit = (ROOT / "docs/license-compatibility-audit.md").read_text().split("## Bundled corpus provenance\n")[1]
    for literal in (
        "data/lat-clementine.usfx.xml",
        "05bcb3f669d9b36deab6cf26950b8a24c7609b292e5af1abfa51717677701950",
        "Biblia Sacra Vulgata Clementina",
        "eBible.org",
        "https://ebible.org/find/details.php?id=latVUC",
        "Public Domain",
        "2026-10-20, maintainer Felix Müller approved",
        "It does not authorize publication or unblock the maintainer release gate.",
    ):
        assert literal in audit


def test_milestone_verse_directly_before_a_closing_tag(tmp_path: Path) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out.jsonl"
    source.write_text('<usfx><book id="GEN"><c id="1"/><p><v id="1"/>a<ve/><v id="2"/></p>b<ve/></book></usfx>')
    result = normalize(source, output)
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_bytes().splitlines()]
    assert [(r["native_ref"], r["text"]) for r in rows if r["record"] == "passage"] == [
        ("GEN.1.1", "a"),
        ("GEN.1.2", "b"),
    ]


def test_explicit_verse_end_after_a_comment_split_at_a_read(tmp_path: Path) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out.jsonl"
    head = '<usfx><book id="A"><c id="1"/><v id="1"><!--'
    source.write_text(f"{head}{'y' * (65536 - len(head) - 2)}/> --></v>later<ve/></book></usfx>")
    assert source.read_bytes()[:65536].endswith(b"/>")
    result = normalize(source, output)
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "empty_passage"
    assert not output.exists()


def test_unterminated_empty_milestone_verse_and_raw_attribute_names(tmp_path: Path) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out.jsonl"
    source.write_text('<usfx><book id="A"><c id="1"/><v id="1"/>a<ve/><v id="2"/></book></usfx>')
    result = normalize(source, output)
    assert result.exit_code == 3, result.output
    diagnostic = json.loads(result.stdout)["diagnostics"][0]
    assert (diagnostic["code"], diagnostic["message"]) == ("invalid_structure", "Unterminated milestone verse")
    source.write_text('<usfx><book id="A"><c id="1"/><v id="1">a <w xml:lang="la">b</w></v></book></usfx>')
    result = normalize(source, output)
    assert result.exit_code == 0, result.output
    passage = [json.loads(line) for line in output.read_bytes().splitlines()][-1]
    assert passage["source_markup"][0]["attributes"] == {"xml:lang": "la"}
