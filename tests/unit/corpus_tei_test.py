"""TEI structure, citations and header metadata through the public CLI, with literal expectations."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from latinitas_cards.cli import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/corpus/tei"
TEI = 'xmlns="http://www.tei-c.org/ns/1.0"'


def normalize(source: Path, output: Path, *flags: str, corpus_id: str = "test-corpus") -> Result:
    before = source.read_bytes()
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "normalize",
            str(source),
            "--adapter",
            "tei",
            "--corpus-id",
            corpus_id,
            "--output",
            str(output),
            "--json",
            *flags,
        ],
    )
    assert source.read_bytes() == before
    return result


def document(
    edition: str, header: str = "", *, encoding: str = "", body_extra: str = "", lang: str = "lat", urn: str = ""
) -> str:
    n = f' n="{urn}"' if urn else ""
    return (
        f"<TEI {TEI}><teiHeader><fileDesc><titleStmt><title>T</title></titleStmt>{header}</fileDesc>"
        f"{encoding}</teiHeader>"
        f'<text><body><div type="edition"{n} xml:lang="{lang}">{edition}</div>{body_extra}</body></text></TEI>'
    )


def run_inline(tmp_path: Path, xml: str, *flags: str, name: str = "input.xml") -> tuple[Result, Path]:
    source, output = tmp_path / name, tmp_path / "out.jsonl"
    source.write_text(xml, encoding="utf-8")
    return normalize(source, output, *flags), output


def rows(output: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]


def passages(output: Path) -> list[dict[str, Any]]:
    return [row for row in rows(output) if row["record"] == "passage"]


def diagnostics(result: Result) -> list[tuple[str, str, str | None]]:
    return [(d["code"], d["severity"], d["ref"]) for d in json.loads(result.stdout)["diagnostics"]]


def test_prose_fixture_matches_literal_expectations_and_golden(tmp_path: Path) -> None:
    output = tmp_path / "prose.jsonl"
    result = normalize(FIXTURES / "prose.xml", output)
    assert result.exit_code == 0, result.output
    assert diagnostics(result) == []
    header, corpus, work, *found = rows(output)
    raw = (FIXTURES / "prose.xml").read_bytes()
    assert header["adapter"] == "tei/1"
    assert header["source"] == {
        "file_name": "prose.xml",
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    assert (header["edition"], header["publisher"], header["url"], header["license"]) == (
        "C. Iuli Caesaris Commentarii",
        "Synthetic Perseus",
        "https://example.org/caesar",
        "https://creativecommons.org/licenses/by-sa/4.0/",
    )
    assert corpus["title"] == "De Bello Gallico"
    assert (work["work"], work["title"], work["order"]) == ("phi0448.phi001", "De Bello Gallico", 1)
    assert [p["native_ref"] for p in found] == ["1.1.pr", "1.1.3a", "1.2.62b", "1.2.1%2E5%20%C3%A9"]
    assert [p["passage_id"] for p in found][-1] == "test-corpus:1.2.1%2E5%20%C3%A9"
    assert [p["order"] for p in found] == [1, 2, 3, 4]
    assert {p["work"] for p in found} == {"phi0448.phi001"}
    assert found[3]["citation"] == {
        "scheme": "tei-cts",
        "display": "1.2.1.5 é",
        "parts": [
            {"unit": "book", "value": "1"},
            {"unit": "chapter", "value": "2"},
            {"unit": "section", "value": "1.5 é"},
        ],
    }
    first, second, third, fourth = found
    assert first["text"] == "Gallia est omnis divisa in partes tres."
    assert first["headings"] == [{"text": "Liber primus"}]
    assert first["notes"] == [{"kind": "note", "text": "nota bene"}]
    assert first["source_markup"] == [
        {"element": "p", "attributes": {}, "span": [0, 39]},
        # Empty spans exclude the whitespace before them, as USFX spans do.
        {"element": "lb", "attributes": {}, "span": [16, 16]},
    ]
    assert second["text"] == "Quarum unam incolunt Belgae."
    assert second["notes"] == [{"kind": "variant", "text": "deleta"}]
    assert second["headings"] is None
    assert second["source_markup"] == [{"element": "p", "attributes": {"n": "x"}, "span": [0, 28]}]
    assert third["text"] == "Apud Helvetios longe nobilissimus fuit Orgetorix."
    assert third["notes"] is None
    assert third["source_markup"] == [
        {"element": "pb", "attributes": {"n": "2"}, "span": [0, 0]},
        {"element": "ab", "attributes": {}, "span": [0, 49]},
        {"element": "milestone", "attributes": {"unit": "para"}, "span": [14, 14]},
    ]
    assert fourth["text"] == "Is M. Messala consule."
    assert fourth["passage_hash"] == "sha256:" + hashlib.sha256(b"Is M. Messala consule.").hexdigest()
    assert fourth["norm"] == "is m. messala consule."
    assert output.read_bytes() == (FIXTURES / "prose.expected.jsonl").read_bytes()
    assert CliRunner().invoke(app, ["corpus", "validate", str(output), "--json"]).exit_code == 0
    assert not output.with_suffix(".manifest.json").exists()


def test_verse_fixture_matches_literal_expectations_and_golden(tmp_path: Path) -> None:
    output = tmp_path / "verse.jsonl"
    result = normalize(FIXTURES / "verse.xml", output)
    assert result.exit_code == 0, result.output
    assert diagnostics(result) == [("skipped_empty_passage", "warning", "1.3")]
    header, corpus, work, *found = rows(output)
    assert (header["edition"], header["publisher"], header["url"], header["license"]) == (
        "Synthetic verse edition",
        "Synthetic Perseus",
        None,
        "Public domain",
    )
    assert (corpus["title"], work["work"], work["title"]) == ("Metamorphoses", "phi0959.phi006", "Metamorphoses")
    assert [p["native_ref"] for p in found] == ["1.1", "1.2", "1.4", "2.1"]
    assert found[0]["citation"] == {
        "scheme": "tei-cts",
        "display": "1.1",
        "parts": [{"unit": "book", "value": "1"}, {"unit": "line", "value": "1"}],
    }
    assert [p["order"] for p in found] == [1, 2, 3, 4]
    line1, line2, line4, book2 = found
    assert line1["text"] == "In nova fert animus mutatas dicere formas"
    assert line1["headings"] == [{"text": "Liber I"}]
    # A between-line note attaches to the preceding line; its l[@n] and del are note text.
    assert line1["notes"] == [{"kind": "note", "text": "Codd. add versus spurius deletum"}]
    assert line2["text"] == "corpora; di, coeptis nam vos mutastis et illas"
    # The omitted all-del line 3 routes its deletion to the preceding kept line, after the line's own note.
    assert line2["notes"] == [{"kind": "note", "text": "nam vos"}, {"kind": "variant", "text": "adspirate meis"}]
    # The milestone after the omitted all-del line 3 attaches to the nearest kept preceding line.
    assert line2["source_markup"] == [
        {"element": "pb", "attributes": {"n": "3"}, "span": [46, 46]},
        {"element": "milestone", "attributes": {"unit": "card", "n": "4"}, "span": [46, 46]},
    ]
    assert line4["text"] == "ad mea perpetuum deducite tempora carmen."
    assert (line4["notes"], line4["headings"], line4["source_markup"]) == (None, None, None)
    # No line precedes the del in book 2, so it attaches to the next line.
    assert book2["text"] == "Regia Solis erat sublimibus alta columnis"
    assert book2["notes"] == [{"kind": "variant", "text": "Nondum"}]
    assert book2["source_markup"] == [{"element": "lb", "attributes": {}, "span": [16, 16]}]
    assert output.read_bytes() == (FIXTURES / "verse.expected.jsonl").read_bytes()
    assert CliRunner().invoke(app, ["corpus", "validate", str(output), "--json"]).exit_code == 0


@pytest.mark.parametrize(
    "edition,code",
    [
        (
            '<div type="textpart" subtype="book" n="1"><l n="1">a</l><l n="1">b</l></div>',
            "duplicate_source_ref",
        ),
        (
            '<div type="textpart" subtype="book" n="1"><div type="textpart" subtype="chapter" n="1">a</div></div>'
            '<div type="textpart" subtype="book" n="2"><div type="textpart" subtype="section" n="1">b</div></div>',
            "invalid_structure",
        ),
        (
            '<div type="textpart" subtype="book" n="1"><l n="1">a</l></div>'
            '<div type="textpart" subtype="book" n="2">prose</div>',
            "invalid_structure",
        ),
        ('<div type="textpart" subtype="book" n="1">stray <l n="1">a</l></div>', "invalid_structure"),
        ('<div type="textpart" subtype="book" n="1"><lg>stray<l n="1">a</l></lg></div>', "invalid_structure"),
        ('<div type="textpart" subtype="book" n="1"><l n="1">a</l><p>b</p></div>', "invalid_structure"),
        ('<div type="textpart" n="1">a</div>', "invalid_structure"),
        ('<div type="textpart" subtype="book">a</div>', "invalid_structure"),
        ('<div type="textpart" n="1"><div type="textpart" subtype="c" n="1">a</div></div>', "invalid_structure"),
        (
            '<div type="textpart" subtype="book" n="1">a<div type="textpart" subtype="c" n="1">b</div></div>',
            "invalid_structure",
        ),
        ('<div type="commentary">a</div>', "invalid_structure"),
        ("", "invalid_structure"),
        ('<div type="textpart" subtype="book" n="1"><l n="1"><del>x</del></l></div>', "invalid_structure"),
        ('<div type="textpart" subtype="book" n="1"><l n="1">a<l n="2">b</l></l></div>', "invalid_structure"),
    ],
)
def test_structure_and_citation_failures_leave_output(tmp_path: Path, edition: str, code: str) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out.jsonl"
    source.write_text(document(edition), encoding="utf-8")
    output.write_bytes(b"previous output\n")
    result = normalize(source, output)
    assert result.exit_code == 3, result.output
    assert diagnostics(result)[0][:2] == (code, "error")
    assert output.read_bytes() == b"previous output\n"


def test_non_numeric_refs_and_unnumbered_lines_are_flattened(tmp_path: Path) -> None:
    edition = (
        '<div type="textpart" subtype="poem" n="pr"><l n="1">Arma</l><note>see <l n="x">nota</l></note></div>'
        '<div type="textpart" subtype="poem" n="62b"><l n="3a">virum<del><l n="9">dele</l></del></l></div>'
    )
    result, output = run_inline(tmp_path, document(edition))
    assert result.exit_code == 0, result.output
    found = passages(output)
    assert [(p["native_ref"], p["text"]) for p in found] == [("pr.1", "Arma"), ("62b.3a", "virum")]
    assert found[0]["notes"] == [{"kind": "note", "text": "see nota"}]
    assert found[1]["notes"] == [{"kind": "variant", "text": "dele"}]
    prose = '<div type="textpart" subtype="section" n="1"><p>Arma <lg><l>virum</l> <l>cano</l></lg></p></div>'
    result, output = run_inline(tmp_path, document(prose))
    assert result.exit_code == 0, result.output
    assert [(p["native_ref"], p["text"]) for p in passages(output)] == [("1", "Arma virum cano")]


def test_lines_directly_in_edition_and_trailing_container_items(tmp_path: Path) -> None:
    edition = '<head>Carmen</head><l n="1">Odi et amo.</l><lg><l n="2">quare id faciam</l></lg><note>end</note>'
    result, output = run_inline(tmp_path, document(edition))
    assert result.exit_code == 0, result.output
    first, second = passages(output)
    assert first["citation"]["parts"] == [{"unit": "line", "value": "1"}]
    assert (first["headings"], second["notes"]) == ([{"text": "Carmen"}], [{"kind": "note", "text": "end"}])
    prose = (
        '<div type="textpart" subtype="book" n="1"><div type="textpart" subtype="chapter" n="1">a</div>'
        "<note>trailing</note><pb/></div>"
    )
    result, output = run_inline(tmp_path, document(prose))
    assert result.exit_code == 0, result.output
    (only,) = passages(output)
    assert only["notes"] == [{"kind": "note", "text": "trailing"}]
    assert only["source_markup"] == [{"element": "pb", "attributes": {}, "span": [1, 1]}]


def test_empty_prose_passage_is_skipped_with_warning(tmp_path: Path) -> None:
    edition = (
        '<div type="textpart" subtype="section" n="1"><p> </p></div>'
        '<div type="textpart" subtype="section" n="2"><p>b</p></div>'
    )
    result, output = run_inline(tmp_path, document(edition))
    assert result.exit_code == 0, result.output
    assert diagnostics(result) == [("skipped_empty_passage", "warning", "1")]
    (kept,) = passages(output)
    assert kept["native_ref"] == "2"
    # The omitted passage's own paragraph span is not moved onto its neighbour.
    assert kept["source_markup"] == [{"element": "p", "attributes": {}, "span": [0, 1]}]


@pytest.mark.parametrize(
    "patterns,warned",
    [
        (["line", "book"], False),
        (["LINE", "Book"], False),
        (["Book", "line"], True),
        (["line"], True),
        ([], False),
    ],
)
def test_cref_pattern_mismatch_only_warns(tmp_path: Path, patterns: list[str], warned: bool) -> None:
    # Patterns are listed deepest first; only the CTS refsDecl counts.
    refs = "".join(f'<cRefPattern n="{label}"/>' for label in patterns)
    encoding = (
        f'<encodingDesc><refsDecl n="CTS">{refs}</refsDecl><refsDecl><cRefPattern n="x"/></refsDecl></encodingDesc>'
    )
    edition = '<div type="textpart" subtype="book" n="1"><l n="1">a</l></div>'
    result, output = run_inline(tmp_path, document(edition, encoding=encoding))
    assert result.exit_code == 0, result.output
    assert [code for code, *_ in diagnostics(result)] == (["citation_mismatch"] if warned else [])
    assert [p["citation"]["parts"][0]["unit"] for p in passages(output)] == ["book"]


@pytest.mark.parametrize(
    "xml,code",
    [
        (
            f'<TEI {TEI}><teiHeader/><text><body><div type="translation" xml:lang="eng">x</div></body></text></TEI>',
            "invalid_structure",
        ),
        (f"<TEI {TEI}><teiHeader/><text><body/></text></TEI>", "invalid_structure"),
        (document('<div type="textpart" subtype="b" n="1">x</div>', lang="grc"), "unsupported_language"),
        (
            document('<div type="textpart" subtype="b" n="1">x</div>').replace(' xml:lang="lat"', ""),
            "unsupported_language",
        ),
        (
            document("x").replace('<div type="edition"', '<div type="edition" xml:lang="la"></div><div type="edition"'),
            "invalid_structure",
        ),
        (f"<TEI {TEI}><text><body/></text></TEI>", "invalid_structure"),
        (document("x").replace(TEI, 'xmlns="http://www.tei-c.org/ns/2.0"'), "unknown_element"),
        (document("x").replace(f" {TEI}", ""), "unknown_element"),
        (document("<sp>x</sp>"), "unknown_element"),
        ('<!DOCTYPE TEI><TEI xmlns="http://www.tei-c.org/ns/1.0"/>', "unsafe_xml"),
        ('<!DOCTYPE TEI [<!ENTITY e "x">]><TEI xmlns="http://www.tei-c.org/ns/1.0">&e;</TEI>', "unsafe_xml"),
        ("<TEI", "invalid_xml"),
    ],
)
def test_edition_language_namespace_and_unsafe_inputs(tmp_path: Path, xml: str, code: str) -> None:
    result, output = run_inline(tmp_path, xml)
    assert result.exit_code == 3, result.output
    assert diagnostics(result)[0][:2] == (code, "error")
    assert not output.exists()


def test_unknown_element_names_and_counts(tmp_path: Path) -> None:
    edition = '<div type="textpart" subtype="b" n="1"><p>a<sp/><sp/><x:y xmlns:x="urn:x"/></p></div>'
    result, _ = run_inline(tmp_path, document(edition))
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["message"] == "Unknown elements: sp=2, {urn:x}y=1"


def test_translation_division_is_skipped_with_warning(tmp_path: Path) -> None:
    extra = '<div type="translation" xml:lang="eng"><p>Gaul <sp>unparsed</sp></p></div><div type="commentary"/>'
    result, output = run_inline(tmp_path, document('<div type="textpart" subtype="b" n="1">a</div>', body_extra=extra))
    assert result.exit_code == 0, result.output
    assert diagnostics(result) == [("skipped_division", "warning", None)]
    assert "commentary, translation" in json.loads(result.stdout)["diagnostics"][0]["message"]
    assert [p["text"] for p in passages(output)] == ["a"]


@pytest.mark.parametrize("depth,code", [(64, None), (65, "depth_exceeded")])
def test_depth_boundary(tmp_path: Path, depth: int, code: str | None) -> None:
    # TEI, text, body, edition div and textpart div are five levels.
    edition = '<div type="textpart" subtype="b" n="1">' + "<p>" * (depth - 5) + "a" + "</p>" * (depth - 5) + "</div>"
    result, output = run_inline(tmp_path, document(edition))
    assert result.exit_code == (3 if code else 0), result.output
    if code:
        assert diagnostics(result)[0][0] == code
        assert not output.exists()


def test_size_boundary(tmp_path: Path) -> None:
    prefix = document('<div type="textpart" subtype="b" n="1">a</div>').encode()
    source = tmp_path / "large.xml"
    with source.open("wb") as stream:
        stream.write(prefix)
        stream.truncate(64 * 1024 * 1024)
    # Pad with trailing whitespace after the root: exactly 64 MiB is accepted.
    with source.open("r+b") as stream:
        stream.seek(len(prefix))
        remaining = 64 * 1024 * 1024 - len(prefix)
        while remaining:
            chunk = b" " * min(remaining, 1 << 20)
            stream.write(chunk)
            remaining -= len(chunk)
    output = tmp_path / "out.jsonl"
    result = normalize(source, output)
    assert result.exit_code == 0, result.output
    assert rows(output)[0]["source"]["bytes"] == 64 * 1024 * 1024
    with source.open("ab") as stream:
        stream.write(b" ")
    output.unlink()
    result = normalize(source, output)
    assert result.exit_code == 3, result.output
    assert diagnostics(result)[0][0] == "input_too_large"
    assert not output.exists()


@pytest.mark.parametrize(
    "header,urn,flags,expected",
    [
        ("", "urn:cts:latinLit:phi0448.phi001.perseus-lat2", ["--work", "BG_1.x"], "BG_1.x"),
        ("", "urn:cts:latinLit:phi0448.phi001.perseus-lat2", [], "phi0448.phi001"),
        ("", "urn:cts:latinLit:phi0448.phi001", [], "phi0448.phi001"),
        (
            '<publicationStmt><idno type="filename">phi0690.phi003.perseus-lat2.xml</idno></publicationStmt>',
            "",
            [],
            "phi0690.phi003",
        ),
        (
            '<publicationStmt><idno type="filename">phi0690.phi003.perseus-lat2.xml</idno></publicationStmt>',
            "urn:cts:latinLit:phi0448",
            [],
            "phi0690.phi003",
        ),
        ('<publicationStmt><idno type="filename">Aeneid.xml</idno></publicationStmt>', "", [], "test-corpus"),
        ('<publicationStmt><idno type="URI">phi0690.phi003.x</idno></publicationStmt>', "", [], "test-corpus"),
        ("", "", [], "test-corpus"),
        ("", "", ["--work", "w" * 64], "w" * 64),
        ("", f"urn:cts:latinLit:{'a' * 60}.phi001", [], "test-corpus"),
    ],
)
def test_work_id_fallbacks(tmp_path: Path, header: str, urn: str, flags: list[str], expected: str) -> None:
    result, output = run_inline(
        tmp_path, document('<div type="textpart" subtype="b" n="1">a</div>', header, urn=urn), *flags
    )
    assert result.exit_code == 0, result.output
    assert [(r["work"], r["title"]) for r in rows(output) if r["record"] == "work"] == [(expected, "T")]
    assert {p["work"] for p in passages(output)} == {expected}


@pytest.mark.parametrize("work", ["", "has space", "x" * 65, "é"])
def test_invalid_work_is_usage_error(tmp_path: Path, work: str) -> None:
    result, output = run_inline(tmp_path, document('<div type="textpart" subtype="b" n="1">a</div>'), "--work", work)
    assert result.exit_code == 2, result.output
    assert diagnostics(result)[0][0] == "usage_error"
    assert not output.exists()


def test_header_fallbacks_and_descriptor_override(tmp_path: Path) -> None:
    header = (
        "<publicationStmt><publisher>Header publisher</publisher>"
        '<idno type="URI">https://example.org/x</idno><availability><licence> CC0  1.0 </licence></availability>'
        "</publicationStmt><sourceDesc><biblStruct><monogr><title>Source <emph>title</emph></title></monogr>"
        "</biblStruct><bibl><title>Second</title></bibl></sourceDesc>"
    )
    xml = document('<div type="textpart" subtype="b" n="1">a</div>', header)
    result, output = run_inline(tmp_path, xml)
    assert result.exit_code == 0, result.output
    header_row, corpus = rows(output)[:2]
    assert (header_row["edition"], header_row["publisher"], header_row["url"], header_row["license"]) == (
        "Source title",
        "Header publisher",
        "https://example.org/x",
        "CC0 1.0",
    )
    assert corpus["title"] == "T"
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(
        json.dumps(
            {
                "format": "latinitas-corpus-descriptor",
                "version": 1,
                "publisher": "Header publisher",
                "license": "CC BY 4.0",
                "title": "Override title",
                "edition": None,
                "period": "classical",
            }
        )
    )
    result = normalize(tmp_path / "input.xml", output, "--descriptor", str(descriptor))
    assert result.exit_code == 0, result.output
    messages = [d["message"] for d in json.loads(result.stdout)["diagnostics"]]
    assert [d[:2] for d in diagnostics(result)] == [("descriptor_override", "warning")] * 2
    assert any("license" in message for message in messages) and any("title" in message for message in messages)
    header_row, corpus, work = rows(output)[:3]
    assert (header_row["edition"], header_row["publisher"], header_row["license"]) == (
        "Source title",
        "Header publisher",
        "CC BY 4.0",
    )
    assert (corpus["title"], corpus["period"], work["title"]) == ("Override title", "classical", "T")


@pytest.mark.parametrize(
    "flags",
    [
        ["--manifest", "unused.json"],
        ["--citation-template", "wsl:work.section.line"],
        ["--approve-reuse", "0=p000001"],
        ["--approve-allocation", "0"],
        ["--approve-removal", "p000001"],
        ["--allow-unknown"],
    ],
)
def test_text_only_options_are_usage_errors(tmp_path: Path, flags: list[str]) -> None:
    result = normalize(FIXTURES / "prose.xml", tmp_path / "out.jsonl", *flags)
    assert result.exit_code == 2, result.output
    assert diagnostics(result)[0][0] == "usage_error"
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize("adapter,source", [("text", "text/wsl-three.txt"), ("usfx", "usfx/gen-1-1-5.usfx.xml")])
def test_work_option_requires_tei(tmp_path: Path, adapter: str, source: str) -> None:
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "normalize",
            str(FIXTURES.parent / source),
            "--adapter",
            adapter,
            "--corpus-id",
            "test-corpus",
            "--output",
            str(tmp_path / "out.jsonl"),
            "--work",
            "x",
            "--json",
        ],
    )
    assert result.exit_code == 2, result.output
    assert diagnostics(result)[0][0] == "usage_error"


@pytest.mark.parametrize(
    "edition",
    [
        '<div type="textpart" subtype="book" n="1"><head><l n="1">x</l></head></div>',
        '<div type="textpart" subtype="book" n="1"><l n="1">a</l><head>T <l n="2">b</l></head><l n="3">c</l></div>',
        '<div type="textpart" subtype="b" n="1"><head>L <l n="9">v</l></head>'
        '<div type="textpart" subtype="c" n="1">a</div></div>',
        '<div type="textpart" subtype="book" n="1"><l n="1">a</l><l n="">b</l></div>',
    ],
)
def test_lines_in_headings_and_empty_line_numbers_are_invalid(tmp_path: Path, edition: str) -> None:
    # The spec counts an l[@n] inside head as a passage; it must never silently become heading text.
    result, output = run_inline(tmp_path, document(edition))
    assert result.exit_code == 3, result.output
    assert diagnostics(result)[0][:2] == ("invalid_structure", "error")
    assert not output.exists()


def test_empty_unnumbered_lines_between_lines_are_flattened(tmp_path: Path) -> None:
    edition = '<div type="textpart" subtype="book" n="1"><l n="1">a</l><l/><lg><l> </l><l n="2">b</l></lg></div>'
    result, output = run_inline(tmp_path, document(edition))
    assert result.exit_code == 0, result.output
    assert [(p["native_ref"], p["text"]) for p in passages(output)] == [("1.1", "a"), ("1.2", "b")]


def test_omitted_passage_attachments_follow_between_line_rules(tmp_path: Path) -> None:
    verse = (
        '<div type="textpart" subtype="book" n="1"><l n="1">a</l>'
        '<l n="2"><head>H</head><del>x</del><note>N</note><pb n="7"/></l><l n="3">c</l></div>'
    )
    result, output = run_inline(tmp_path, document(verse))
    assert result.exit_code == 0, result.output
    assert diagnostics(result) == [("skipped_empty_passage", "warning", "1.2")]
    first, third = passages(output)
    # Notes, deletions and spans go to the preceding kept line; the heading goes to the next.
    assert first["notes"] == [{"kind": "variant", "text": "x"}, {"kind": "note", "text": "N"}]
    assert first["source_markup"] == [{"element": "pb", "attributes": {"n": "7"}, "span": [1, 1]}]
    assert (first["headings"], third["headings"], third["notes"]) == (None, [{"text": "H"}], None)
    prose = (
        '<div type="textpart" subtype="chapter" n="1"><head>Caput</head><note>nota</note></div>'
        '<div type="textpart" subtype="chapter" n="2"><p>Gallia</p></div>'
    )
    result, output = run_inline(tmp_path, document(prose))
    assert result.exit_code == 0, result.output
    (second,) = passages(output)
    assert (second["headings"], second["notes"]) == ([{"text": "Caput"}], [{"kind": "note", "text": "nota"}])


def test_nfc_equal_citations_are_duplicates(tmp_path: Path) -> None:
    edition = '<div type="textpart" subtype="b" n="\u00e9">a</div><div type="textpart" subtype="b" n="e\u0301">b</div>'
    result, _ = run_inline(tmp_path, document(edition))
    assert result.exit_code == 3, result.output
    assert diagnostics(result)[0][:2] == ("duplicate_source_ref", "error")


def test_descriptor_filling_absent_header_fields_does_not_warn(tmp_path: Path) -> None:
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(
        json.dumps({"format": "latinitas-corpus-descriptor", "version": 1, "url": "https://x", "publisher": "P"})
    )
    result, output = run_inline(
        tmp_path, document('<div type="textpart" subtype="b" n="1">a</div>'), "--descriptor", str(descriptor)
    )
    assert result.exit_code == 0, result.output
    assert diagnostics(result) == []
    assert (rows(output)[0]["url"], rows(output)[0]["publisher"]) == ("https://x", "P")
