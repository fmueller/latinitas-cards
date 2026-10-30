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
    assert [p["native_ref"] for p in found] == ["1.1.pr", "1.1.3a", "1.2.62b", "1.2.1%2E5%20%C3%A9", "1.3.1"]
    assert [p["passage_id"] for p in found][3] == "test-corpus:1.2.1%2E5%20%C3%A9"
    assert [p["order"] for p in found] == [1, 2, 3, 4, 5]
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
    first, second, third, fourth, fifth = found
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
    # Standalone reg and sic (Cicero/Caesar shapes) keep their words; desc is a note.
    assert fifth["text"] == "Is Helvetiis persuasit, Marco Pisone consulibus."
    assert fifth["notes"] == [{"kind": "note", "text": "codex"}]
    assert fifth["source_markup"] == [
        {"element": "p", "attributes": {}, "span": [0, 48]},
        {"element": "reg", "attributes": {}, "span": [3, 12]},
        {"element": "expan", "attributes": {}, "span": [24, 29]},
        {"element": "ex", "attributes": {}, "span": [25, 29]},
        {"element": "emph", "attributes": {}, "span": [30, 36]},
        {"element": "sic", "attributes": {}, "span": [37, 47]},
    ]
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


DRAMA = (
    '<div type="textpart" subtype="b" n="1"><p>a <sp who="#davus"><speaker>Davus</speaker> <p>quid</p></sp>'
    ' <x:y xmlns:x="urn:x" k="v">z</x:y><sp/></p></div>'
)


def test_unknown_element_names_and_counts(tmp_path: Path) -> None:
    edition = '<div type="textpart" subtype="b" n="1"><p>a<sp/><sp/><x:y xmlns:x="urn:x"/></p></div>'
    result, _ = run_inline(tmp_path, document(edition))
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["message"] == "Unknown elements: sp=2, {urn:x}y=1"
    output = tmp_path / "out.jsonl"
    output.write_bytes(b"previous output\n")
    result, _ = run_inline(tmp_path, document(DRAMA))
    assert result.exit_code == 3, result.output
    (only,) = json.loads(result.stdout)["diagnostics"]
    assert (only["code"], only["severity"]) == ("unknown_element", "error")
    assert only["message"] == "Unknown elements: sp=2, speaker=1, {urn:x}y=1"
    assert output.read_bytes() == b"previous output\n"


def test_allow_unknown_keeps_unknown_elements_in_source_extra(tmp_path: Path) -> None:
    result, output = run_inline(tmp_path, document(DRAMA), "--allow-unknown")
    assert result.exit_code == 0, result.output
    (warning,) = json.loads(result.stdout)["diagnostics"]
    assert (warning["code"], warning["severity"]) == ("unknown_element", "warning")
    assert warning["message"] == "Unknown elements: sp=2, speaker=1, {urn:x}y=1"
    (passage,) = passages(output)
    assert passage["text"] == "a Davus quid z"
    assert passage["source_markup"] == [
        {"element": "p", "attributes": {}, "span": [0, 14]},
        {"element": "p", "attributes": {}, "span": [8, 12]},
    ]
    assert passage["source_extra"] == [
        {"element": "sp", "attributes": {"who": "#davus"}, "span": [2, 12]},
        {"element": "speaker", "attributes": {}, "span": [2, 7]},
        {"element": "{urn:x}y", "attributes": {"k": "v"}, "span": [13, 14]},
        {"element": "sp", "attributes": {}, "span": [14, 14]},
    ]
    validated = CliRunner().invoke(app, ["corpus", "validate", str(output), "--json"])
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.stdout)["result"]["counts"]["passage"] == 1
    # Unknown elements outside passages still have no place to keep their text.
    edition = '<div type="textpart" subtype="b" n="1"><div type="textpart" subtype="c" n="1">a</div><sp>b</sp></div>'
    result, _ = run_inline(tmp_path, document(edition), "--allow-unknown")
    assert result.exit_code == 3, result.output
    assert diagnostics(result)[-1][:2] == ("invalid_structure", "error")


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


def test_markup_fixture_keeps_listed_elements_as_spans_and_alternatives_as_notes(tmp_path: Path) -> None:
    output = tmp_path / "markup.jsonl"
    result = normalize(FIXTURES / "markup.xml", output)
    assert result.exit_code == 0, result.output
    assert diagnostics(result) == [("gap_in_text", "warning", "1.5")]
    first, second, third, fourth, fifth, sixth = passages(output)

    def span(element: str, start: int, end: int, **attributes: str) -> dict[str, Any]:
        return {"element": element, "attributes": attributes, "span": [start, end]}

    # app keeps lem; each rdg is a variant; bibl inside a note is note text; head markup flattens.
    assert first["text"] == "Arma virumque cano."
    assert first["headings"] == [{"text": "Caput primum"}]
    assert first["notes"] == [
        {"kind": "variant", "text": "virosque"},
        {"kind": "variant", "text": "virum"},
        {"kind": "note", "text": "cf. Aen. 1.1"},
    ]
    assert first["source_markup"] == [span("p", 0, 19)]
    # Inside choice, corr/expan/reg are kept without spans and sic/abbr/orig become variants.
    assert second["text"] == "Troiae qui primus ab oris Italiam"
    assert second["notes"] == [
        {"kind": "variant", "text": "Troie"},
        {"kind": "variant", "text": "pri."},
        {"kind": "variant", "text": "Ytaliam"},
    ]
    assert second["source_markup"] == [span("p", 0, 33), span("ex", 14, 17)]
    assert third["text"] == "I. Aeneas et Achates venerunt Carthaginem Libyae."
    assert third["notes"] is None
    assert third["source_markup"] == [
        span("p", 0, 49),
        span("label", 0, 2),
        span("persName", 3, 9),
        span("name", 3, 9),
        span("add", 10, 12, place="above"),
        span("supplied", 13, 20, reason="lost"),
        span("unclear", 21, 29),
        span("placeName", 30, 41),
        span("cb", 41, 41, n="b"),
        span("geogName", 42, 48),
    ]
    assert fourth["text"] == "Quid arma logos tres idibus est hic."
    assert fourth["source_markup"] == [
        span("p", 0, 36),
        span("q", 0, 4),
        span("quote", 5, 9),
        span("foreign", 10, 15, **{"xml:lang": "grc"}),
        span("num", 16, 20, value="3"),
        span("date", 21, 27, when="-0044"),
        span("w", 28, 31, lemma="sum"),
        span("seg", 32, 35, type="x"),
    ]
    # Standalone corr/abbr/orig keep their text; gap is an empty span before the whitespace it follows.
    assert fifth["text"] == "Aeneis recte cos. quom finis ut ait."
    assert fifth["notes"] == [{"kind": "note", "text": "Cic."}, {"kind": "variant", "text": "x"}]
    assert fifth["source_markup"] == [
        span("p", 0, 36),
        span("title", 0, 6),
        span("corr", 7, 12),
        span("abbr", 13, 17),
        span("orig", 18, 22),
        span("gap", 22, 22, reason="lost"),
        span("hi", 23, 28, rend="bold"),
        span("cit", 29, 35),
        span("quote", 29, 35),
        span("milestone", 36, 36, unit="para"),
    ]
    assert sixth["text"] == "A <B> & C"
    assert sixth["notes"] == [{"kind": "note", "text": "descriptio"}]
    assert sixth["source_markup"] == [span("p", 0, 9), span("hi", 8, 9, rend='a"b')]
    assert all(p["source_extra"] is None and p["headings"] is None for p in passages(output)[1:])
    assert output.read_bytes() == (FIXTURES / "markup.expected.jsonl").read_bytes()
    assert CliRunner().invoke(app, ["corpus", "validate", str(output), "--json"]).exit_code == 0


def test_unnumbered_lines_between_lines_attach_their_items(tmp_path: Path) -> None:
    # T-158 failed on the note and silently dropped the page break.
    edition = (
        '<div type="textpart" subtype="book" n="1"><l n="1">a</l><l><note>N</note></l><l><pb n="5"/></l>'
        '<l n="2">b</l></div>'
    )
    result, output = run_inline(tmp_path, document(edition))
    assert result.exit_code == 0, result.output
    first, second = passages(output)
    assert first["notes"] == [{"kind": "note", "text": "N"}]
    assert first["source_markup"] == [{"element": "pb", "attributes": {"n": "5"}, "span": [1, 1]}]
    assert (second["notes"], second["source_markup"]) == (None, None)


def test_gaps_warn_once_per_passage_with_their_count(tmp_path: Path) -> None:
    edition = (
        '<div type="textpart" subtype="s" n="1"><p>a <gap/>b <gap/><gap/>c</p></div>'
        '<div type="textpart" subtype="s" n="2">d</div>'
    )
    result, output = run_inline(tmp_path, document(edition))
    assert result.exit_code == 0, result.output
    (warning,) = json.loads(result.stdout)["diagnostics"]
    assert (warning["code"], warning["severity"], warning["ref"]) == ("gap_in_text", "warning", "1")
    assert warning["message"] == "Passage 1 has 3 gaps in its text"
    first, _ = passages(output)
    assert [entry["span"] for entry in first["source_markup"] if entry["element"] == "gap"] == [[1, 1], [3, 3], [3, 3]]


@pytest.mark.parametrize(
    "edition",
    [
        '<div type="textpart" subtype="b" n="1"><p>a <note>n <foo k="v">z</foo></note></p></div>',
        '<div type="textpart" subtype="b" n="1"><head>H <foo/></head><p>a</p></div>',
        '<div type="textpart" subtype="b" n="1"><p>a <choice><sic><foo>x</foo></sic><corr>y</corr></choice></p></div>',
        '<div type="textpart" subtype="b" n="1"><p>a <app><lem>b</lem><rdgGrp><rdg>c</rdg></rdgGrp></app></p></div>',
        '<div type="textpart" subtype="b" n="1"><l n="1">a</l><foo/><l n="2">b</l></div>',
    ],
)
def test_allowed_unknown_elements_without_a_text_position_are_invalid(tmp_path: Path, edition: str) -> None:
    # --allow-unknown keeps unknown elements only where they have a span in passage text.
    result, output = run_inline(tmp_path, document(edition), "--allow-unknown")
    assert result.exit_code == 3, result.output
    assert diagnostics(result) == [("invalid_structure", "error", None)]
    assert not output.exists()


@pytest.mark.parametrize("unknown", ["<foo/>", "<foo> \n </foo>", "<foo><bar/></foo>"])
@pytest.mark.parametrize("verse", [False, True])
@pytest.mark.parametrize("allow_unknown", [False, True])
def test_unknown_in_omitted_passage_fails_without_replacing_output(
    tmp_path: Path, unknown: str, verse: bool, allow_unknown: bool
) -> None:
    if verse:
        edition = f'<div type="textpart" subtype="b" n="1"><l n="1">a</l><l n="2">{unknown}</l><l n="3">b</l></div>'
    else:
        edition = (
            '<div type="textpart" subtype="s" n="1">a</div>'
            f'<div type="textpart" subtype="s" n="2">{unknown}</div>'
            '<div type="textpart" subtype="s" n="3">b</div>'
        )
    output = tmp_path / "out.jsonl"
    output.write_bytes(b"existing output\n")
    flags = ("--allow-unknown",) if allow_unknown else ()
    result, _ = run_inline(tmp_path, document(edition), *flags)
    assert result.exit_code == 3, result.output
    code = "invalid_structure" if allow_unknown else "unknown_element"
    assert diagnostics(result) == [(code, "error", None)]
    if allow_unknown:
        assert "no canonical text position" in result.stdout
    assert output.read_bytes() == b"existing output\n"


def test_between_line_items_attach_across_groups_and_unnumbered_lines(tmp_path: Path) -> None:
    edition = (
        '<div type="textpart" subtype="book" n="1"><lg><l n="1">a</l></lg><rdg>A</rdg><l><note>N</note></l><cb/>'
        '<lg><l n="2">b</l></lg><l><pb n="5"/></l> <gap/><lg><l n="3">c</l></lg></div>'
    )
    result, output = run_inline(tmp_path, document(edition))
    assert result.exit_code == 0, result.output
    assert diagnostics(result) == [("gap_in_text", "warning", "1.2")]
    first, second, third = passages(output)
    assert first["notes"] == [{"kind": "variant", "text": "A"}, {"kind": "note", "text": "N"}]
    assert first["source_markup"] == [{"element": "cb", "attributes": {}, "span": [1, 1]}]
    assert second["source_markup"] == [
        {"element": "pb", "attributes": {"n": "5"}, "span": [1, 1]},
        {"element": "gap", "attributes": {}, "span": [1, 1]},
    ]
    assert (second["notes"], third["notes"], third["source_markup"]) == (None, None, None)


@pytest.mark.parametrize(
    "edition",
    [
        '<l n="1">a</l><bibl>B</bibl><l n="2">b</l>',
        '<l n="1">a</l><desc>D</desc><l n="2">b</l>',
        '<l n="1">a</l><l>stray</l><l n="2">b</l>',
        '<l n="1">a</l><app><rdg>x</rdg></app><l n="2">b</l>',
        '<l n="1">a<choice>x<sic>y</sic><corr>z</corr></choice></l>',
        '<l n="1">a<choice><sic>y</sic><hi>z</hi></choice></l>',
        '<l n="1">a<app><lem>b</lem>c</app></l>',
        '<l n="1">a<app><lem>b</lem><note>c</note></app></l>',
        '<l n="1">a<lem>b</lem></l>',
        '<l n="1">a<gap>lost</gap></l>',
        '<l n="1">a<cb>b</cb></l>',
        # More than one kept alternative would join readings into one word.
        '<l n="1">x <choice><sic>Troie</sic><corr>Troiae</corr><corr>Troia</corr></choice> y</l>',
        '<l n="1">x <choice><reg>Troiae</reg><corr>Troia</corr></choice> y</l>',
        '<l n="1"><app><lem>a</lem><lem>b</lem></app></l>',
        # Flattened notes, headings and variants have no text position for alternatives or gaps.
        '<head>Bellum <choice><sic>Troie</sic><corr>Troiae</corr></choice></head><l n="1">a</l>',
        '<l n="1">a<note>cf. <choice><abbr>Verg</abbr><expan>Vergilius</expan></choice></note></l>',
        '<l n="1">a</l><note>lege <app><lem>cano</lem><rdg>canto</rdg></app></note><l n="2">b</l>',
        '<l n="1">a<app><lem>x</lem><rdg>y<choice><sic>z</sic><corr>w</corr></choice></rdg></app></l>',
        '<l n="1">a<choice><sic><choice><abbr>b</abbr><expan>bb</expan></choice></sic><corr>c</corr></choice></l>',
        '<l n="1">a<note>x <gap/> y</note></l>',
        '<l n="1">a<del>b<gap/></del></l>',
        '<l n="1">a<bibl><hi>Verg.</hi> <gap/></bibl></l>',
        '<head>A<lem>B</lem></head><l n="1">a</l>',
        '<l n="1">a<note>lege <rdg>cano</rdg><rdg>canto</rdg></note></l>',
    ],
)
def test_misplaced_editorial_markup_is_invalid(tmp_path: Path, edition: str) -> None:
    result, output = run_inline(tmp_path, document(f'<div type="textpart" subtype="book" n="1">{edition}</div>'))
    assert result.exit_code == 3, result.output
    assert diagnostics(result)[0][:2] == ("invalid_structure", "error")
    assert not output.exists()


def test_alternatives_inside_lines_and_bare_readings(tmp_path: Path) -> None:
    edition = (
        '<div type="textpart" subtype="book" n="1">'
        '<l n="1"><choice> <corr>Troiae</corr> <sic>Troie <l n="9">x</l></sic> </choice> qui</l>'
        '<l n="2">a<app><rdg>b</rdg><rdg/></app> <rdg>c</rdg>d <sic>quom</sic> <reg>cum</reg></l></div>'
    )
    result, output = run_inline(tmp_path, document(edition))
    assert result.exit_code == 0, result.output
    first, second = passages(output)
    # The kept alternative records no span; whitespace between alternatives is not text.
    assert (first["text"], first["source_markup"]) == ("Troiae qui", None)
    assert first["notes"] == [{"kind": "variant", "text": "Troie x"}]
    assert second["text"] == "a d quom cum"
    assert second["notes"] == [
        {"kind": "variant", "text": "b"},
        {"kind": "variant", "text": ""},
        {"kind": "variant", "text": "c"},
    ]
    assert second["source_markup"] == [
        {"element": "sic", "attributes": {}, "span": [4, 8]},
        {"element": "reg", "attributes": {}, "span": [9, 12]},
    ]


def test_annotations_segment_works_on_a_tei_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "markup.jsonl"
    assert normalize(FIXTURES / "markup.xml", corpus).exit_code == 0
    result = CliRunner().invoke(
        app, ["annotations", "segment", str(corpus), "--timestamp", "2026-01-01T00:00:00Z", "--json"]
    )
    assert result.exit_code == 0, result.output
    lines = corpus.with_suffix(".annotations.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if json.loads(line)["record"] == "annotation"]
    assert len(records) == 6
    sentences = {r["anchor"]["passage_id"]: r["value"]["sentences"] for r in records}

    def sentence(start: int, end: int, tokens: list[int], *, continued: bool = False, continues: bool = False) -> Any:
        return {"continued": continued, "continues": continues, "span": [start, end], "tokens": tokens}

    assert sentences["test-corpus:1.1"] == [sentence(0, 19, [0, 4])]
    # Section 2 has no terminator, so its sentence continues into section 3.
    assert sentences["test-corpus:1.2"] == [sentence(0, 33, [0, 6], continues=True)]
    assert sentences["test-corpus:1.3"] == [sentence(0, 2, [0, 2], continued=True), sentence(3, 49, [2, 9])]
    assert sentences["test-corpus:1.6"] == [sentence(0, 9, [0, 6], continues=True)]
