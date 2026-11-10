"""Independent literal expectations for strict markup and XML trust boundaries."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner, Result

from latinitas_cards.cli import app
from latinitas_cards.cli_contract import ContractError
from latinitas_cards.usfx_adapter import build_usfx

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/corpus/usfx/markup"


def run(source: Path, output: Path, *flags: str) -> Result:
    return CliRunner().invoke(
        app,
        [
            "corpus",
            "normalize",
            str(source),
            "--adapter",
            "usfx",
            "--corpus-id",
            "test-corpus",
            "--output",
            str(output),
            *flags,
        ],
    )


@pytest.mark.parametrize("fixture,work", [("milestone", "GEN"), ("container", "EXO")])
def test_markup_literals(tmp_path: Path, fixture: str, work: str) -> None:
    output = tmp_path / "out"
    result = run(FIXTURES / f"{fixture}.xml", output, "--json")
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    passages = [r for r in rows if r["record"] == "passage"]
    first, second = passages
    assert first["text"] == "Māter ait vērum ."
    assert first["source_markup"] == [
        {"element": "wj", "attributes": {}, "span": [6, 15]},
        {"element": "w", "attributes": {"s": "G1", "l": "loquor", "x-morph": "V"}, "span": [10, 15]},
    ]
    assert first["text"][6:15] == "ait vērum"
    assert first["text"][10:15] == "vērum"
    assert first["notes"] == [
        {"kind": "f", "text": "1:3 nota"},
        {"kind": "x", "text": "cross"},
        {"kind": "fe", "text": "end"},
    ]
    assert (first["passage_id"], first["headings"]) == (
        f"test-corpus:{work}.1.3a" if work == "GEN" else f"test-corpus:{work}.2.3a",
        [{"text": "First heading"}],
    )
    assert second["headings"] == [{"text": "Next heading"}]
    assert second["notes"] == [{"kind": "f", "text": "next note"}]
    assert output.read_bytes() == (FIXTURES / f"{fixture}.expected.jsonl").read_bytes()
    assert CliRunner().invoke(app, ["corpus", "validate", str(output)]).exit_code == 0


def test_unknown_text_hash_warning_and_spans(tmp_path: Path) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out"
    source.write_text(
        '<usfx><book id="GEN"/><c n="1"/><v n="1"/>a <zz k="v">Māter</zz><foo>!</foo><zz>?</zz><ve/></usfx>'
    )
    result = run(source, output, "--json")
    assert result.exit_code == 3
    assert not output.exists()
    assert json.loads(result.stdout)["diagnostics"][0]["message"] == "Unknown elements: foo=1, zz=2"
    result = run(source, output, "--allow-unknown", "--json")
    assert result.exit_code == 0, result.output
    diagnostic = json.loads(result.stdout)["diagnostics"][0]
    assert diagnostic["severity"] == "warning"
    passage = json.loads(output.read_text().splitlines()[-1])
    assert passage["text"] == "a Māter!?"
    assert passage["source_extra"] == [
        {"element": "zz", "attributes": {"k": "v"}, "span": [2, 7]},
        {"element": "foo", "attributes": {}, "span": [7, 8]},
        {"element": "zz", "attributes": {}, "span": [8, 9]},
    ]
    assert CliRunner().invoke(app, ["corpus", "validate", str(output)]).exit_code == 0
    text = run(source, output, "--allow-unknown")
    assert "warning unknown_element" in text.stderr


@pytest.mark.parametrize(
    "body,text,spans,slices",
    [
        ("a\u0327<w>\u0301</w>!", "á\u0327!", [[0, 1]], ["á"]),
        ("a<w>\u0327</w>\u0301!", "á\u0327!", [[1, 2]], ["\u0327"]),
        ("A<w> \t </w>B", "A B", [[1, 1]], [""]),
        ("<wj>\U0001f600 <w>e\u0301</w></wj>", "\U0001f600 é", [[0, 3], [2, 3]], ["\U0001f600 é", "é"]),
    ],
)
def test_offsets_across_composition_and_whitespace(
    tmp_path: Path, body: str, text: str, spans: list[list[int]], slices: list[str]
) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out"
    source.write_text('<usfx><book id="GEN"/><c n="1"/><v n="1">' + body + "</v></usfx>")
    result = run(source, output, "--json")
    assert result.exit_code == 0, result.output
    passage = json.loads(output.read_text().splitlines()[-1])
    assert passage["text"] == text
    assert [entry["span"] for entry in passage["source_markup"]] == spans
    assert [text[start:end] for start, end in spans] == slices


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("case", json.loads((FIXTURES / "cases.json").read_text()))
def test_committed_cases(tmp_path: Path, case: str, existing: bool) -> None:
    expected = json.loads((FIXTURES / "cases.json").read_text())[case]
    source = FIXTURES / (case.removesuffix("-allowed") + ".xml")
    output = tmp_path / "out"
    if existing:
        output.write_bytes(b"previous output")
    flags = ["--allow-unknown"] if case.endswith("-allowed") else []
    result = run(source, output, *flags, "--json")
    assert result.exit_code == expected["exit"], result.output
    if expected["exit"]:
        assert json.loads(result.stdout)["diagnostics"][0]["code"] == expected["code"]
        assert output.read_bytes() == b"previous output" if existing else not output.exists()
    else:
        assert output.read_bytes() == (FIXTURES / f"{case}.expected.jsonl").read_bytes()
        assert CliRunner().invoke(app, ["corpus", "validate", str(output)]).exit_code == 0
        if case == "unknown-allowed":
            passage = json.loads(output.read_text().splitlines()[-1])
            assert passage["text"][2:3] == "b"
            assert passage["source_markup"][0]["span"] == [2, 3]
            assert passage["notes"] == [{"kind": "f", "text": "note"}]
            assert (passage["passage_id"], passage["headings"]) == ("test-corpus:GEN.1.1", [{"text": "Heading"}])


@pytest.mark.parametrize("tail", ["<s>orphan</s>", "<f>orphan</f>"])
@pytest.mark.parametrize("ending", ["</usfx>", '<book id="EXO"/><c n="1"/><v n="1">b</v></usfx>'])
def test_pending_content_cannot_escape_work(tmp_path: Path, tail: str, ending: str) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out"
    source.write_text('<usfx><book id="GEN"/><c n="1"/><v n="1">a</v>' + tail + ending)
    result = run(source, output, "--json")
    assert result.exit_code == 3
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "invalid_structure"
    assert not output.exists()


def test_heading_in_verse_attaches_next_without_entering_text(tmp_path: Path) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out"
    source.write_text(
        '<usfx><book id="GEN"/><c n="1"/><v n="1">a<s>Next <zz>heading</zz></s>b</v>'
        '<v n="2">c<f>nested <v>not a verse</v> note</f>d</v></usfx>'
    )
    result = run(source, output, "--json")
    assert result.exit_code == 0, result.output
    first, second = [json.loads(line) for line in output.read_text().splitlines()[-2:]]
    assert (first["text"], first["headings"], second["text"]) == ("ab", None, "cd")
    assert (second["passage_id"], second["headings"]) == ("test-corpus:GEN.1.2", [{"text": "Next heading"}])
    assert second["notes"] == [{"kind": "f", "text": "nested not a verse note"}]


@pytest.mark.parametrize("element", ["wj", "f"])
@pytest.mark.parametrize("depth,code", [(64, None), (65, "depth_exceeded")])
def test_depth_boundary(tmp_path: Path, depth: int, code: str | None, element: str) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out"
    source.write_text(
        '<usfx><book id="GEN"><c n="1"/><v n="1">body '
        + f"<{element}>" * (depth - 3)
        + "a"
        + f"</{element}>" * (depth - 3)
        + "</v></book></usfx>"
    )
    result = run(source, output, "--json")
    assert result.exit_code == (3 if code else 0), result.output
    if code:
        assert json.loads(result.stdout)["diagnostics"][0]["code"] == code
        assert not output.exists()


def test_size_stat_and_streaming(tmp_path: Path) -> None:
    source = tmp_path / "large.xml"
    with source.open("wb") as stream:
        stream.write(b"<usfx>")
        stream.seek(64 * 1024 * 1024)
        stream.write(b" ")
    output = tmp_path / "out"
    result = run(source, output, "--json")
    assert result.exit_code == 3
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "input_too_large"
    # Hide the true size from the preflight, exercising the real streaming limit.
    with source.open("r+b") as stream:
        stream.seek(6)
        block = b" " * 65536
        for _ in range(1024):
            stream.write(block)
    with (
        patch.object(Path, "stat", return_value=type("Stat", (), {"st_size": 1})()),
        pytest.raises(ContractError) as error,
    ):
        build_usfx(source, "test-corpus", "la", None)
    assert error.value.diagnostic.code == "input_too_large"
    assert not output.exists()

    # Exactly 64 MiB is permitted; whitespace is outside the passage, not buffered.
    prefix = b'<usfx><book id="GEN"/><c n="1"/><v n="1">a</v>'
    suffix = b"</usfx>"
    with source.open("wb") as stream:
        stream.write(prefix)
        remaining = 64 * 1024 * 1024 - len(prefix) - len(suffix)
        while remaining:
            chunk = b" " * min(remaining, 65536)
            stream.write(chunk)
            remaining -= len(chunk)
        stream.write(suffix)
    result = run(source, output, "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text().splitlines()[0])["source"]["bytes"] == 64 * 1024 * 1024
    assert json.loads(output.read_text().splitlines()[-1])["text"] == "a"


def test_billion_laughs_subprocess(tmp_path: Path) -> None:
    source, output = tmp_path / "input.xml", tmp_path / "out"
    declarations = ['<!ENTITY a "ENTITY_SECRET">']
    for index in range(1, 10):
        previous = "a" if index == 1 else f"e{index - 1}"
        declarations.append(f'<!ENTITY e{index} "' + f"&{previous};" * 10 + '">')
    source.write_text("<!DOCTYPE usfx [" + "".join(declarations) + "]><usfx>&e9;</usfx>")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "latinitas_cards",
            "corpus",
            "normalize",
            str(source),
            "--adapter",
            "usfx",
            "--corpus-id",
            "test-corpus",
            "--output",
            str(output),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "unsafe_xml"
    assert "ENTITY_SECRET" not in result.stdout + result.stderr
    assert not output.exists()


def test_large_marked_verse_has_bounded_mapping_memory(tmp_path: Path) -> None:
    source = tmp_path / "marked.xml"
    text = b"a" * (2 * 1024 * 1024)
    source.write_bytes(b'<usfx><book id="GEN"/><c n="1"/><v n="1"><w/>' + text + b"</v></usfx>")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, resource, sys
from pathlib import Path
from latinitas_cards.usfx_adapter import build_usfx
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
passage = build_usfx(Path(sys.argv[1]), "test-corpus", "la", None)[-1]
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
scale = 1 if sys.platform == "darwin" else 1024
print(json.dumps({"growth": (after-before)*scale, "hash": passage.passage_hash,
                  "length": len(passage.text), "span": passage.source_markup[0].span}))
""",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["length"] == len(text)
    assert evidence["span"] == [0, 0]
    assert evidence["hash"] == "sha256:" + hashlib.sha256(text).hexdigest()
    assert evidence["growth"] < 64 * 1024 * 1024, evidence
