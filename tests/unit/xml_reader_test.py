"""The hardened XML reader shared by the XML corpus adapters."""

import ast
import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from latinitas_cards.cli_contract import Code, ContractError
from latinitas_cards.xml_reader import XmlEventReader

ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = "<a xmlns='urn:d' xmlns:p='urn:p' p:x='1' y='2'><p:b>t</p:b><c/></a>"
ENCODINGS = pytest.mark.parametrize(
    "encode",
    [
        lambda xml: xml.encode(),
        lambda xml: xml.encode("utf-16"),
        lambda xml: b"\xfe\xff" + xml.encode("utf-16-be"),
    ],
    ids=["utf-8", "utf-16", "utf-16-be"],
)


def write_padded(path: Path, xml: str, size: int) -> None:
    """Pad with whitespace after the document element up to exactly ``size`` bytes."""
    path.write_text(xml)
    with path.open("ab") as stream:
        remaining = size - len(xml)
        while remaining:
            chunk = min(remaining, 1 << 20)
            stream.write(b" " * chunk)
            remaining -= chunk


@pytest.mark.parametrize("namespaces", [False, True])
@pytest.mark.parametrize(
    "defect,code",
    [
        ("doctype", Code.UNSAFE_XML),
        ("entity", Code.UNSAFE_XML),
        ("external", Code.UNSAFE_XML),
        ("depth", Code.DEPTH_EXCEEDED),
        ("size", Code.INPUT_TOO_LARGE),
    ],
)
def test_unsafe_inputs_are_rejected(tmp_path: Path, namespaces: bool, defect: str, code: Code) -> None:
    path = tmp_path / "unsafe.xml"
    xml = "<r xmlns='urn:d'>x</r>"
    if defect == "doctype":
        xml = "<!DOCTYPE r>" + xml
    elif defect == "entity":
        xml = '<!DOCTYPE r [<!ENTITY e "x">]><r>&e;</r>'
    elif defect == "external":
        xml = '<!DOCTYPE r [<!ENTITY e SYSTEM "file:///etc/passwd">]><r>&e;</r>'
    elif defect == "depth":
        xml = "<r>" * 65 + "</r>" * 65
    path.write_text(xml)
    if defect == "size":
        write_padded(path, xml, 64 * 1024 * 1024 + 1)
    reader = XmlEventReader(path, namespaces=namespaces)
    with pytest.raises(ContractError) as error:
        list(reader)
    assert error.value.diagnostic.code == code
    assert reader.source is None


@pytest.mark.parametrize("namespaces", [False, True])
def test_limits_are_inclusive(tmp_path: Path, namespaces: bool) -> None:
    path = tmp_path / "boundary.xml"
    write_padded(path, "<r>" * 64 + "</r>" * 64, 64 * 1024 * 1024)
    reader = XmlEventReader(path, namespaces=namespaces)
    assert sum(kind == "start" for kind, _, _ in reader) == 64
    assert reader.source is not None
    assert reader.source.bytes == 64 * 1024 * 1024
    assert reader.source.sha256 == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_events_keep_raw_names_without_namespace_processing(tmp_path: Path) -> None:
    path = tmp_path / "doc.xml"
    path.write_text(DOCUMENT)
    reader = XmlEventReader(path)
    assert list(reader) == [
        ("start", "a", {"xmlns": "urn:d", "xmlns:p": "urn:p", "p:x": "1", "y": "2"}),
        ("start", "p:b", {}),
        ("text", "t", {}),
        ("end", "p:b", {}),
        ("start", "c", {}),
        ("empty_end", "c", {}),
        ("end", "a", {}),
    ]
    assert reader.source is not None
    assert (reader.source.file_name, reader.source.bytes) == ("doc.xml", len(DOCUMENT))


def test_namespace_processing_uses_expanded_names(tmp_path: Path) -> None:
    path = tmp_path / "doc.xml"
    path.write_text(DOCUMENT.replace("y='2'", "y='2' xml:lang='la'"))
    assert list(XmlEventReader(path, namespaces=True)) == [
        ("start", "{urn:d}a", {"{urn:p}x": "1", "y": "2", "{http://www.w3.org/XML/1998/namespace}lang": "la"}),
        ("start", "{urn:p}b", {}),
        ("text", "t", {}),
        ("end", "{urn:p}b", {}),
        ("start", "{urn:d}c", {}),
        ("empty_end", "{urn:d}c", {}),
        ("end", "{urn:d}a", {}),
    ]


@ENCODINGS
@pytest.mark.parametrize(
    "xml,expected",
    [
        ("<a><c/></a>", ["c:empty_end", "a:end"]),
        ("<a><a/></a>", ["a:empty_end", "a:end"]),
        ("<a><c></c></a>", ["c:end", "a:end"]),
        ("<a><c x='/'><!--/--></c></a>", ["c:end", "a:end"]),
        ("<a><c><d/></c></a>", ["d:empty_end", "c:end", "a:end"]),
        # One start tag spanning several 64 KiB reads.
        (f"<a><c x='{'y' * 200_000}'/></a>", ["c:empty_end", "a:end"]),
        (f"<a><c x='{'y' * 200_000}'></c></a>", ["c:end", "a:end"]),
    ],
    ids=["before-close", "same-name", "explicit", "comment", "child", "long-empty", "long-explicit"],
)
def test_empty_and_explicit_ends(tmp_path: Path, encode: Callable[[str], bytes], xml: str, expected: list[str]) -> None:
    path = tmp_path / "doc.xml"
    path.write_bytes(encode(xml))
    ends = [f"{name}:{kind}" for kind, name, _ in XmlEventReader(path) if kind.endswith("end")]
    assert ends == expected


@ENCODINGS
@pytest.mark.parametrize("opening,closing", [("<!--", "-->"), ("<?pi ", "?>")], ids=["comment", "pi"])
def test_explicit_end_after_markup_split_at_a_read(
    tmp_path: Path, encode: Callable[[str], bytes], opening: str, closing: str
) -> None:
    # A comment or PI emits no event; its '/>' ends exactly at the first 64 KiB read.
    head = f"<a><c>{opening}"
    step = len(encode(head + "y/>")) - len(encode(head + "/>"))
    filler, remainder = divmod(65536 - len(encode(head + "/>")), step)
    assert not remainder
    path = tmp_path / "doc.xml"
    path.write_bytes(encode(f"{head}{'y' * filler}/>{closing}</c></a>"))
    assert path.read_bytes()[:65536].endswith(encode("/>")[-step * 2 :])
    ends = [f"{name}:{kind}" for kind, name, _ in XmlEventReader(path) if kind.endswith("end")]
    assert ends == ["c:end", "a:end"]


def test_only_the_shared_module_creates_expat_parsers() -> None:
    creators = []
    for path in sorted((ROOT / "src/latinitas_cards").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr == "ParserCreate":
                creators.append(path.name)
            elif isinstance(node, ast.ClassDef) and node.name == "_EventReader":
                creators.append(f"{path.name}:_EventReader")
    assert creators == ["xml_reader.py"]
