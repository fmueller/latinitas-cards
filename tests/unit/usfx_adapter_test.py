"""Contracts for the behavior-preserving legacy USFX extraction."""

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest
from cli_test import _create_colpkg_with_anki2, get_command
from click.testing import CliRunner
from legacy_usfx_reference import parse_usfx_to_df

from latinitas_cards import cli, usfx_adapter

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/goldens/legacy-vulgate"


def test_legacy_projection_is_owned_by_adapter() -> None:
    assert callable(usfx_adapter.legacy_projection)
    tree = ast.parse((ROOT / "src/latinitas_cards/cli.py").read_text())
    assert not any(isinstance(node, ast.FunctionDef) and node.name == "parse_usfx_to_df" for node in ast.walk(tree))
    for module in ("cli", "usfx_adapter"):
        tree = ast.parse((ROOT / f"src/latinitas_cards/{module}.py").read_text())
        assert not any(
            isinstance(node, ast.Import) and any(alias.name.startswith("xml.etree") for alias in node.names)
            for node in ast.walk(tree)
        )


def test_normalize_latin_reexport() -> None:
    assert cli.normalize_latin is usfx_adapter.normalize_latin


def test_differential_legacy_projection(tmp_path: Path) -> None:
    paths = sorted(FIXTURES.rglob("*.xml"))
    assert paths
    # AST constants include Python's implicitly concatenated inline XML strings.
    tree = ast.parse((ROOT / "tests/unit/cli_test.py").read_text())
    inline = sorted(
        {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.lstrip().startswith("<usfx")
        }
    )
    assert inline
    inline.append(
        "<usfx><B code='ZZZ'><chapter n='ch2'/><verse n='3a'> Jæ <f>note</f> tail </verse>"
        "<v n='3'>duplicate</v><v n='4'>  </v></B>"
        "<book n='AAA'><c id='1'/><v>zero<w>nested</w>end</v><v id='2'>final</v></book></usfx>"
    )
    inline += [
        "<usfx xmlns='urn:test'><book id='GEN'><c id='1'/><v id='1'>ignored</v></book></usfx>",
        "<usfx><b code='Z'><chapter n='ch2'/><VERSE n='3a'>a&amp;b<!--skip-->c</VERSE>tail"
        "<ve>included<w>excluded</w></ve><v id='4'>" + "x" * 70000 + "</v></b></usfx>",
    ]
    for index, xml in enumerate(inline):
        path = tmp_path / f"inline-{index}.xml"
        path.write_text(xml, encoding="utf-8")
        paths.append(path)
    for path in paths:
        try:
            expected = parse_usfx_to_df(path)
        except (ET.ParseError, ValueError) as exc:
            with pytest.raises(ValueError) as actual:
                usfx_adapter.legacy_projection(path)
            assert str(actual.value) == str(exc), path
        else:
            assert usfx_adapter.legacy_projection(path).equals(expected), path


@pytest.mark.parametrize("route", ["preview", "generate-csv", "generate-apkg", "cloze-auto", "cloze-usfx", "validate"])
def test_legacy_commands_resolve_adapter_attribute(route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = FIXTURES / "inputs/sample.usfx.xml"
    source = FIXTURES / "inputs/input.csv"
    output = tmp_path / "out.csv"
    if route == "generate-apkg":
        source = tmp_path / "input.apkg"
        output = tmp_path / "out.apkg"
        _create_colpkg_with_anki2(source, [["amo", "I love", ""], ["sum", "I am", ""]])
    if route.startswith("cloze"):
        argv = [
            "cloze",
            "--input",
            str(FIXTURES / "inputs/forms.csv"),
            "--output",
            str(output),
            "--corpus",
            str(corpus),
        ]
        if route == "cloze-usfx":
            argv += ["--corpus-format", "usfx"]
    else:
        command = "generate" if route.startswith("generate") else route
        argv = [command, "--input", str(source), "--usfx", str(corpus)]
        if command == "generate":
            argv += ["--output", str(output)]
    calls: list[Path] = []
    original = usfx_adapter.legacy_projection

    def recording(path: Path) -> pd.DataFrame:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(usfx_adapter, "legacy_projection", recording)
    result = CliRunner().invoke(get_command(cli.app), argv)
    assert result.exit_code == 0, result.output
    assert calls == [corpus]


@pytest.mark.parametrize("route", ["preview", "generate", "generate-apkg", "cloze", "validate"])
@pytest.mark.parametrize("defect", ["doctype", "entity", "depth", "size"])
def test_legacy_xml_safety(route: str, defect: str, tmp_path: Path) -> None:
    corpus = tmp_path / "unsafe.xml"
    xml = "<usfx><book id='GEN'><c id='1'/><v id='1'>amo</v></book></usfx>"
    code = {"doctype": "unsafe_xml", "entity": "unsafe_xml", "depth": "depth_exceeded", "size": "input_too_large"}[
        defect
    ]
    if defect == "doctype":
        xml = "<!DOCTYPE usfx>" + xml
    elif defect == "entity":
        xml = '<!DOCTYPE usfx [<!ENTITY x "amo">]>' + xml
    elif defect == "depth":
        xml = "<x>" * 62 + xml + "</x>" * 62
    corpus.write_text(xml)
    if defect == "size":
        with corpus.open("ab") as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
    output = tmp_path / "out.csv"
    source = FIXTURES / "inputs" / ("forms.csv" if route == "cloze" else "input.csv")
    if route == "generate-apkg":
        source = tmp_path / "input.apkg"
        output = tmp_path / "out.apkg"
        _create_colpkg_with_anki2(source, [["amo", "I love", ""]])
    command = "generate" if route == "generate-apkg" else route
    argv = [command, "--input", str(source), "--corpus" if route == "cloze" else "--usfx", str(corpus)]
    if command in {"generate", "cloze"}:
        argv += ["--output", str(output)]
    result = CliRunner().invoke(get_command(cli.app), argv)
    assert result.exit_code == 3, (result.output, result.exception)
    assert code in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("depth", [4, 64])
def test_legacy_xml_boundary_controls(depth: int, tmp_path: Path) -> None:
    corpus = tmp_path / "boundary.xml"
    xml = "<x>" * (depth - 3) + "<usfx><book id='GEN'><c id='1'/><v id='1'>amo</v></book></usfx>" + "</x>" * (depth - 3)
    # Exactly 64 MiB, with padding outside the document element.
    corpus.write_text(xml)
    with corpus.open("ab") as stream:
        block = b" " * 65536
        remaining = 64 * 1024 * 1024 - len(xml)
        while remaining:
            size = min(remaining, len(block))
            stream.write(block[:size])
            remaining -= size
    assert usfx_adapter.legacy_projection(corpus)["text"].tolist() == ["amo"]
