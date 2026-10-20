"""Public CLI contract, independent of the legacy diagnostic APIs."""

import ast
from pathlib import Path

import pytest
from source_guard_helpers import active_source_tree

from latinitas_cards.cli_contract import DIAGNOSTIC_CODES, Code

ROOT = Path(__file__).resolve().parents[2]


def test_registry_matches_documentation() -> None:
    rows = (ROOT / "docs/agent-cli-contract.md").read_text().splitlines()
    documented = {row.split("|")[1].strip(" `"): int(row.split("|")[2]) for row in rows if row.startswith("| `")}
    assert set(Code) == set(DIAGNOSTIC_CODES) == set(documented)
    assert {code: entry.exit_code for code, entry in DIAGNOSTIC_CODES.items()} == documented
    meanings = [entry.meaning for entry in DIAGNOSTIC_CODES.values()]
    assert len(meanings) == len(set(meanings))


def literal_contract_codes(source: str, *, contract_module: bool = False, module: str = "") -> list[int]:
    """Check contract constructors, without changing legacy Diagnostic APIs."""
    tree = active_source_tree(source, module=module)
    names = {"Diagnostic", "ContractError"}
    constructors = set(names) if contract_module else set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.endswith("cli_contract"):
                constructors.update(alias.asname or alias.name for alias in node.names if alias.name in names)
                if any(alias.name == "*" for alias in node.names):
                    constructors.update(names)
            for alias in node.names:
                if alias.name == "cli_contract":
                    constructors.update(f"{alias.asname or alias.name}.{name}" for name in names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith(".cli_contract"):
                    constructors.update(f"{alias.asname or alias.name}.{name}" for name in names)
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if ast.unparse(node.func) not in constructors:
            continue
        codes = list(node.args[:1]) + [kw.value for kw in node.keywords if kw.arg == "code"]
        if any(isinstance(code, ast.Constant) and isinstance(code.value, str) for code in codes):
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize(
    "source",
    [
        'from x.cli_contract import Diagnostic\nDiagnostic(code="busy")',
        'import latinitas_cards.cli_contract as contract\ncontract.Diagnostic(code="busy")',
        'from latinitas_cards.cli_contract import ContractError as Failure\nFailure("busy", "message")',
        'from latinitas_cards import cli_contract as contract\ncontract.ContractError("busy", "message")',
        'import latinitas_cards.cli_contract\nlatinitas_cards.cli_contract.Diagnostic(code="busy")',
        'from latinitas_cards.cli_contract import *\nDiagnostic(code="busy")',
    ],
)
def test_literal_code_guard_negative_controls(source: str) -> None:
    assert literal_contract_codes(source) == [2]


def test_literal_code_guard_ignores_inactive_mutant_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    source = (
        "from x.cli_contract import Diagnostic\n"
        "def x_result__mutmut_orig():\n"
        "    return Diagnostic(code=Code.BUSY)\n"
        "def x_result__mutmut_1():\n"
        "    return Diagnostic(code='busy')\n"
    )
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
    assert literal_contract_codes(source) == []
    assert literal_contract_codes(source.replace("Code.BUSY", "'busy'")) == [3]
    monkeypatch.setenv("MUTANT_UNDER_TEST", "other.x_result__mutmut_1")
    assert literal_contract_codes(source, module="example") == []
    monkeypatch.setenv("MUTANT_UNDER_TEST", "example.x_result__mutmut_1")
    assert literal_contract_codes(source, module="example") == [5]
    monkeypatch.delenv("MUTANT_UNDER_TEST")
    assert literal_contract_codes(source) == [5]


def test_contract_codes_are_enum_members() -> None:
    for path in (ROOT / "src/latinitas_cards").rglob("*.py"):
        module = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
        assert not literal_contract_codes(
            path.read_text(), contract_module=path.name == "cli_contract.py", module=module
        ), path


def test_strict_envelope() -> None:
    import json

    from pydantic import ValidationError

    from latinitas_cards.cli_contract import CliResult

    envelope = CliResult(command="test", ok=True, exit_code=0, diagnostics=[], result=None).model_dump(mode="json")
    assert CliResult.model_validate_json(json.dumps(envelope)).ok
    patches: list[dict[str, object]] = [
        {"version": 1.0},
        {"version": True},
        {"extra": 1},
        {"ok": False},
        {"exit_code": 6},
        {"diagnostics": [{"code": "busy", "severity": "info", "message": "x"}]},
        {"diagnostics": [{"code": "unregistered", "severity": "error", "message": "x"}]},
        {"diagnostics": [{"code": "busy", "severity": "error", "message": "x", "extra": 1}]},
    ]
    for patch in patches:
        with pytest.raises(ValidationError):
            CliResult.model_validate_json(json.dumps(envelope | patch))


@pytest.mark.parametrize(
    "codes,expected",
    [
        ([Code.STALE, Code.DENIED], 5),
        ([Code.DENIED, Code.BUSY], 4),
        ([Code.BUSY, Code.EXISTS], 3),
        ([Code.STALE], 1),
        ([], 0),
    ],
)
def test_exit_precedence(codes: list[Code], expected: int) -> None:
    from latinitas_cards.cli_contract import Diagnostic, exit_code_for

    assert exit_code_for([Diagnostic(code=code, message="test") for code in codes]) == expected
    assert exit_code_for([Diagnostic(code=Code.DENIED, severity="warning", message="test")]) == 0
