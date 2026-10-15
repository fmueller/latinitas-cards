"""Executable envelope exits and extensible real-command coverage matrix."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import pytest
import typer
from pydantic import JsonValue
from typer.core import TyperGroup
from typer.main import get_command
from typer.testing import CliRunner

from latinitas_cards.cli import app as real_app
from latinitas_cards.cli_contract import CliResult, Code, ContractError, Diagnostic, run_contract

app = typer.Typer()


@app.command()
def exercise(
    kind: Annotated[int, typer.Option()],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def body() -> tuple[JsonValue, list[Diagnostic]]:
        print("accidental callback output")
        if kind == 70:
            raise RuntimeError("secret internal details")
        if kind == 2:
            raise ContractError(Code.USAGE_ERROR, "bad flags")
        codes = {1: Code.STALE, 3: Code.UNREADABLE, 4: Code.BUSY, 5: Code.DENIED}
        diagnostics = [Diagnostic(code=codes[kind], message="bad\x1b[31m\nline")] if kind else []
        return {"value": "safe\x1b[31m"}, diagnostics

    run_contract("test exercise", json_output, body)


@pytest.mark.parametrize("exit_code", [0, 1, 2, 3, 4, 5, 70])
def test_envelope_exit_classes(exit_code: int) -> None:
    result = CliRunner().invoke(app, ["--kind", str(exit_code), "--json"])
    assert result.exit_code == exit_code, result.output
    envelope = CliResult.model_validate_json(result.stdout)
    assert envelope.exit_code == exit_code
    assert envelope.ok == (exit_code == 0)
    assert "secret internal details" not in result.stdout
    assert result.stderr == ""
    print(result.stdout.strip())


def test_click_parse_error_has_no_envelope() -> None:
    result = CliRunner().invoke(app, ["--json"])
    assert result.exit_code == 2
    assert result.stdout == ""


def test_text_output_escapes_controls() -> None:
    result = CliRunner().invoke(app, ["--kind", "1"])
    assert result.exit_code == 1
    assert "value" in result.stdout
    assert "stale" in result.stderr
    assert "\\x1b" in result.stderr
    assert "\x1b" not in result.stdout + result.stderr


# Later contract tasks add executable cases indexed by (command, callback exit).
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/corpus"
CONTRACT_EXIT_CASES: dict[tuple[str, int], Callable[[Path], list[str]]] = {
    ("corpus validate", 0): lambda _: ["corpus", "validate", str(FIXTURES / "valid/ordinal.jsonl")],
    ("corpus validate", 1): lambda _: ["corpus", "validate", str(FIXTURES / "invalid/hash.jsonl")],
    ("corpus validate", 3): lambda tmp: ["corpus", "validate", str(tmp / "missing.jsonl")],
}


def documented_pairs() -> set[tuple[str, int]]:
    text = (Path(__file__).resolve().parents[2] / "docs/agent-cli-contract.md").read_text()
    table = text.split("| command | exit codes |\n| --- | --- |\n", 1)[1].split("\n\n", 1)[0]
    pairs: set[tuple[str, int]] = set()
    for row in table.splitlines():
        if row.startswith("|"):
            _, command, exits, _ = row.split("|")
            pairs.update((command.strip(), int(exit_code.strip())) for exit_code in exits.split(","))
    return pairs


def test_matrix_covers_registered_contract_commands() -> None:
    commands: set[str] = set()

    def walk(group: TyperGroup, prefix: str) -> None:
        for name, command in group.commands.items():
            full = f"{prefix} {name}".strip()
            if isinstance(command, TyperGroup):
                walk(command, full)
            else:
                commands.add(full)

    root = get_command(real_app)
    assert isinstance(root, TyperGroup)
    for name in ("corpus", "annotations", "policy"):
        if name in root.commands:
            group = root.commands[name]
            assert isinstance(group, TyperGroup)
            walk(group, name)
    pairs = documented_pairs()
    assert {command for command, _ in pairs} == commands
    assert pairs == set(CONTRACT_EXIT_CASES)
    assert all(exit_code != 70 for _, exit_code in pairs)


@pytest.mark.parametrize("pair", CONTRACT_EXIT_CASES)
def test_contract_exit_case(pair: tuple[str, int], tmp_path: Path) -> None:
    result = CliRunner().invoke(real_app, [*CONTRACT_EXIT_CASES[pair](tmp_path), "--json"])
    envelope = CliResult.model_validate_json(result.stdout)
    assert envelope.command == pair[0]
    assert envelope.exit_code == result.exit_code == pair[1]
