"""Executable envelope exits and extensible real-command coverage matrix."""

from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Annotated, cast

import click
import pytest
import typer
from click.testing import CliRunner as ClickRunner
from pydantic import JsonValue
from typer.core import TyperGroup
from typer.main import get_command
from typer.testing import CliRunner

from latinitas_cards.annotation_plans import AnnotationPlan, approve_plan, create_plan
from latinitas_cards.atomic_io import writer_lock
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


def normalize_case(tmp: Path, exit_code: int) -> list[str]:
    source, output = tmp / "source.txt", tmp / "out.jsonl"
    source.write_text("arma\n")
    args = [
        "corpus",
        "normalize",
        str(source),
        "--adapter",
        "text",
        "--corpus-id",
        "test-corpus",
        "--output",
        str(output),
    ]
    if exit_code in (1, 4):
        assert CliRunner().invoke(real_app, args).exit_code == 0
        if exit_code == 1:
            source.write_text("edited\n")
        else:
            output.write_text("foreign corpus bytes")
    if exit_code == 2:
        args += ["--language", "bad_tag"]
    elif exit_code == 3:
        source.unlink()
    return args


def annotations_case(tmp: Path, exit_code: int) -> list[str]:
    corpus = FIXTURES / ("invalid/hash.jsonl" if exit_code == 3 else "valid/ordinal.jsonl")
    args = ["annotations", "segment", str(corpus), "--output", str(tmp / "out.jsonl")]
    return args + (["--timestamp", "not-a-date"] if exit_code == 2 else ["--timestamp", "2026-01-01T00:00:00Z"])


def annotation_plan_case(tmp: Path, exit_code: int) -> list[str]:
    patch = FIXTURES.parent / "annotation-plans" / ("invalid.patch.json" if exit_code == 1 else "add.patch.json")
    return [
        "annotations",
        "plan",
        str(patch),
        "--corpus",
        str(FIXTURES / "valid/ordinal.jsonl"),
        "--sidecar",
        str(tmp / "missing" if exit_code == 3 else FIXTURES.parent / "annotations/ordinal.annotations.jsonl"),
        "--output",
        str(tmp / "missing" / "plan.json" if exit_code == 3 else tmp / "out.jsonl"),
    ]


def annotation_approve_case(tmp: Path, exit_code: int) -> list[str]:
    plan = FIXTURES.parent / "annotation-plans/add.plan.json"
    operation_id = AnnotationPlan.model_validate_json(plan.read_bytes()).operations[0].operation_id
    return [
        "annotations",
        "approve",
        str(plan),
        "--operation",
        "unknown" if exit_code == 3 else operation_id,
        "--actor",
        "agent:test" if exit_code == 5 else "human:test",
        "--review",
        "checked",
        "--output",
        str(tmp / "out.jsonl"),
    ]


def annotation_apply_case(tmp: Path, exit_code: int) -> list[str]:
    corpus = tmp / "corpus.jsonl"
    corpus.write_bytes((FIXTURES / "valid/ordinal.jsonl").read_bytes())
    sidecar, plan = tmp / "out.jsonl", tmp / "plan.json"
    create_plan(FIXTURES.parent / "annotation-plans/add.patch.json", corpus, sidecar, None, plan)
    args = ["annotations", "apply", str(plan), "--corpus", str(corpus), "--sidecar", str(sidecar)]
    if exit_code != 2:
        args += ["--timestamp", "2026-01-01T00:00:00Z"]
    if exit_code == 3:
        plan.unlink()
    if exit_code == 5:
        receipt = tmp / "receipt.json"
        model = AnnotationPlan.model_validate_json(plan.read_bytes())
        approve_plan(plan, [model.operations[0].operation_id], "human:test", "checked", receipt)
        raw = receipt.read_text().replace("human:test", "agent:test")
        receipt.write_text(raw)
        args += ["--receipt", str(receipt)]
    return args


def import_translations_case(tmp: Path, exit_code: int) -> list[str]:
    corpus = FIXTURES / "valid/ordinal.jsonl"
    parallel = tmp / "parallel.jsonl"
    # The same corpus relabelled German shares every native_ref with the Latin original.
    header, *rows = corpus.read_text().splitlines()
    parallel.write_text("\n".join([header.replace('"language":"la"', '"language":"de"'), *rows]) + "\n")
    return [
        "annotations",
        "import-translations",
        str(corpus),
        "--parallel",
        str(corpus if exit_code == 3 else parallel),
        "--language",
        "de",
        "--producer-kind",
        "human",
        "--producer-name",
        "fixture",
        "--producer-version",
        "1",
        "--output",
        str(tmp / "out.jsonl"),
        *(["--timestamp", "not-a-date"] if exit_code == 2 else []),
    ]


CONTRACT_EXIT_CASES: dict[tuple[str, int], Callable[[Path], list[str]]] = {
    **{
        ("annotations import-translations", code): partial(import_translations_case, exit_code=code)
        for code in (0, 2, 3, 4)
    },
    ("corpus show", 2): lambda tmp: [
        "corpus",
        "show",
        str(FIXTURES / "wsl-three.corpus.jsonl"),
        "--claims",
        str(tmp / "claims.json"),
    ],
    ("corpus validate", 2): lambda tmp: [
        "corpus",
        "validate",
        str(FIXTURES / "wsl-three.corpus.jsonl"),
        "--claims",
        str(tmp / "claims.json"),
    ],
    ("corpus config", 0): lambda _: ["corpus", "config"],
    ("corpus config", 3): lambda tmp: ["corpus", "config", "--corpus", str(tmp / "missing.jsonl")],
    **{("annotations plan", code): partial(annotation_plan_case, exit_code=code) for code in (0, 1, 3, 4)},
    **{("annotations approve", code): partial(annotation_approve_case, exit_code=code) for code in (0, 3, 4, 5)},
    **{("annotations apply", code): partial(annotation_apply_case, exit_code=code) for code in (0, 2, 3, 4, 5)},
    ("annotations inspect", 0): lambda _: [
        "annotations",
        "inspect",
        str(FIXTURES.parent / "annotations/wsl.annotations.jsonl"),
        "--corpus",
        str(FIXTURES / "wsl-three.corpus.jsonl"),
    ],
    ("annotations inspect", 1): lambda _: [
        "annotations",
        "inspect",
        str(FIXTURES.parent / "annotations/wsl.annotations.jsonl"),
        "--corpus",
        str(FIXTURES / "wsl-three-edited.corpus.jsonl"),
        "--check",
    ],
    ("annotations inspect", 3): lambda tmp: [
        "annotations",
        "inspect",
        str(FIXTURES.parent / "annotations/wsl.annotations.jsonl"),
        "--corpus",
        str(FIXTURES / "wsl-three.corpus.jsonl"),
        "--claims",
        str(tmp / "missing.claims.json"),
    ],
    ("corpus show", 0): lambda _: ["corpus", "show", str(FIXTURES / "wsl-three.corpus.jsonl")],
    ("corpus show", 3): lambda _: [
        "corpus",
        "show",
        str(FIXTURES / "wsl-three.corpus.jsonl"),
        "--passage",
        "wsl:missing",
    ],
    ("annotations segment", 0): lambda tmp: annotations_case(tmp, 0),
    ("annotations segment", 2): lambda tmp: annotations_case(tmp, 2),
    ("annotations segment", 3): lambda tmp: annotations_case(tmp, 3),
    ("annotations segment", 4): lambda tmp: annotations_case(tmp, 4),
    ("corpus validate", 0): lambda _: ["corpus", "validate", str(FIXTURES / "valid/ordinal.jsonl")],
    ("corpus validate", 1): lambda _: ["corpus", "validate", str(FIXTURES / "invalid/hash.jsonl")],
    ("corpus validate", 3): lambda tmp: ["corpus", "validate", str(tmp / "missing.jsonl")],
    ("corpus normalize", 0): lambda tmp: normalize_case(tmp, 0),
    ("corpus normalize", 1): lambda tmp: normalize_case(tmp, 1),
    ("corpus normalize", 2): lambda tmp: normalize_case(tmp, 2),
    ("corpus normalize", 3): lambda tmp: normalize_case(tmp, 3),
    ("corpus normalize", 4): lambda tmp: normalize_case(tmp, 4),
    ("policy validate", 0): lambda _: ["policy", "validate", str(FIXTURES.parent / "policy/first-match.json")],
    ("policy validate", 3): lambda tmp: ["policy", "validate", str(tmp / "missing-policy.json")],
    ("corpus-cards profile validate", 0): lambda _: [
        "corpus-cards",
        "profile",
        "validate",
        str(FIXTURES.parents[2] / "docs/examples/corpus-profile.json"),
    ],
    ("corpus-cards profile validate", 3): lambda tmp: ["corpus-cards", "profile", "validate", str(tmp / "absent.json")],
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
    for name in ("corpus", "annotations", "policy", "corpus-cards"):
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
    class NoReadInput(BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size == 0, "Contract command attempted to read stdin"
            return b""

        def read1(self, size: int | None = -1) -> bytes:
            raise AssertionError("Contract command attempted to read stdin")

        def readline(self, size: int | None = -1) -> bytes:
            raise AssertionError("Contract command attempted to read stdin")

    with writer_lock(tmp_path / "out.jsonl") if pair[1] == 4 and pair[0].startswith("annotations ") else nullcontext():
        result = ClickRunner().invoke(
            cast(click.Command, get_command(real_app)),
            [*CONTRACT_EXIT_CASES[pair](tmp_path), "--json"],
            input=NoReadInput(),
        )
    envelope = CliResult.model_validate_json(result.stdout)
    assert envelope.command == pair[0]
    assert envelope.exit_code == result.exit_code == pair[1]


@pytest.mark.parametrize("exit_code", [0, 2, 3, 4])
def test_usfx_normalize_exit_matrix(exit_code: int, tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    source = FIXTURES / "usfx/gen-1-1-5.usfx.xml"
    if exit_code == 3:
        source = FIXTURES / "usfx/duplicate.usfx.xml"
    args = [
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
    ]
    if exit_code == 2:
        args += ["--manifest", str(tmp_path / "unused")]
    if exit_code == 4:
        with writer_lock(output):
            result = CliRunner().invoke(real_app, args)
    else:
        result = CliRunner().invoke(real_app, args)
    envelope = CliResult.model_validate_json(result.stdout)
    assert envelope.command == "corpus normalize"
    assert envelope.exit_code == result.exit_code == exit_code
    assert output.exists() == (exit_code == 0)
