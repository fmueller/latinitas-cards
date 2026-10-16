"""No hidden prompts, including prompts introduced by future commands."""

import ast
import subprocess
import sys
import time
from pathlib import Path

import pytest
from typer._click.core import Command
from typer.core import TyperGroup, TyperOption
from typer.main import get_command

from latinitas_cards import interaction
from latinitas_cards.cli import app


@pytest.mark.parametrize("stdin_tty,stdout_tty", [(False, False), (False, True), (True, False), (True, True)])
def test_interactive_requires_both_ttys(monkeypatch: pytest.MonkeyPatch, stdin_tty: bool, stdout_tty: bool) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin_tty)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout_tty)
    assert interaction._is_interactive() == (stdin_tty and stdout_tty)


def prompt_calls(source: str) -> tuple[int, list[int]]:
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    calls = 0
    unguarded = []

    def interactive(test: ast.AST) -> bool:
        return any(
            isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "_is_interactive"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "_is_interactive"
            )
            for node in ast.walk(test)
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name not in {
            "typer.prompt",
            "typer.confirm",
            "click.prompt",
            "click.confirm",
            "input",
            "rich.prompt.Prompt.ask",
            "rich.prompt.Confirm.ask",
            "Prompt.ask",
            "Confirm.ask",
        }:
            continue
        calls += 1
        child: ast.AST = node
        guarded = False
        while child in parents:
            parent = parents[child]
            if isinstance(parent, ast.If) and child in parent.body and interactive(parent.test):
                guarded = not isinstance(parent.test, ast.UnaryOp)
            if isinstance(parent, ast.FunctionDef):
                first = parent.body[0]
                if (
                    isinstance(first, ast.If)
                    and isinstance(first.test, ast.UnaryOp)
                    and isinstance(first.test.op, ast.Not)
                    and interactive(first.test)
                    and any(isinstance(stmt, ast.Raise) for stmt in first.body)
                ):
                    guarded = True
            child = parent
        if not guarded:
            unguarded.append(node.lineno)
    return calls, unguarded


def test_all_prompts_are_guarded() -> None:
    total = 0
    violations = {}
    for path in Path("src").rglob("*.py"):
        count, lines = prompt_calls(path.read_text())
        total += count
        if lines:
            violations[str(path)] = lines
    print(f"prompt-call count: {total}")
    assert total > 0
    assert not violations


def test_scanner_negative_control() -> None:
    assert prompt_calls("typer.prompt('unsafe')\ninput('unsafe')") == (2, [1, 2])
    assert prompt_calls("if interaction._is_interactive():\n    click.confirm('safe')") == (1, [])
    assert prompt_calls("def f():\n    if not _is_interactive():\n        raise ValueError()\n    input()") == (1, [])


def test_click_options_never_prompt() -> None:
    def walk(command: Command) -> None:
        for param in command.params:
            if isinstance(param, TyperOption):
                assert not param.prompt
                assert not param.confirmation_prompt
        if isinstance(command, TyperGroup):
            for child in command.commands.values():
                walk(child)

    walk(get_command(app))


@pytest.mark.parametrize("choice,confirm", [(False, False), (True, False), (False, True), (True, True)])
def test_setup_devnull(tmp_path: Path, choice: bool, confirm: bool) -> None:
    source = tmp_path / "source.csv"
    profile = tmp_path / "profile.json"
    source.write_text(
        "Stable ID,Lemma,Other,Forms\n"
        "entry-1,dīcō," + ("dīcō" if choice else "lesson-1") + ',"dīcere,dīcō,dīxī,dictum"\n'
        "entry-2,ferō," + ("ferō" if choice else "lesson-2") + ',"ferre,ferō,tulī,lātum"\n'
    )
    before = source.read_bytes()
    args = ["setup", "--input", str(source), "--profile", str(profile)]
    if confirm:
        args += ["--confirm"]
        if choice:
            args += ["--lexical-entry-field", "Lemma", "--principal-parts-field", "Forms"]
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "latinitas_cards", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
    )
    print(
        f"setup choice={choice} confirm={confirm}: exit={result.returncode}, seconds={time.monotonic() - started:.3f}"
    )
    assert result.returncode == (0 if confirm else 2), result.stderr
    assert profile.exists() == confirm
    assert source.read_bytes() == before
    if not confirm:
        assert "prompt_required" in result.stderr
        assert "--confirm" in result.stderr
        for flag in ["--lexical-entry-field", "--principal-parts-field"] if choice else ["--non-interactive"]:
            assert flag in result.stderr


def test_cloze_devnull(tmp_path: Path) -> None:
    source, output = tmp_path / "forms.csv", tmp_path / "out.csv"
    source.write_text("form\namo\n")
    corpus = Path("tests/fixtures/parallel-prompt.csv").resolve()
    before = corpus.read_bytes()
    started = time.monotonic()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "latinitas_cards",
            "cloze",
            "--input",
            str(source),
            "--output",
            str(output),
            "--corpus",
            str(corpus),
            "--corpus-format",
            "csv",
            "--parallel-mode",
            "prompt",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
    )
    print(f"cloze: exit={result.returncode}, seconds={time.monotonic() - started:.3f}")
    assert result.returncode == 0, result.stderr
    assert "Ignoring translations" in " ".join(result.stderr.split())
    assert "translation_" not in output.read_text().splitlines()[0]
    assert corpus.read_bytes() == before
