"""Byte contracts captured before the v0.3.0 legacy adapter refactors."""

import json
import os
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "tests/fixtures/goldens"


@pytest.mark.parametrize("manifest", sorted(GOLDENS.glob("*/cases.json")), ids=lambda path: path.parent.name)
def test_legacy_characterization_goldens(manifest: Path) -> None:
    harness = runpy.run_path(str(ROOT / "scripts/regen-legacy-goldens.py"))
    cases = json.loads(manifest.read_bytes())
    for case in cases:
        if case.get("full_vulgate") and os.environ.get("MUTANT_UNDER_TEST"):
            continue
        harness["legacy_case"](case, capture=False, directory=manifest.parent)


def test_legacy_audit_rejects_existing_input_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = runpy.run_path(str(ROOT / "scripts/regen-legacy-goldens.py"))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "input.csv").write_bytes(b"original input")
    (tmp_path / "empty.stdout").write_bytes(b"")

    def overwrite(argv: list[str], work: Path, expected: int = 0) -> bytes:
        (work / "input.csv").write_bytes(b"changed input")
        return b""

    namespace = harness["legacy_case"].__globals__
    monkeypatch.setitem(namespace, "LEGACY", tmp_path)
    monkeypatch.setitem(namespace, "invoke", overwrite)
    with pytest.raises(AssertionError, match="unexpected writes"):
        harness["legacy_case"](
            {
                "name": "overwrite",
                "argv": [],
                "exit_code": 0,
                "stdout": "empty.stdout",
                "outputs": {},
                "exclusions": {},
            },
            capture=False,
        )


def test_subprocess_uses_mutation_sandbox_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = runpy.run_path(str(ROOT / "scripts/regen-legacy-goldens.py"))
    package = tmp_path / "mutants/src/latinitas_cards"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text('print("mutation sandbox sentinel")\n')
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("MUTANT_UNDER_TEST", "latinitas_cards.sentinel__mutmut_1")
    monkeypatch.setitem(harness["invoke"].__globals__, "ROOT", tmp_path / "mutants")
    assert harness["invoke"](["--help"], work) == b"mutation sandbox sentinel\n"


@pytest.mark.parametrize("flag", [[], ["--i-know-this-rewrites-characterization-goldens"]])
def test_capture_requires_explicit_consent_and_original_head(flag: list[str]) -> None:
    import subprocess
    import sys

    # In a mutation sandbox git metadata need not be copied.
    if os.environ.get("MUTANT_UNDER_TEST"):
        pytest.skip("capture guard needs the original git checkout")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/regen-legacy-goldens.py"), *flag, "--check-only"],
        cwd=ROOT,
        capture_output=True,
    )
    if not flag:
        assert result.returncode != 0
    else:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        capture = json.loads((GOLDENS / "CAPTURE").read_bytes())
        assert (result.returncode == 0) == (head == capture["commit"])
