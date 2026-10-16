import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
WORKFLOW = ROOT / ".github/workflows/release.yml"


def test_installation_guide() -> None:
    readme = (ROOT / "README.md").read_text()
    installation = readme.split("## Installation\n", 1)[1].split("## Quick start\n", 1)[0]
    for required in (
        "uv tool install latinitas-cards",
        "pipx install latinitas-cards",
        "python -m venv .venv",
        "source .venv/bin/activate",
        "pip install latinitas-cards",
        "uv tool upgrade latinitas-cards",
        "pipx upgrade latinitas-cards",
        "pip install --upgrade latinitas-cards",
        "Python 3.13 or 3.14",
        "Linux and macOS are supported; Windows is not supported",
        "only after the future v0.3.0 maintainer release gate",
        "PyPI releases start with v0.3.0",
        "development checkout",
        "git clone --branch v0.3.0 --depth 1 https://github.com/fmueller/latinitas-cards.git",
        "default install needs no CLTK, Stanza or PyTorch",
        "Both `annotate` and `annotate-gpu` are experimental",
        "uv's CPU torch index and extra conflict do not transfer to pip",
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        "two extras must not be combined",
        "`annotate` downloads Stanza models on first run",
        "bundled Vulgate is not in the wheel or sdist",
        "your own USFX path",
        "--descriptor FILE",
    ):
        assert required in installation, required
    documentation = readme.split("## Documentation\n", 1)[1]
    assert "https://github.com/fmueller/latinitas-cards/blob/main/docs/releasing.md" in documentation
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert changelog.index("## [Unreleased]") < changelog.index("## [0.3.0]")
    candidate = changelog.split("## [0.3.0]", 1)[1].split("## [", 1)[0]
    assert "PyPI installation" in candidate


def workflow_names(workflow: str) -> list[str]:
    jobs = workflow.split("\njobs:\n", 1)[1]
    return [
        WORKFLOW.name,
        *re.findall(r"^  ([\w-]+):$", jobs, re.MULTILINE),
        *re.findall(r"^    environment: (\S+)$", jobs, re.MULTILINE),
        *re.findall(r"^      - '([^']+)'$", workflow, re.MULTILINE),
        *re.findall(r"^      - uses: (pypa/[^@]+)@", jobs, re.MULTILINE),
    ]


def assert_workflow_documented(workflow: str, runbook: str) -> None:
    for name in workflow_names(workflow):
        assert f"`{name}`" in runbook, name


def test_release_runbook_matches_workflow() -> None:
    workflow = WORKFLOW.read_text()
    runbook = (ROOT / "docs/releasing.md").read_text()
    assert_workflow_documented(workflow, runbook)
    renamed = workflow.replace("  verify-pypi:\n", "  renamed-verification:\n")
    assert renamed != workflow
    with pytest.raises(AssertionError, match="renamed-verification"):
        assert_workflow_documented(renamed, runbook)


def test_release_runbook_preserves_maintainer_gate() -> None:
    runbook = (ROOT / "docs/releasing.md").read_text()
    for required in (
        "T-038",
        "PyPI project `latinitas-cards`",
        "owner `fmueller`",
        "repository `latinitas-cards`",
        "required reviewer",
        "tag-only deployment",
        "data/lat-clementine.usfx.xml",
        "eBible.org latVUC",
        "docs/license-compatibility-audit.md",
        "descriptor values",
        "Build",
        "CodeQL",
        "version/changelog",
        "exactly one annotated `vX.Y.Z` tag",
        "approve the `pypi` deployment",
        "PyPI file hashes equal the build artifacts",
        "GitHub release",
        "evidence commit",
        "No TestPyPI",
        "API tokens",
        "manual uploads",
        "tag rewrites",
        "blind retries",
        "published version cannot be re-uploaded",
    ):
        assert required in runbook, required
