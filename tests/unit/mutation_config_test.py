"""Keep repository-file assertions runnable inside mutmut's copied project."""

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "required",
    [
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "CHANGELOG.md",
        "THIRD_PARTY_NOTICES.md",
        "docs/extraction-content-review.md",
        "docs/legacy-transition.md",
        "docs/reference-note-type.md",
        "docs/deterministic-csv-export.md",
        "docs/representative-deck-validation.md",
        "docs/principal-part-parsing.md",
        "docs/release-readiness.md",
        "docs/license-compatibility-audit.md",
        "planning/tasks/T-014-publish-v0-1-0.md",
    ],
)
def test_mutation_sandbox_includes_repository_test_dependencies(required: str) -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    path = Path(required)
    copied = [Path(entry) for entry in project["tool"]["mutmut"]["also_copy"]]
    assert any(entry == path or entry in path.parents for entry in copied), f"mutmut does not copy {required}"
    assert (ROOT / path).is_file(), f"missing repository test dependency: {required}"
