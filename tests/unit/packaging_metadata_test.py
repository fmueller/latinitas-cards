"""Distribution contracts without building packages or accessing the network."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
URL = "https://github.com/fmueller/latinitas-cards"


def test_package_metadata_preserves_install_contract() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = config["project"]
    description = project["description"].lower()
    assert re.match(r".*anki.*(?:deck|csv).*principal.part cards", description)
    for secondary in ("corpus", "cloze", "annotate"):
        assert secondary not in description or description.index("principal") < description.index(secondary)
    assert project["urls"] == {
        "Homepage": URL,
        "Repository": URL,
        "Changelog": f"{URL}/blob/main/CHANGELOG.md",
        "Issues": f"{URL}/issues",
    }
    assert {
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    } <= set(project["classifiers"])
    assert not any(value.startswith("License ::") for value in project["classifiers"])
    assert project["license"] == "GPL-3.0-or-later"
    assert project["license-files"] == ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    assert project["requires-python"] == ">=3.13,<3.15"
    runtime_names = {re.split(r"[\s\[<>=!~;]", requirement, maxsplit=1)[0] for requirement in project["dependencies"]}
    assert not {"cltk", "stanza", "torch"} & runtime_names
    for extra in ("annotate", "annotate-gpu"):
        requirements = project["optional-dependencies"][extra]
        assert any(requirement.startswith("cltk[stanza]") for requirement in requirements)
        assert any(requirement.startswith("torch") for requirement in requirements)
    inventory = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    # GHSA-27vj-qcqg-25rc: optional NLP dependencies must not lock vulnerable fsspec.
    fsspec = next(package for package in inventory if package["name"] == "fsspec")
    assert tuple(map(int, fsspec["version"].split("."))) >= (2026, 6, 0)
    historical_audit = ROOT / "docs/license-compatibility-audit-2026-09-25.md"
    assert historical_audit.is_file()
    current_audit = (ROOT / "docs/license-compatibility-audit.md").read_text()
    assert "[historical snapshot](license-compatibility-audit-2026-09-25.md)" in current_audit


def relative_links(markdown: str) -> list[str]:
    targets = re.findall(r"\]\(([^\s)]+)", markdown)
    targets += re.findall(r"^\s*\[[^\]]+\]:\s*(\S+)", markdown, re.MULTILINE)
    return [target for target in targets if not target.startswith(("https://", "mailto:", "#"))]


def test_readme_links_work_on_package_indexes() -> None:
    assert relative_links("[bad](docs/x.md)") == ["docs/x.md"]
    assert relative_links("[bad]: docs/x.md") == ["docs/x.md"]
    assert not relative_links((ROOT / "README.md").read_text())


def test_package_ci_is_build_only_and_guards_are_integrated() -> None:
    lines = (ROOT / ".github/workflows/build.yml").read_text().splitlines()
    start = lines.index("  package:")
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^  [\w-]+:$", lines[i])), len(lines))
    package = lines[start:end]
    assert "    needs: lint-and-typecheck" in package
    assert '        python-version: ["3.13", "3.14"]' in package
    assert "      UV_PYTHON: ${{ matrix.python-version }}" in package
    assert "          python-version: ${{ matrix.python-version }}" in package
    assert "        run: bash scripts/check-dist.sh" in package
    assert '        run: bash scripts/check-install.sh "$(uv python find)"' in package
    assert not any("id-token" in line or "publish" in line.lower() for line in package)
    guards = lines[lines.index("  guards:") : lines.index("  unit-tests:")]
    assert "        run: bash scripts/check-dist-test.sh" in guards
    mise = tomllib.loads((ROOT / "mise.toml").read_text())
    assert "bash scripts/check-dist-test.sh" in mise["tasks"]["check"]["run"]
