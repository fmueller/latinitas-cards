"""Distribution contracts without building packages or accessing the network."""

import hashlib
import json
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
    assert project["dependencies"] == [
        "pydantic (>=2.11.9,<3.0.0)",
        "pandas (>=2.3.3,<3.0.0)",
        "rich>=13.9.4,<16.0.0",
        "click>=8.1,<8.5",
        "typer>=0.19.2,<0.28.0",
        "zstandard (>=0.25.0,<0.26.0)",
    ]
    assert project["optional-dependencies"] == {
        extra: ["cltk[stanza] (>=2.5.1,<3.0.0)", "torch>=2.9"] for extra in ("annotate", "annotate-gpu")
    }
    inventory = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    assert hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest() == (
        "901378ad9589536b0c6c4a1f916e3f750e6b236354e2bbf7113b96c4997e1385"
    )


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
