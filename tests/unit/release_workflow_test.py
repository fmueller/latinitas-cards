"""Offline, line-oriented release contract checks; never invoke publication."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]


def section(lines: list[str], heading: str, indent: int) -> list[str]:
    start = lines.index(" " * indent + heading + ":") + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= indent),
        len(lines),
    )
    return lines[start:end]


def test_release_workflow_contract() -> None:
    lines = (ROOT / ".github/workflows/release.yml").read_text().splitlines()
    trigger = section(lines, "on", 0)
    assert trigger == ["  push:", "    tags:", "      - 'v[0-9]+.[0-9]+.[0-9]+'", ""]
    assert section(lines, "permissions", 0) == ["  contents: read", ""]
    jobs = section(lines, "jobs", 0)
    job_names = ("build", "verify", "publish", "verify-pypi")
    assert [line.strip()[:-1] for line in jobs if re.fullmatch(r"  [\w-]+:", line)] == list(job_names)
    blocks = {name: section(jobs, name, 2) for name in job_names}
    for previous, name in zip(job_names, job_names[1:], strict=False):
        assert f"    needs: {previous}" in blocks[name]
    build = "\n".join(blocks["build"])
    assert "fetch-depth: 0" in build
    assert "tomllib" in build and '["project"]["version"]' in build
    assert '[[ "$GITHUB_REF_NAME" == "v$version" ]]' in build
    assert "git fetch origin main" in build
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in build
    assert "bash scripts/check-dist.sh" in build
    assert 'sha256sum dist/* >> "$GITHUB_STEP_SUMMARY"' in build
    assert "actions/upload-artifact@" in build
    assert "          name: dist" in blocks["build"] and "          path: dist/" in blocks["build"]
    for name in ("verify", "publish"):
        assert any("actions/download-artifact@" in line for line in blocks[name])
        assert "          name: dist" in blocks[name] and "          path: dist/" in blocks[name]
        assert not any("check-dist.sh" in line or "uv build" in line for line in blocks[name])
    verify = "\n".join(blocks["verify"])
    assert 'python-version: ["3.13", "3.14"]' in verify
    assert 'bash scripts/check-install.sh "$(uv python find)"' in verify
    assert "UV_PYTHON: ${{ matrix.python-version }}" in verify
    for name, block in blocks.items():
        assert ("    environment: pypi" in block) == (name == "publish")
        assert any("id-token: write" in line for line in block) == (name == "publish")
    publish = "\n".join(blocks["publish"])
    assert re.search(r"pypa/gh-action-pypi-publish@[0-9a-f]{40} # v1\.\d+\.\d+", publish)
    assert "packages-dir: dist/" in publish
    text = "\n".join(lines)
    for forbidden in (
        "password:",
        "secrets.",
        "PYPI_API_TOKEN",
        "twine",
        "uv publish",
        "repository-url",
        "test.pypi.org",
    ):
        assert forbidden not in text
    pypi = "\n".join(blocks["verify-pypi"])
    retry = [
        '          until timeout 60s uv pip install --no-config --no-cache --python "$python" '
        '--index-url https://pypi.org/simple "latinitas-cards==$VERSION"; do',
        "            (( SECONDS < deadline )) || exit 1",
        "            sleep 10",
        "          done",
    ]
    deadline = blocks["verify-pypi"].index("          deadline=$((SECONDS + 540))")
    assert blocks["verify-pypi"][deadline + 1 : deadline + 1 + len(retry)] == retry
    for expected in (
        "timeout-minutes: 10",
        "uv venv",
        "--help",
        'distribution("cltk")',
        'distribution("stanza")',
        'distribution("torch")',
    ):
        assert expected in pypi


def test_publish_guard_runs_in_local_and_ci_gates() -> None:
    command = "bash scripts/check-publish-guard-test.sh"
    assert command in (ROOT / "mise.toml").read_text()
    build = (ROOT / ".github/workflows/build.yml").read_text().splitlines()
    assert any(command in line for line in section(section(build, "jobs", 0), "guards", 2))
