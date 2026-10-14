---
id: T-123-document-pypi-installation-extras-and-the
title: Document PyPI installation, extras and the maintainer release runbook
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies:
    - T-122-add-the-tag-only-trusted-publishing-release
updated_at: "2026-10-14T05:42:59Z"
---

# T-123-document-pypi-installation-extras-and-the Document PyPI installation, extras and the maintainer release runbook

## Description

Documents installation from PyPI and the maintainer release procedure, implementing the
Docs bullet and the runbook clause of `specs/v0.3.0.md#pypi-publishing`. Today the
README `## Installation` section only covers a `git clone --branch v0.2.1` checkout with
`uv sync --locked`, documents the `annotate` extras through uv only, and says PyPI
publishing remains deferred; no runbook exists. This task writes the user install guide
and `docs/releasing.md` against the names in `release.yml` (T-122). The
"Current release" wording and the 0.3.0 version belong to T-124; the
release itself to T-038. No `pyproject.toml` or extra changes.

## Acceptance

- README `## Installation` documents `uv tool install latinitas-cards`, `pipx install
  latinitas-cards`, and `python -m venv` plus `pip install latinitas-cards` inside the
  venv; upgrades via `uv tool upgrade latinitas-cards`, `pipx upgrade latinitas-cards`
  and `pip install --upgrade latinitas-cards`; supported Python 3.13 and 3.14; Linux and
  macOS supported, Windows not; the git checkout path kept as the development install.
  A pytest test asserts each exact command string and statement.
- The extras subsection keeps both `annotate` and `annotate-gpu` as experimental, states
  that uv's CPU torch index and the extra conflict do not transfer to pip, gives
  `--extra-index-url https://download.pytorch.org/whl/cpu` for a CPU install with pip,
  says the two extras must not be combined, and says `annotate` downloads Stanza models on
  first run; the test asserts each phrase, and that the default install needs no CLTK,
  Stanza or PyTorch.
- The guide states that PyPI releases start with v0.3.0 and that the bundled Vulgate is
  not in the wheel or sdist: installed users pass their own USFX path and, optionally,
  `--descriptor FILE`. The test asserts both statements.
- `docs/releasing.md` lists maintainer prerequisites (PyPI project `latinitas-cards`,
  pending Trusted Publisher owner `fmueller`, repository `latinitas-cards`, workflow
  `release.yml`, environment `pypi`; GitHub environment `pypi` with a required reviewer
  and tag-only deployment) and the steps: candidate Build and CodeQL (where enabled)
  green, version/changelog ready, one annotated `vX.Y.Z` tag pushed, `build` and `verify`
  pass, approve the `pypi` deployment, `publish`, `verify-pypi`, PyPI file hashes equal
  the build artifacts, GitHub release, evidence commit. It forbids TestPyPI, API tokens,
  manual uploads, tag rewrites and blind retries, and says a published version cannot be
  re-uploaded.
- A pytest test extracts from `release.yml` text the workflow file name, every job id,
  the environment name, the tag pattern and the publish action repository, and asserts
  each appears verbatim in `docs/releasing.md`; a negative control with a renamed job in
  an in-memory copy of `release.yml` makes the comparison fail.
- README stays free of relative links (the T-121 test still passes) and its
  Documentation list links `docs/releasing.md` by absolute URL; `[Unreleased]` in
  `CHANGELOG.md` gains one short user-focused bullet for PyPI installation.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record that `docs/releasing.md` names were checked against the committed `release.yml`.

## Implementation Notes

- Tests in `tests/unit/install_docs_test.py`. Extract job ids with a line-oriented parse
  (two-space-indented keys under `jobs:`); add no YAML dependency. Reuse the README
  link helper from T-121 if it lives in a shared test module, else duplicate it
  minimally.
- Write docs first as failing assertions (RED), then the prose (GREEN). Keep the README
  install section short; put the full runbook only in `docs/releasing.md`.
- Do not change the README "Current release" line, the `git clone --branch v0.2.1`
  command or `release_candidate_test.py`; T-124 owns them.
- The PyPI commands work only after the gate publishes 0.3.0; say so instead of implying
  current availability. Do not document unverified macOS GPU or Windows behaviour.
- `docs/` is already in `[tool.mutmut].also_copy`; `release.yml` is covered by the
  `.github/workflows/` entry from T-121.
- Size guard: README section, one runbook and one test module. If docs for other
  commands need rework, file a follow-up rather than widen.
