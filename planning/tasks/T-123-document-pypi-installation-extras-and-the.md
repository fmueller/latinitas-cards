---
id: T-123-document-pypi-installation-extras-and-the
title: Document PyPI installation, extras and the maintainer release runbook
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies:
    - T-122-add-the-tag-only-trusted-publishing-release
updated_at: "2026-10-16T14:09:02Z"
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

## Workflow-v3 cycle audit (2026-10-16)

1. Understand/source: fetched origin/main before lifecycle writes. HEAD and
   origin/main both equaled accepted cumulative base
   `6a56cf742205c30fb41e257a8e61aeaeff6f6f26`; worktree was clean. Active spec was
   `specs/v0.3.0.md`, SHA256
   `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
   `mise exec -- taskrail validate` returned `state valid`; selector returned this
   task, todo, with zero active. Started only T-123 using Taskrail. Read the task,
   PyPI Publishing contract and relevant README/workflow/packaging settings.
   Scope: README installation and absolute documentation link, one runbook,
   one test module and one user-focused Unreleased bullet; no product changes.
2. Strict TDD: added `tests/unit/install_docs_test.py` before prose.
   `uv run pytest -q tests/unit/install_docs_test.py` returned 3 failed:
   missing `uv tool install latinitas-cards` and missing runbook (two tests).
   After the minimal documentation additions, docs plus unchanged packaging
   metadata/absolute-link tests returned `6 passed`.
3. Initial exact chain: `uv run ruff check` returned `All checks passed!`;
   `uv run mypy` returned no issues in 169 files; `uv run pytest -v` returned
   `2455 passed, 1 warning in 141.14s`. The warning is Pydantic serialization,
   not a failed test. No failing gate was skipped or suppressed.
4. Dedicated simplifier Task loaded personal `code-simplifier`, its
   `ATTRIBUTION.md` and license. Result: "No edits made." It found no concrete
   safe reduction of the contract-focused extraction, negative control or gate
   safeguards. Post-simplification focused docs/metadata tests: `6 passed`.
5. Separate parallel independent review Tasks loaded `code-reviewer` and the
   reviewer index. General loaded ECC code-reviewer and common rule sets;
   Python loaded python-reviewer and python-patterns; Security loaded
   security-reviewer, security-review and common security/code-review rules.
   General was mandatory; Python covered test parsing/coverage; Security covered
   package-index trust and publishing instructions. Database, frameworks, ML,
   network configuration and other languages were omitted: no runtime, schema,
   model, infrastructure or framework changes. Within specialist budget.
   General: "No concrete task-relevant findings."
   Security: "No concrete task-relevant findings."
   General also ran docs/metadata/release-candidate tests: `9 passed`;
   Python ran docs/metadata: `6 passed`.
6. Python candidate below is preserved verbatim. A fresh candidate-validation
   Task loaded code-reviewer and Python references and rejected PY-1: the actual
   acceptance requires the publish action repository, not every action. The test
   extracts `pypa/gh-action-pypi-publish` and the runbook documents it. No accepted
   findings, deferrals or finding-fix TDD were applicable. Candidate validator
   ran focused docs tests: `3 passed`.
7. Final exact chain: `uv run ruff check` passed; `uv run mypy` passed (169 files);
   `uv run pytest -v` returned `2455 passed, 1 warning in 131.42s`.
   `mise run check` exited 0, including format, mypy, all 2455 tests and commit,
   push-message, author, mutation, distribution and publishing synthetic guards.
   A fresh disposition-verification Task loaded code-reviewer plus General,
   Python and Security references. It independently confirmed PY-1 rejected and
   returned "No unresolved or new task-relevant findings." Focused tests:
   `3 passed`. One review cycle, no fixes needed. `git diff --check` passed.
8. Finalization: all review/disposition/check gates preceded Taskrail verify and
   complete. Hooks installed with `mise exec -- lefthook install`; git identity
   confirmed Felix Müller. No attribution trailers. Taskrail verification and
   completion timestamps below are authoritative lifecycle evidence.

### Verbatim review candidate and disposition

FINDING PY-1 — tests
Severity: medium
Evidence: `tests/unit/install_docs_test.py:52` restricts action extraction to `pypa/[^@]+`; `.github/workflows/release.yml` also uses `actions/checkout`, `astral-sh/setup-uv`, `actions/setup-python`, and `actions/upload-artifact` / `actions/download-artifact`.
Finding: The workflow/runbook test does not check every action repository named by the acceptance criteria.
Failure/impact: Renaming or changing any non-PyPA action repository would leave this test green, and the current runbook’s omission of those repository names is not detected.
Recommended direction: Extract repository identifiers from every `uses:` line in the workflow and assert each is documented; keep the in-memory renamed-job negative control.

Disposition: rejected (not an acceptance requirement), independently confirmed
by candidate validation and final disposition verification. No code change.

### Consistency and limits

- Checked names against committed `release.yml`: file name; `build`, `verify`,
  `publish`, `verify-pypi`; `pypi`; `v[0-9]+.[0-9]+.[0-9]+`;
  `pypa/gh-action-pypi-publish`. In-memory renamed-job control proves drift fails.
- Runbook retains maintainer prerequisites including corpus provenance/descriptor
  confirmation, candidate Build/applicable CodeQL, protected environment review,
  artifact hashes, ordered approval/publication/verification/release/evidence,
  no TestPyPI/tokens/manual upload/tag rewrite/blind retry/re-upload.
- Only authored docs/tests; did not execute documented release commands or change
  tags, releases, deployments, secrets, providers, environments or infrastructure.
  No current PyPI availability, macOS GPU or Windows verification claimed.
- Current release line, clone v0.2.1 command and release_candidate_test.py retained
  for T-124. T-038 stays blocked/maintainer-only; T-127 remains todo/unfixed;
  T-128 remains future-spec todo/unstarted. No second task or persistent thread.
- 2026-10-16T14:09:02Z: verification pass
- 2026-10-16T14:09:02Z: Workflow-v3 complete after independent reviews and fresh disposition verification; exact ruff/mypy/pytest and mise run check passed. Documentation only; T038 remains blocked.
