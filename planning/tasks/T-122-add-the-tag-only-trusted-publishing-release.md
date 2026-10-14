---
id: T-122-add-the-tag-only-trusted-publishing-release
title: Add the tag-only Trusted Publishing release workflow with no-publish guards
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies:
    - T-121-complete-package-metadata-notices-and-distribution
updated_at: "2026-10-14T05:42:59Z"
---

# T-122-add-the-tag-only-trusted-publishing-release Add the tag-only Trusted Publishing release workflow with no-publish guards

## Description

Adds `.github/workflows/release.yml` and the publish guard required by
`specs/v0.3.0.md#pypi-publishing` (the `release.yml` and guard bullets and the
Acceptance paragraph's workflow test). Today only `build.yml`, `caches.yml` (with
`actions: write`) and `mutation.yml` exist and none publishes. `release.yml` reuses
`scripts/check-dist.sh` and `scripts/check-install.sh` from T-121. The workflow
cannot run unattended, so acceptance here is structural; Trusted Publishing itself is
proven only at the maintainer gate (T-038). `docs/releasing.md` belongs to T-123;
no tag, environment, Trusted Publisher or publication is created here.

## Acceptance

- A pytest test reading `release.yml` as text line by line (no YAML dependency) asserts
  that `on:` holds only `push:` with `tags:` `v[0-9]+.[0-9]+.[0-9]+`, with no `branches`,
  `pull_request`, `workflow_dispatch`, `schedule` or `release` trigger, and that the
  top-level `permissions:` is `contents: read`.
- The same test asserts the jobs `build`, `verify`, `publish` and `verify-pypi` appear in
  that order, each `needs:` the previous one; `build` checks out full history, fails
  unless `GITHUB_REF_NAME == "v" + project.version` (read with `tomllib`), fails unless
  the tag commit is an ancestor of `origin/main`, runs `scripts/check-dist.sh` and uploads
  `dist/` as one artifact; `verify` downloads that artifact and runs
  `scripts/check-install.sh` on `"3.13"` and `"3.14"`.
- `publish` is the only job with `environment: pypi` and `id-token: write`; it downloads
  the same artifact and uses `pypa/gh-action-pypi-publish@` followed by a 40-hex commit
  SHA and a version comment. `release.yml` contains no `password:`, `secrets.`,
  `PYPI_API_TOKEN`, `twine`, `uv publish`, `repository-url` or `test.pypi.org`.
- `verify-pypi` installs `latinitas-cards==<version>` from PyPI into a fresh venv with a
  retry loop bounded at 10 minutes, runs `latinitas-cards --help`, and asserts `cltk`,
  `stanza` and `torch` are absent; the test asserts the bound and the three checks.
- `scripts/check-publish-guard.sh [DIR]` (default `.github/workflows`) scans every
  `*.yml`/`*.yaml` file and fails, naming the file, if any file other than `release.yml`
  contains `id-token: write`, `pypa/gh-action-pypi-publish`, `twine upload`, `uv
  publish` or `environment: pypi`, or if `release.yml` has a non-tag trigger or
  `id-token: write` outside `publish`. It exits 0 on the repository.
- `scripts/check-publish-guard-test.sh` copies the real workflows into temp dirs: the
  unchanged copy passes; adding `id-token: write` to `build.yml`, a
  `pypa/gh-action-pypi-publish` step to `mutation.yml`, a `twine upload` step to a new
  `extra.yml`, `workflow_dispatch:` to `release.yml`, and `id-token: write` to the
  `build` job of `release.yml` each fail. It runs in `mise run check` and the `build.yml`
  `guards` job, and a pytest test asserts both invocations in `mise.toml` and `build.yml`.
- `build.yml`, `caches.yml` and `mutation.yml` gain no permission, trigger or publish
  step; `release.yml` triggers nothing on branches, so routine CI cannot publish.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the pinned `gh-action-pypi-publish` SHA and the release tag it resolves to, and
  the guard-test cases that failed as intended.

## Implementation Notes

- Keep the guard grep/awk based; it takes the workflow directory as an argument so the
  test can point it at fixtures. Model it on `check-mutation-config-test.sh` and the
  `fail()` helper used by the other `scripts/check-*-test.sh` suites.
- Resolve the publish action SHA with `git ls-remote --tags
  https://github.com/pypa/gh-action-pypi-publish` for its current `v1` release and record
  the tag in the trailing comment. Resolve the SHA first, before any other work, so an
  offline block costs nothing: if no network is available, `taskrail block` this task with
  the reason `publish action SHA unresolved: no network` rather than inventing one.
- Tag-on-main: `actions/checkout` with `fetch-depth: 0`, `git fetch origin main`, then
  `git merge-base --is-ancestor "$GITHUB_SHA" origin/main`. Upload and download one
  artifact (for example `dist`) so `publish` ships exactly what `verify` installed.
- Have `build` print the SHA-256 of each dist file to `$GITHUB_STEP_SUMMARY` so the gate
  can compare them with the PyPI JSON API (supports the gate's hash check; not required
  by the spec text).
- Tests in `tests/unit/release_workflow_test.py`; a small line-oriented helper extracts
  top-level job ids and per-job keys. Confirm `.github/workflows/` is in
  `[tool.mutmut].also_copy` (added by T-121). Add `mise.toml` to `[tool.mutmut].also_copy`
  and to `mutation_config_test.py` (the guard-invocation test reads it inside the mutmut
  sandbox).
- Size guard: one workflow, one guard script with its test and one pytest module. If
  other workflows need restructuring, file a follow-up rather than widen.
