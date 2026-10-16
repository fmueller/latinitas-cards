---
id: T-122-add-the-tag-only-trusted-publishing-release
title: Add the tag-only Trusted Publishing release workflow with no-publish guards
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies:
    - T-121-complete-package-metadata-notices-and-distribution
updated_at: "2026-10-16T13:49:55Z"
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

## Workflow-v3 implementation audit (2026-10-16 UTC)

### Understand and ownership

- Accepted cumulative base: fbe2de1eff9e1e023528321e6bb0a5ed7f46e664,
  identical to fetched origin/main and local HEAD at preflight. After unshallowing,
  `git merge-base --is-ancestor 81921a6307ae75214328ca47076c0949a9218b84 HEAD`
  exited 0 (tested T-121 is contained). No source-overseer local main transferred.
- Spec SHA256: fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704.
  Read this task and PyPI Publishing acceptance/contracts and relevant exclusions;
  corpus, Anki and runtime application behavior remain untouched.
- Preflight `mise exec -- taskrail validate`: `state valid`; `next --json` selected
  this exact todo task; STATE current_task empty, in_progress count 0. Then
  `taskrail start` transitioned only T-122 to in_progress. No competing ownership.
- First operation was git ls-remote for the action. There is no bare v1 tag;
  version-sorted v1 tags resolved current v1.14.2, independently rechecked:
  annotated tag a892a5a61159132606e93a2fa6f4358831b04d26, peeled commit
  dc37677b2e1c63e2034f94d8a5b11f265b73ba33. Workflow pins the peeled commit.
- One workflow, one grep/awk guard plus synthetic suite and one pytest module;
  build.yml adds only the guard invocation, mise check adds the same invocation.
  mise.toml and workflows were already in mutmut also_copy and its regression
  coverage, so those existing entries needed no change.

### Strict TDD and initial checks

- `uv run pytest -q tests/unit/release_workflow_test.py`: RED, 2 failed
  (missing release.yml and missing local guard invocation); implementation GREEN,
  2 passed. Guard suite first exited 127 because guard did not exist; after
  implementation the unchanged copy passed and prescribed negatives failed.
- Self-review added inline top-level OIDC case: RED `FAIL: accepted unsafe
  release.yml: inline top-level OIDC permission`; minimal awk scope fix GREEN.
- Initial exact ruff/mypy/pytest chain: ruff passed, mypy passed (168 files),
  pytest 2451 passed / 1 failed in 164.88s, phrase appearance browser-state
  collision. Concurrent mise check used the same fixed phrase-appearance browser
  session; its pytest passed 2452 with one existing warning in 159.21s and all
  guards passed. Subsequent full suites run sequentially, not concurrently.
- Sequential full-chain recheck: ruff passed, mypy 168 files clean, pytest
  2452 passed, 1 warning in 141.76s. Existing warning is pydantic serialization
  of a deliberately invalid floating-point fixture, not a release change.
- `bash scripts/check-dist.sh`: distribution inspection passed, including
  rebuilt-wheel membership. `bash scripts/check-install.sh` with uv's Python
  3.14: both wheel and sdist passed --help, version, corpus normalize and
  corpus validate (ok=true); cltk/stanza/torch absent. No publishing invoked.

### Simplification and independent reviews

- Dedicated simplifier Task loaded personal code-simplifier. Reused expected job
  tuple for job order and needs assertions, no behavioral changes. Focused pytest
  2 passed, ruff/format passed, 12 guard negatives rejected, diff check passed.
- Separate parallel read-only lane Tasks: General (ECC code-reviewer, no companion
  prescribed), Security (ECC security-reviewer, security-review REFERENCE,
  cloud-infrastructure-security, common security rules), Python (ECC
  python-reviewer, python-patterns REFERENCE). Each loaded personal code-reviewer
  and reviewer-index routing. Three lanes are within the soft budget. No database,
  UI/framework, ML/RAG/healthcare or network-runtime changes warrant other lanes.
- General: `No concrete task-relevant findings.` Security and Python candidates
  were sent to a fresh code-reviewer candidate-validation Task, which loaded
  security/Python references and reproduced the quoted-key bypass.

### Verbatim validated findings and dispositions

- SEC-1 (security, medium): "The publish guard can miss OIDC write permission in
  another workflow when the YAML key is quoted, so its all-workflows boundary
  check can be bypassed." Evidence: scripts/check-publish-guard.sh:9, copied
  build.yml with quoted id-token key exited 0. FIXED: match single/double quoted
  keys/values in non-release grep and release awk. Regression RED: `FAIL: accepted
  unsafe build.yml: permissions: {"id-token": "write"}`; GREEN: quoted build.yml,
  single-quoted extra.yaml and quoted release build permission rejected.
  Related scope regressions also RED then GREEN: quoted job after publish cannot
  inherit publish scope; quoted non-tag trigger is rejected.
- F2 (tests, medium): "The PyPI test can pass with the retry loop removed or
  disconnected from the installation command, provided the checked strings remain
  elsewhere in the job text." Evidence: tests/unit/release_workflow_test.py retry
  string-presence assertions. FIXED: assert the contiguous deadline, until-install,
  deadline exit, sleep and done lines. A temporary workflow with retry replaced by
  comments and one-shot install passed the old test, failed the strengthened test
  with AssertionError, then the real connected retry passed. No retained scratch
  files or source regression left behind.
- Rejected SEC-2: "The release tag filter does not match ordinary version tags
  such as v1.2.3, so those tags will not start the release workflow."
- Rejected/deduplicated F1: "The tag filter uses regex-style + quantifiers, but
  GitHub Actions interprets tag filters as glob patterns, where + does not mean
  “one or more.”" Both claim + is literal. Official GitHub filter-pattern cheat
  sheet explicitly lists v[12].[0-9]+.[0-9]+ matching v1.10.1/v2.0.0, disproving
  that claim. Keep the exact task/spec pattern, not generic fnmatch semantics:
  https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#filter-pattern-cheat-sheet
- No validated finding deferred. Actual publication and Trusted Publisher/environment
  configuration remain exclusively the maintainer T-038 gate. T-127 stays todo and
  T-128 stays unstarted/off-spec; neither was changed or used to widen this cycle.

### Final recheck and disposition verification

- Start timestamp: 2026-10-16T13:29:50Z. At 13:49:17Z, pre-finalization
  validate still reported state valid, selector returned only this active task,
  only T-122 had in_progress status, and the spec pin still matched.
- Final code exact chain ended 2026-10-16T13:49:07Z: `uv run ruff check`
  All checks passed; `uv run mypy` Success: no issues found in 168 source files;
  `uv run pytest -v` 2452 passed, 1 warning in 138.89s. Exit 0.
- `mise run check` passed sequentially before that final exact chain: pytest
  2452 passed, 1 warning in 139.53s; all policy/mutation suites, 54 distribution
  inspection cases and the publish suite passed. Publish suite: unchanged copy
  accepted and 17 unsafe cases rejected, including the five prescribed negatives,
  yaml extension, uv publish, pypi environment, all other triggers, inline/quoted
  OIDC, quoted job boundary and quoted trigger. `bash -n` and diff check passed.
- Fresh read-only code-reviewer disposition-verification Task read reviewer routing
  and General/Security/Python references and the final diff. Returned:
  "SEC-1 — RESOLVED", "F2 — RESOLVED", "SEC-2 and F1 — REJECTED",
  "New task-relevant issues: None found." It independently reran syntax checks,
  the 17 negative fixtures, focused pytest (2 passed), and diff check. One
  review/fix cycle, no unresolved findings or deferrals.
- Hooks installed with `mise exec -- lefthook install`; maintainer identity
  Felix Müller / felix@fmueller.io. No attribution trailers. Only an implementation
  commit and non-force origin/main push are authorized; no tag, manual workflow
  run, release, deployment, publication or provider/environment change performed.
- 2026-10-16T13:49:49Z: verification pass
- 2026-10-16T13:49:55Z: Implementation only complete after 2026-10-16T13:49:49Z passing verification and fresh independent disposition verification. All 2452 tests, mandatory exact chain, mise check and 17 no-publish negatives passed. No tags or publication performed; T038 remains maintainer-only.
- Post-completion validate: state valid. Read-only selector observed
  T-123-document-pypi-installation-extras-and-the next (todo); not started.
  This thread owns no second task and created no persistent internal threads.
