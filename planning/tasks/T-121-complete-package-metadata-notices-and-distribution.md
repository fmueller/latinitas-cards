---
id: T-121-complete-package-metadata-notices-and-distribution
title: Complete package metadata, notices and distribution and install check scripts
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies:
    - T-109-normalize-usfx-sources-into-the-corpus-model-with
    - T-103-define-the-cli-result-envelope-exit-codes
updated_at: "2026-10-16T13:23:39Z"
---

# T-121-complete-package-metadata-notices-and-distribution Complete package metadata, notices and distribution and install check scripts

## Description

Delivers the offline packaging deliverables of `specs/v0.3.0.md#pypi-publishing`: the
Metadata bullet, `scripts/check-dist.sh` with `check-dist-test.sh`,
`scripts/check-install.sh` and the `build.yml` `package` job.
Today `pyproject.toml` has a cloze-first `description`, no `[project.urls]`, Python-only
classifiers and `license-files = ["LICENSE"]`; README links `docs/...`, `LICENSE` and
`AGENTS.md` relatively; no script builds or installs distributions and `build.yml` has no
`package` job. The release workflow and publish guard belong to T-122, install docs and
the runbook to T-123, the 0.3.0 bump to T-124; nothing here tags or publishes.

## Acceptance

- A pytest test reads `pyproject.toml` with `tomllib` and asserts: `description` leads
  with the deck-first workflow (Anki deck or CSV export to principal-part cards) before
  any corpus, cloze or annotate wording; `[project.urls]` has `Homepage`, `Repository`,
  `Changelog` and `Issues` under `https://github.com/fmueller/latinitas-cards`;
  classifiers include `Programming Language :: Python :: 3.13`, `... :: 3.14`,
  `Operating System :: POSIX :: Linux` and `Operating System :: MacOS` and no `License ::`
  classifier; `license == "GPL-3.0-or-later"`; `license-files == ["LICENSE",
  "THIRD_PARTY_NOTICES.md"]`; `requires-python`, `dependencies` and both `annotate` and
  `annotate-gpu` extras equal their current values; `uv.lock` package inventory unchanged.
- A README test finds no relative link: every inline `](target)` and reference definition
  is `https://`, `mailto:` or an in-page `#fragment`; the same helper flags a negative
  control string containing `](docs/x.md)`. Existing README assertions
  (`reference_templates_test.py`, `release_candidate_test.py`) are updated to the absolute
  form, not deleted.
- `scripts/check-dist.sh` builds sdist and wheel with `uv build` in a temp dir, extracts
  the sdist and rebuilds a wheel from it, then inspects all artifacts: `METADATA` `Name`,
  `Version` equal to `project.version`, `License-Expression: GPL-3.0-or-later`,
  `License-File` for `LICENSE` and `THIRD_PARTY_NOTICES.md`, `Requires-Python`,
  `Provides-Extra` for both extras, entry point `latinitas-cards =
  latinitas_cards.cli:app`, every non-`__pycache__` file under `src/latinitas_cards/`
  present in the wheel, equal member lists for both wheels, and no `tests/`, `data/`,
  `planning/` or `.amp/` member in any artifact.
- `scripts/check-dist-test.sh` runs only the inspection phase (no build, no network) on
  synthetic archives built in a temp dir: a conforming wheel and sdist pass; a wrong
  version, a missing `License-Expression`, a missing notices `License-File`, a missing
  entry point, and a `tests/`, `data/lat-clementine.usfx.xml`, `planning/` or `.amp/`
  member each fail with a message naming the defect. It runs in `mise run check` and in
  the `build.yml` `guards` job.
- `scripts/check-install.sh PYTHON` installs the wheel and, separately, the sdist from
  `dist/` into fresh venvs under a temp dir with the working directory outside the
  checkout; runs `latinitas-cards --help`, `latinitas-cards --version` (exactly
  `latinitas-cards <project.version>`), `corpus normalize` on the small committed USFX
  fixture and `corpus validate --json` (exit 0, `"ok": true`); and fails if `cltk`,
  `stanza` or `torch` is installed in either venv.
- `build.yml` gains a `package` job (`needs: lint-and-typecheck`, matrix `"3.13"`,
  `"3.14"`, no `id-token`, no publish step) running `check-dist.sh` then
  `check-install.sh`; a pytest test asserts this on the YAML text line by line with no
  YAML dependency, and that the `guards` job runs `check-dist-test.sh`.
- Run `UV_OFFLINE=1 bash scripts/check-dist.sh` and `UV_OFFLINE=1 bash
  scripts/check-install.sh "$(uv python find 3.14)"` against the cache warmed by `uv sync
  --locked --dev` (neither script runs inside pytest). If either fails only because a
  package is not cached, rerun without `UV_OFFLINE`. If no network is available, record
  "check-install not run: offline" with the error line and continue. The `package` CI job
  (verified on the candidate by T-038) is the binding check, and the offline
  `check-dist-test.sh` in `mise run check` remains mandatory.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the `check-dist.sh` and `check-install.sh` exit codes, the interpreter version,
  and the wheel/sdist file names and SHA-256 values from the local run.

## Implementation Notes

- Keep setuptools>=77 (PEP 639 `license-files`, Metadata 2.4). setuptools rejects
  `License ::` classifiers next to a license expression; do not add one. If the sdist
  picks up forbidden paths, add `MANIFEST.in` prune lines; do not change the package
  layout. `data/lat-clementine.usfx.xml` stays out of both artifacts.
- Split `check-dist.sh` into build and inspect phases (for example an `--inspect DIR`
  mode) so `check-dist-test.sh` exercises inspection offline. Inspect with `python3 -I`
  and stdlib `zipfile`/`tarfile`, or `unzip -l`/`tar -tzf`; add no dependency.
- `check-install.sh`: `uv venv --python "$PYTHON"` in `mktemp -d`, `uv pip install` the
  artifact with no extras, `cd` into the temp dir before running the CLI; pass the
  fixture by absolute path. Use the small USFX fixture committed by T-109; `--version`
  comes from T-103.
- Rewrite README links as `https://github.com/fmueller/latinitas-cards/blob/main/<path>`.
  Update the two existing README assertions first (RED), then the README (GREEN).
- New tests in `tests/unit/packaging_metadata_test.py`; workflow assertions use
  `str.splitlines()`. Add `.github/workflows/` to `[tool.mutmut].also_copy`, and
  `.github/workflows/build.yml` to the `mutation_config_test.py` parameter list (it asserts
  files, not directories), since the tests read `build.yml` inside the mutmut sandbox.
- Size guard: metadata, two scripts with one test suite and one CI job. If the work
  needs more (for example new runtime resources), file a follow-up rather than widen.

## Execution Audit — 2026-10-16

### 1. Understand and acquire

- Fetched `origin/main`: checkout and remote both at
  `e83c2f13c39844f6f5ceca83f4985dc3805e442c`. After unshallowing, confirmed
  tested T120 `13c65c6120317a7fddebadb5d7eafbe3255d8d84` is an ancestor.
- Pinned `specs/v0.3.0.md` SHA-256:
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
- `mise exec -- taskrail validate`: state valid. `next --json` selected exactly
  T121, todo; status showed zero active tasks. No source-checkout guard applies
  (this adopter has no Taskrail source helper). `start` recorded in_progress at
  2026-10-16T13:03:42Z. All Taskrail commands use `mise exec -- taskrail`.
- Ownership: setuptools/pyproject, README, build workflow, mise gate and script
  guards; no runtime product modules changed. Deck-first metadata serves package
  readers; installed smoke commands prove packaged runtime files work outside
  editable/source imports. Dependencies, both extras, uv.lock and release version
  remain unchanged. Only this task is owned; no second task started.
- Installed hooks with `mise exec -- lefthook install`; confirmed maintainer
  identity Felix Müller. `uv sync --locked --dev`: resolved 93, checked 41.

### 2. Strict TDD

- First changed README assertions and added metadata/link/line-by-line workflow
  acceptance tests, then ran `uv run pytest -q` on packaging_metadata_test.py,
  reference_templates_test.py and release_candidate_test.py: 5 failed, 10 passed.
  Failures named old cloze-first metadata, relative README links, absent package
  job and the two absolute-link README assertions. Implemented metadata, links
  and CI: same files plus mutation_config_test.py passed, 35 tests.
- Added offline synthetic guard before inspector: `bash scripts/check-dist-test.sh`
  exited 1 because check-dist.sh was absent. Implemented inspector: 41 cases passed.
  `check-install.sh` initially exited 127 because absent; implemented it and later
  exercised real wheel and sdist installs rather than running builds inside pytest.
- Actual setuptools build exposed reordered `Requires-Python` specifiers. Changed
  the conforming synthetic metadata to `<3.15,>=3.13`: guard exited 1 with
  `Requires-Python must equal >=3.13,<3.15`. Inspector now compares comma-separated
  constraint sets, retaining the discriminating `>=3.14` negative: 41 passed.
- Added mise.toml mutation-sandbox parameter first: 1 failed, 20 deselected,
  `mutmut does not copy mise.toml`. Added also_copy entry: 21 passed. Workflow
  directory and concrete build.yml dependency are copied/asserted too.
- Final implementer cross-check found empty directory members were ignored.
  Added synthetic directory cases: exited 1 at `wheel: forbidden directory tests/`
  because defective wheel passed. Inspect all archive names, including directories.
  Added unequal rebuilt-wheel directory list: exited 1 because defect passed;
  compare complete member names instead of regular files only. Final 54 cases pass
  (one conforming control and 53 isolated defects); tar directory diagnostics use
  normalized component names because tarfile strips trailing slashes.

### 3. Initial checks and real artifacts

- Initial exact `uv run ruff check` → `uv run mypy` → `uv run pytest -v`:
  all passed; mypy 167 source files; pytest 2449 passed, one existing
  PydanticSerializationUnexpectedValue warning, 102.36 seconds. Subsequent addition
  of the mise sandbox parameter increases the full count by one.
- Shell syntax, `git diff --check` and `uv run ruff format --check` passed.
- First `UV_OFFLINE=1 bash scripts/check-dist.sh`: exit 2, missing cached
  setuptools (`setuptools>=77.0.1 needs to be downloaded from a registry`).
  Prescribed `bash scripts/check-dist.sh` network retry fetched build dependencies,
  then exited 1 on the specifier-order inspector bug above. After its TDD fix,
  offline build/rebuild inspection exited 0. No MANIFEST.in was necessary: actual
  artifacts already exclude forbidden paths.
- First `UV_OFFLINE=1 bash scripts/check-install.sh "$(uv python find 3.14)"`:
  exit 1, `pydantic was not found in the cache`. Prescribed network retry exited 0
  for both wheel and sdist. A warmed-offline repetition also exited 0. This is the
  explicitly scoped acceptance-permitted cache/network fallback, not an offline
  success claim about the initial cache. Nothing was skipped or waived otherwise.
- Final offline build/rebuild and offline install both exited 0 at
  2026-10-16T13:16:36Z, after the complete-member TDD fixes. Python 3.14.2
  (Clang 21.1.4); stdlib inspection uses `python3 -I`. Each installed artifact ran
  `--help`, exact `latinitas-cards 0.2.1`, USFX `corpus normalize`, and
  `corpus validate --json` with exit 0 and `ok=true`. Fresh separate temp venvs and
  working directory outside checkout; cltk, stanza and torch distributions absent.
  No optional resources were installed or downloaded.

| Final artifact | SHA-256 |
|---|---|
| latinitas_cards-0.2.1-py3-none-any.whl | 823ede512c98b1f5185fe9c6960855e7146178b4639c178ade7253d27b50d972 |
| latinitas_cards-0.2.1.tar.gz | 8d34910ea87942d3239fa45b131f3e56ffccbe265060cf4995a7876abb988f24 |
| rebuilt latinitas_cards-0.2.1-py3-none-any.whl | 3f9e0ed704d14a83c366cb0a6cba51f642b7432af8a280f5bf122c2e2ac6278b |

Both wheels have identical 91-file member lists; sdist has 96 regular files.
Rebuilt bytes need not match (archive timestamps); acceptance compares member lists.
Synthetic metadata/content is independently authored, not produced by the inspector.

### 4–6. Simplification, review and dispositions

- Dedicated Task loaded personal code-simplifier, inspected actual diff and bounded
  task contract: **no changes made**, no concerns. Its 41-case synthetic suite,
  24 focused tests, ruff/format and shell syntax checks passed. No suggestions rejected.
- Selected separate parallel General, Python and Security Tasks. General is always
  required; Python covers stdlib archive logic/tests; Security covers shell/temp-file
  effects, archive leakage, URLs and CI permissions. Two specialist lanes are within
  budget. Omitted FastAPI (no API/model changes), Database (no persistence changes),
  ML/RAG (optional resources explicitly excluded), Network Configuration (no network
  infrastructure changes), and other languages/frameworks (not changed).
- All loaded personal code-reviewer and reviewer-index. General loaded ECC
  code-reviewer (no companions); Python loaded python-reviewer and python-patterns;
  Security loaded security-reviewer, security-review and common security,
  code-review and testing rules. Each inspected actual diff plus untracked files.
- General verbatim: `No concrete task-relevant findings.` Its 41 synthetic cases,
  36 focused tests, diff and syntax checks passed.
- Python verbatim: `No concrete task-relevant findings.` Its 41 synthetic cases,
  36 focused tests, README scan, inventory comparison and diff checks passed.
- Security verbatim: `No concrete task-relevant findings.` Its 41 synthetic cases,
  actual warmed-offline build/rebuild and diff checks passed. It did not claim
  fresh installs or CI itself.
- Fresh candidate-validation Task loaded code-reviewer and reviewer index:
  zero supplied, zero validated, zero rejected/deduplicated; found no contradiction.
  Verbatim: `No concrete task-relevant findings.` Its synthetic/diff checks passed.
- No reviewer finding IDs, fixes or deferrals exist. The implementer's later empty
  directory acceptance correction is documented with RED/GREEN above and is included
  in fresh final disposition verification. One review cycle; no persistent threads.

### 7. Final recheck and fresh disposition verification

- After all implementation corrections, restarted the entire mandatory chain at
  2026-10-16T13:16:21Z: `uv run ruff check` (all checks passed) → `uv run mypy`
  (167 source files, no issues) → `uv run pytest -v` (2450 passed, one existing
  serializer warning, 103.31 seconds). Exit 0 at 2026-10-16T13:18:05Z.
- `mise run check`: exit 0 at 2026-10-16T13:19:53Z. Ruff, format, mypy, full pytest
  and all six shell guard suites passed, including the 54-case offline dist guard.
- Fresh dedicated Task loaded code-reviewer in Disposition Verification Mode,
  reviewer-index, General ECC reviewer and applicable general references. Read
  actual final changes, acceptance, audit and late directory correction. Verbatim:
  `Disposition: RESOLVED — no prior finding IDs or deferrals; none are unresolved.`
  `No concrete task-relevant findings.` It confirmed complete-name leakage and
  wheel-member comparisons plus RED/GREEN coverage; no new issues, no edits.
  It did not rerun checks/builds or modify the recorded artifacts.

### Scope and delivery limits

- All quality/review gates preceded Taskrail verification/completion. No changes
  to spec, priorities, dependencies, acceptance gates, runtime modules, extras or
  lock inventory. No task followed automatically after this one.
- Local distribution checks are not candidate CI acceptance. Python 3.13/3.14
  `package` CI is the binding T038 maintainer gate; no manual workflow run,
  id-token, tag, publication, release or deployment was performed here.
- T127 exporter remains todo/unfixed; T128 future v0.8 memory remains excluded
  and unstarted; T038 stays blocked/excluded. T122 release workflow, T123 docs
  and T124 version bump retain their own scope. No additional follow-up created.
- 2026-10-16T13:23:39Z: verification pass
- 2026-10-16T13:23:39Z: Implementation 81921a6307ae75214328ca47076c0949a9218b84 passed committed exact ruff/mypy/pytest chain at 2026-10-16T13:23:27Z; full mise check and independent review/disposition verification passed. Built and freshly installed wheel/sdist offline after prescribed cache fallback. CI acceptance is maintainer T038, not claimed here.

### 8. Finalize

- Reviewed implementation commit: `81921a6307ae75214328ca47076c0949a9218b84`.
  Repeated the exact ruff → mypy → pytest chain on this committed candidate:
  exit 0 at 2026-10-16T13:23:27Z; 2450 passed, one existing warning, 105.29 seconds.
  No product changes followed these checks; the final commit only records lifecycle
  and audit state. Commit hooks passed under Felix Müller's identity, no trailers.
- Rechecked pin, valid state and sole T121 active ownership before verify/complete.
  `mise exec -- taskrail verify ... --result pass` recorded pass at
  2026-10-16T13:23:39Z; then `complete` recorded completed at that timestamp.
  Final `validate`: state valid. No manual STATE.md edits.
- Final snapshot: 121 done, zero active, one blocked, six todo. Read-only next
  observed `T-122-add-the-tag-only-trusted-publishing-release`, todo and unstarted.
  T127/T128 still todo; T038 still blocked. Existing single orphan signal remains
  unchanged, not repaired or used to bypass selection.
- Pre-delivery fetch still found `origin/main` at the accepted cumulative base
  `e83c2f13c39844f6f5ceca83f4985dc3805e442c`, an ancestor of this candidate.
  Authorized delivery is a non-force push to origin/main; remote containment and
  the delivered lifecycle/audit revision are reported in the completion callback.
