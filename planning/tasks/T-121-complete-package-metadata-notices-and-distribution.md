---
id: T-121-complete-package-metadata-notices-and-distribution
title: Complete package metadata, notices and distribution and install check scripts
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies:
    - T-109-normalize-usfx-sources-into-the-corpus-model-with
    - T-103-define-the-cli-result-envelope-exit-codes
updated_at: "2026-10-14T05:42:59Z"
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
