# Repository Guidelines

## Project Overview

Latinitas Cards is a Typer CLI for Latin Anki workflows. It supports deck profiles,
principal-part card generation, authored-note import, managed CSV updates, reviewed
form parsing, and experimental deck inspection, splitting, annotation, and corpus-based
cloze generation for CSV + Anki package inputs.

Current command set:
- `inspect`: inspect note types/fields in `.apkg`/`.colpkg`
- `split`: split multi-form notes into one-note-per-form
- `annotate`: add CLTK-based annotations
- `cloze`: generate corpus-based cloze examples from text or parallel corpora
- `preview`: preview principal-part cards (`--profile`) or experimental Vulgate clozes (`--usfx`)
- `generate`: export principal-part CSV (`--profile`) or experimental Vulgate clozes (`--usfx`)
- `validate`: validate corpus/input integrity
- `setup`: inspect a source and confirm a reusable deck profile
- `authored {validate,preview,export}`: import authored JSONL notes
- `managed {plan,approve,emit,observe,reconcile}`: offline managed CSV update handoff
- `form-parsing {preview,export}`: reviewed contextual form-parsing exercises

## Architecture

Code lives in `src/latinitas_cards/`.

- `cli.py` contains shared/domain logic and the Typer app instance.
- `commands/` contains one module per command. Older commands (`inspect`, `split`,
  `annotate`, `cloze`, `preview`, `generate`, `validate`) are thin re-exports of a
  `*_impl` callback in `cli.py`; newer command groups implement Typer sub-apps
  (`authored.py`, `managed.py`, `form_parsing.py`) or re-export a domain module
  (`setup.py`). `principal_parts.py` holds the `--profile` helpers used by
  `preview`/`generate`, not a command of its own.
- Domain modules (`cards.py`, `profile.py`, `sources.py`, ...) sit beside `cli.py`.
- `__main__.py` is the CLI entry module.

### Refactoring Direction

Don't grow `cli.py` further: put new command implementations and domain helpers in
focused modules, and move an existing `*_impl` out of `cli.py` when you
substantially change it.

## Runtime Prerequisites

Some features need runtime resources beyond package installation:

- `annotate` depends on CLTK/Stanza Latin resources. CLTK ships in the optional
  `annotate` extra (`uv sync --extra annotate`, CPU-only PyTorch) or the highly
  experimental conflicting `annotate-gpu` extra (CUDA PyTorch). The default dev
  environment and CI run without either, so tests must not import CLTK.
- Optional LLM/Ollama analysis requires a running endpoint when enabled.

If a command fails due to missing runtime resources, treat it as an environment/setup
issue first, not necessarily a code regression.

## Project Structure

- `src/latinitas_cards/`: application package
- `src/latinitas_cards/cli.py`: shared parsing/input/annotation/cloze logic
- `src/latinitas_cards/commands/`: command callback modules
- `tests/unit/`: unit tests
- `data/`: sample corpora/deck artifacts
- `specs/`: versioned release specs (`specs/vX.Y.Z.md`, index in `specs/README.md`)
- `planning/`: Taskrail tracked work (`STATE.md`, `tasks/`, `artifacts/`)
- `scripts/check-*.sh`: commit policy guards, each with its own `*-test.sh` suite
- Root configs: `pyproject.toml`, `mise.toml`, `lefthook.yml`, `.github/workflows/*.yml`

## Tracked Work (Taskrail)

Planning and task state live in the repo, managed by the `taskrail` CLI.

- `specs/` — versioned specs. `specs/v0.1.0.md` defines the first release and
  `specs/v0.2.0.md` is active for managed updates, extraction, form parsing and
  morphology themes. Activate later specs through Taskrail.
- `planning/STATE.md` — current focus, blockers, next action.
- `planning/tasks/` — one file per task, each linked to a spec heading via `spec_ref`.
- `planning/artifacts/` — verification artifacts, gitignored.

```bash
taskrail status                 # current snapshot (read-only)
taskrail next                   # deterministic next eligible task
taskrail start <task-id>        # mark active (one task at a time)
taskrail verify <task-id>       # write verification artifacts
taskrail complete <task-id>     # mark implemented
taskrail block <task-id>        # record a blocker
taskrail validate               # check structure and state
taskrail coverage               # spec coverage / orphan / drift signals
```

Do not hand-edit `planning/STATE.md`; go through the CLI. New work needs a task
(`taskrail task new --title ... --area <spec-anchor>`) so no change bypasses a spec heading.
Run `taskrail verify`/`complete` only after the mandatory ruff/mypy/pytest chain passes, and
never paste a concrete `planning/artifacts/...` path into a committed note — `validate`
rejects committed references to gitignored artifact paths; cite the verify run timestamp.

Repo-agnostic tracked-work skills are installed under `.claude/skills/` and `.agents/skills/`
(`taskrail init --with-skills`).

## Build, Test, and Development Commands

Use uv for environment and task execution. `mise.toml` pins the rest of the toolchain.

- `mise run setup` (pinned tools, `uv sync --locked --dev`, `lefthook install`)
- `mise run check` (full local gate, mirrors CI)
- `uv sync --locked --dev`
- `uv run pytest -v`
- `uv run pytest tests/unit/cli_test.py`
- `uv run pytest -k "test_name"`
- `uv run ruff check`
- `uv run ruff check --fix`
- `uv run ruff format`
- `uv run mypy`
- `uv run latinitas-cards --help`

## Coding Style & Naming Conventions

- Python target: `>=3.13,<3.15`; keep 3.13 compatibility.
- Ruff rules: `E, F, UP, B, SIM, I`; line length `120`.
- Format with `ruff format` and sort imports via ruff's `I` rule; this project does not use black or isort.
- mypy strict mode; type public functions and non-trivial internals explicitly.
- Use snake_case for functions/variables and lowercase module names.
- Keep modules focused and composable.
- Test files must be named `*_test.py`.

## Testing Guidelines

- Framework: `pytest`. `pytest-cov` is not a dependency, so don't pass `--cov`; no custom
  markers (e.g. `pytest.mark.unit`) are registered, so don't add them.
- Place tests under `tests/unit/`.
- Add regression coverage for:
  - USFX parsing and normalization,
  - APKG/COLPKG import/export behavior,
  - split heuristics,
  - annotation fallback/error paths,
  - cloze difficulty/translation-column handling,
  - CLI command registration and option behavior.

### TDD Requirement for Refactors

For non-trivial refactors (especially module splits), use red/green TDD:

1. Add/adjust a test that demonstrates the target structure/behavior (`RED`).
2. Implement the smallest refactor to make it pass (`GREEN`).
3. Clean up while keeping tests green (`REFACTOR`).

### Mandatory Validation After Code Changes

Run this exact chain after code changes:

1. `uv run ruff check`
2. `uv run mypy`
3. `uv run pytest -v`

If any command fails, fix it and rerun the **full chain from the start**.

## Commit & Pull Request Guidelines

- Coding agents must run `mise run setup` (or `lefthook install`) before creating
  their first commit in a worktree; do not assume the hooks are already installed.
- Use Conventional Commits with imperative subjects. Types: `feat fix refactor
  docs test chore build perf ci`.
- Include a descriptive body after the subject, wrap body lines at 72 characters,
  and suffix tracked-task subjects with the short key, for example `(T-001)`.
- Never add attribution trailers: no co-authorship line, no agent session or
  thread trailer, and no session link. `scripts/check-attribution.sh` is the one
  policy the `commit-msg` and `pre-push` hooks both apply.
- Commit under the maintainer's git identity. `scripts/check-author.sh` refuses
  an agent author in `pre-commit` and again in `pre-push`.
- Keep commits focused and atomic (code + tests together).
- PRs should include:
  - concise behavior summary,
  - linked issue(s) when applicable,
  - validation evidence (`ruff`, `mypy`, `pytest`).
- If CLI output changes, include a short before/after example.

## Changelog Guidelines

- Keep `CHANGELOG.md` user-focused: describe what users can do, what changed for
  them, or which observable problem was fixed. Lead with the command or capability.
- Use short, one- or two-line bullets and fold related changes into one entry.
  Omit internal refactors, tests, CI/toolchain work, dependency noise, task IDs,
  verification evidence, and implementation mechanics; keep those in task notes,
  focused docs, or commit messages.
- Use `Added`, `Changed`, `Fixed`, and `Limitations` sections only when useful.
  Include practical safety limits, but do not duplicate the spec's exclusions.
- Put pending changes under `[Unreleased]`. At release time, move them under
  `## [<version>] - YYYY-MM-DD`, leave an empty `[Unreleased]` section, and update
  release/comparison links. GitHub release notes should follow the same concise style.
