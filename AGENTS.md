# Repository Guidelines

## Project Overview

Latinitas Cards is a Typer CLI for Latin Anki workflows. It supports deck inspection,
card splitting, grammatical annotation, corpus-based cloze generation, and validation
for CSV + Anki package inputs.

Current command set:
- `inspect`: inspect note types/fields in `.apkg`/`.colpkg`
- `split`: split multi-form notes into one-note-per-form
- `annotate`: add CLTK-based annotations
- `cloze`: generate corpus-based cloze examples from text or parallel corpora
- `preview`: preview Vulgate clozes without writing output
- `generate`: write Vulgate clozes back to CSV/APKG-compatible output
- `validate`: validate corpus/input integrity

## Architecture

Code lives in `src/latinitas_cards/`.

- `cli.py` contains shared/domain logic and the Typer app instance.
- `commands/` contains one file per command callback.
- `__main__.py` is the CLI entry module.

### Refactoring Direction

The legacy implementation concentrated most logic in `cli.py`. Do **not** add new
large command implementations to one monolithic file. Prefer extracting command entry
points and domain helpers into focused modules.

## Runtime Prerequisites

Some features need runtime resources beyond package installation:

- `annotate` depends on CLTK/Stanza Latin resources.
- Optional LLM/Ollama analysis requires a running endpoint when enabled.

If a command fails due to missing runtime resources, treat it as an environment/setup
issue first, not necessarily a code regression.

## Project Structure

- `src/latinitas_cards/`: application package
- `src/latinitas_cards/cli.py`: shared parsing/input/annotation/cloze logic
- `src/latinitas_cards/commands/`: command callback modules
- `tests/unit/`: unit tests
- `data/`: sample corpora/deck artifacts
- Root configs: `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/*.yml`

## Build, Test, and Development Commands

Use Poetry for environment and task execution.

- `poetry install`
- `poetry run pytest -v`
- `poetry run pytest tests/unit/cli_test.py`
- `poetry run pytest -k "test_name"`
- `poetry run ruff check`
- `poetry run ruff check --fix`
- `poetry run ruff format`
- `poetry run mypy`
- `poetry run latinitas-cards --help`

## Coding Style & Naming Conventions

- Python target: `>=3.10,<3.13`; keep 3.10 compatibility.
- Ruff rules: `E, F, UP, B, SIM, I`; line length `120`.
- mypy strict mode; type public functions and non-trivial internals explicitly.
- Use snake_case for functions/variables and lowercase module names.
- Keep modules focused and composable.
- Test files must be named `*_test.py`.

## Testing Guidelines

- Framework: `pytest`.
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

1. `poetry run ruff check`
2. `poetry run mypy`
3. `poetry run pytest -v`

If any command fails, fix it and rerun the **full chain from the start**.

## Commit & Pull Request Guidelines

- Use Conventional Commits: `feat:`, `fix:`, `chore:`, `ci:`.
- Keep commits focused and atomic (code + tests together).
- PRs should include:
  - concise behavior summary,
  - linked issue(s) when applicable,
  - validation evidence (`ruff`, `mypy`, `pytest`).
- If CLI output changes, include a short before/after example.
