# Repository Guidelines

## Project Overview

LatinitasCards generates Anki flashcards for learning Latin. The main feature parses a Latin Vulgate Bible in USFX XML format, matches Latin words from an Anki deck export (CSV or .apkg/.colpkg), and produces cloze-deletion cards with verse references.

## Architecture

Single-module CLI tool using a `src/` layout. All logic lives in `src/latinitas_cards/cli.py`:

- **CLI entry point**: Typer app exposed as `latinitas-cards` via `tool.poetry.scripts`. Entry via `__main__.py`.
- **USFX parser** (`parse_usfx_to_df`): Walks Vulgate XML using a manual stack-based traversal, extracting book/chapter/verse/text into a pandas DataFrame.
- **Bucket index** (`build_bucket_index`): First-letter bucketing of normalized verse text for fast word lookup.
- **Cloze generator** (`generate_clozes_for_word`): Finds verses containing a word and wraps the first occurrence in Anki `{{c1::...}}` syntax.
- **Input loader** (`_load_input_to_dataframe`): Reads either CSV (with header auto-detection) or .colpkg (extracts SQLite from zip, reads notes table).
- **Latin normalization** (`normalize_latin`): Lowercases and normalizes æ→ae, œ→oe, j→i.

## Project Structure

- `src/latinitas_cards/`: application package.
- `src/latinitas_cards/cli.py`: core logic (USFX parsing, input loading, cloze generation).
- `src/latinitas_cards/__main__.py`: CLI entry module.
- `tests/unit/`: unit tests (currently focused on CLI behavior).
- `data/`: source artifacts such as `.usfx.xml`, `.apkg`, and `.colpkg` files.
- Root config files: `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/*.yml`.

## Build, Test, and Development Commands

Use Poetry for environment and task execution.

- `poetry install`: install dependencies (including dev tools).
- `poetry run pytest -v`: run the full unit test suite.
- `poetry run pytest tests/unit/cli_test.py`: run a single test module.
- `poetry run pytest -k "test_name"`: run a specific test by name.
- `poetry run ruff check`: run lint checks.
- `poetry run ruff check --fix`: auto-fix lint issues where possible.
- `poetry run ruff format`: format code.
- `poetry run mypy`: run strict type checks.
- `poetry run latinitas-cards --help`: inspect CLI usage.

## Coding Style & Naming Conventions

- Target Python `>=3.10,<3.13`; keep code compatible with 3.10. CI tests on 3.10 and 3.12.
- Ruff for formatting and linting (rules: E, F, UP, B, SIM, I). Line length 120.
- mypy strict mode (`target: 3.10`); add explicit types on public functions and non-trivial internals.
- Follow snake_case for functions/variables and lowercase module names.
- Keep modules focused; prefer small helpers over large monolithic functions.
- Test files named `*_test.py` (enforced by pre-commit `name-tests-test` hook).
- Pre-commit hooks run ruff, ruff-format, and standard checks (trailing whitespace, TOML/YAML validation, LF line endings).

## Testing Guidelines

- Framework: `pytest`.
- Place tests under `tests/unit/` mirroring package behavior.
- Add regression tests for parser edge cases and CLI input/output changes.
- Run `poetry run pytest -v` before opening a PR.

### Mandatory Validation After Code Changes

After completing any task that includes code changes, run this exact validation chain:

1. `poetry run ruff check`
2. `poetry run mypy`
3. `poetry run pytest -v`

If any command fails, fix the issue and then rerun the full chain from the beginning
(`ruff -> mypy -> pytest`) until all checks pass. This is required to reduce
regressions introduced while fixing earlier errors.

## Commit & Pull Request Guidelines

- Follow Conventional Commit style seen in history: `feat:`, `fix:`, `chore:`, `ci:`.
- Keep commits focused and atomic (code + tests for the same change).
- PRs should include:
  - a concise summary of behavior changes,
  - linked issue(s) when applicable,
  - test evidence (`pytest`, `ruff`, `mypy`) in the description.
- If CLI output changes, include a short before/after example in the PR body.
