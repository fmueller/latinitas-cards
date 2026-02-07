# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python CLI project using a `src/` layout.

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
- `poetry run ruff check`: run lint checks.
- `poetry run ruff check --fix`: auto-fix lint issues where possible.
- `poetry run ruff format`: format code.
- `poetry run mypy`: run strict type checks.
- `poetry run latinitas-cards --help`: inspect CLI usage.

## Coding Style & Naming Conventions
- Target Python `>=3.10,<3.14`; keep code compatible with 3.10.
- Use Ruff for formatting and linting (line length 120, double quotes, import sorting via `I` rules).
- mypy is strict; add explicit types on public functions and non-trivial internals.
- Follow snake_case for functions/variables and lowercase module names.
- Keep modules focused; prefer small helpers over large monolithic functions.

## Testing Guidelines
- Framework: `pytest`.
- Test files must be named `*_test.py` (enforced by pre-commit hook).
- Place tests under `tests/unit/` mirroring package behavior.
- Add regression tests for parser edge cases and CLI input/output changes.
- Run `poetry run pytest -v` before opening a PR.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style seen in history: `feat:`, `fix:`, `chore:`, `ci:`.
- Keep commits focused and atomic (code + tests for the same change).
- PRs should include:
  - a concise summary of behavior changes,
  - linked issue(s) when applicable,
  - test evidence (`pytest`, `ruff`, `mypy`) in the description.
- If CLI output changes, include a short before/after example in the PR body.
