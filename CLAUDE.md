# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LatinitasCards generates Anki flashcards for learning Latin. The main feature parses a Latin Vulgate Bible in USFX XML format, matches Latin words from an Anki deck export (CSV or .apkg/.colpkg), and produces cloze-deletion cards with verse references.

## Commands

```bash
poetry install                  # Install dependencies
poetry run pytest -v            # Run all tests
poetry run pytest tests/unit/cli_test.py  # Run a single test file
poetry run pytest -k "test_name"          # Run a specific test by name
poetry run ruff check           # Lint
poetry run ruff check --fix     # Lint with auto-fix
poetry run ruff format          # Format code
poetry run mypy                 # Type check (strict mode)
```

## Architecture

Single-module CLI tool using a `src/` layout. All logic lives in `src/latinitas_cards/cli.py`:

- **CLI entry point**: Typer app exposed as `latinitas-cards` via `tool.poetry.scripts`. Entry via `__main__.py`.
- **USFX parser** (`parse_usfx_to_df`): Walks Vulgate XML using a manual stack-based traversal, extracting book/chapter/verse/text into a pandas DataFrame.
- **Bucket index** (`build_bucket_index`): First-letter bucketing of normalized verse text for fast word lookup.
- **Cloze generator** (`generate_clozes_for_word`): Finds verses containing a word and wraps the first occurrence in Anki `{{c1::...}}` syntax.
- **Input loader** (`_load_input_to_dataframe`): Reads either CSV (with header auto-detection) or .colpkg (extracts SQLite from zip, reads notes table).
- **Latin normalization** (`normalize_latin`): Lowercases and normalizes æ→ae, œ→oe, j→i.

## Code Conventions

- Python >=3.10, <3.14. CI tests on 3.10 and 3.13.
- Ruff for linting (rules: E, F, UP, B, SIM, I) and formatting. Line length 120.
- mypy strict mode (`target: 3.10`).
- Test files named `*_test.py` (enforced by pre-commit `name-tests-test` hook).
- Pre-commit hooks run ruff, ruff-format, and standard checks (trailing whitespace, TOML/YAML validation, LF line endings).
