# Third-Party Notices

This project (`latinitas-cards`) is licensed under GPL-3.0-or-later.

The project depends on the following third-party runtime libraries (from `pyproject.toml` / `poetry.lock`):

- `pydantic` — MIT
- `pandas` — BSD
- `rich` — MIT
- `typer` — MIT
- `zstandard` — BSD-3-Clause

Plus transitive runtime dependencies resolved in `poetry.lock` (e.g., `click`, `numpy`, `python-dateutil`, `pytz`, `shellingham`, `typing-extensions`, etc.), all reviewed as GPLv3-compatible in `docs/license-compatibility-audit.md`.

For full license compatibility analysis and dependency-by-dependency details, see:

- `docs/license-compatibility-audit.md`

When redistributing packaged artifacts that bundle dependencies, include corresponding third-party license notices/texts as required by each dependency license.
