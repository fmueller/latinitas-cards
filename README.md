# Latinitas Cards

[![CI](https://github.com/fmueller/latinitas-cards/actions/workflows/build.yml/badge.svg)](https://github.com/fmueller/latinitas-cards/actions/workflows/build.yml)
[![Python 3.13 | 3.14](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue)](https://www.python.org/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-green)](LICENSE)

Latinitas Cards is an offline-first CLI for turning an existing Latin Anki deck or
CSV export into repeatable principal-part study cards. It inspects a source, saves a
confirmed profile, previews generated cards, and writes a deterministic UTF-8 Anki
text-import CSV without modifying the source.

> **Status:** v0.1.0 is still being prepared. Nothing has been released or published:
> there is no published package, tag, or GitHub release to install yet.

## v0.1.0 workflow

The supported workflow is **deck-first**:

1. Inspect a CSV, APKG, or COLPKG source and confirm its field mapping.
2. Save the human-readable, versioned profile and reuse it on later runs.
3. Preview principal-part completion and recognition cards.
4. Generate a deterministic CSV with `LatinitasID` first for repeat imports.
5. Import the CSV into a prepared Anki note type, then repeat the import with the
   same note type and first-field matching.

Profiles select recipes explicitly; the approved representative profile uses
`principal_part_completion` and `principal_part_recognition`. It uses the semantic role
`perfect_passive_participle` (PPP), not the legacy `supine` role. CLI controls and
diagnostics are English; generated study content is German with the profile's explicit
`language_tag` (default `de`).

The profile workflow never mutates the source CSV/APKG/COLPKG. ID-less CSV sources
require explicit identity reconciliation decisions on first use or when review items
appear; unchanged rows with a valid manifest are reused automatically. See the focused
[identity contract](docs/stable-generated-note-identity.md) and
[CSV/import workflow](docs/deterministic-csv-export.md) for ownership, recovery, and
Anki setup details.

## Installation

**Prerequisites:** Python 3.13 or 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/fmueller/latinitas-cards.git
cd latinitas-cards
uv sync --locked
uv run latinitas-cards --help
```

The optional `annotate` command is experimental and needs the CPU-only extra. It
installs CLTK 2, Stanza, and CPU-only PyTorch; the first run downloads Latin Stanza
models:

```bash
uv sync --locked --extra annotate
```

> [!WARNING]
> **Highly experimental:** `uv sync --locked --extra annotate-gpu` selects CUDA
> PyTorch instead. It needs a matching NVIDIA driver, is not tested in CI, and cannot
> be combined with the CPU `annotate` extra. Core deck-first workflows do not require
> CLTK, Stanza, PyTorch, or network access.
>
> `annotate --use-llm` sends source-derived forms and analyses to the configured Ollama
> endpoint. Do not use a remote endpoint with private deck data without reviewing that
> endpoint's privacy and retention policy.

## Quick start

The repository's only deck fixture is a committed, sanitized APKG used for smoke
tests. It is not a private university deck or a product-specific input. Replace the
fixture path below with your own source when using the workflow:

```bash
workdir="$(mktemp -d)"
profile="$workdir/profile.json"
output="$workdir/generated-principal-parts.csv"
source="tests/fixtures/representative-university-latin.apkg"

# Propose, explicitly confirm, and save a reusable profile.
uv run latinitas-cards setup \
  --input "$source" \
  --profile "$profile" \
  --recipe principal_part_completion \
  --recipe principal_part_recognition \
  --non-interactive \
  --confirm \
  --json

# Preview without writing the generated CSV.
uv run latinitas-cards preview \
  --input "$source" \
  --profile "$profile" \
  --limit 2

# Preview again, then write deterministic UTF-8 Anki text-import CSV.
uv run latinitas-cards generate \
  --input "$source" \
  --profile "$profile" \
  --output "$output"
```

For an interactive setup, omit `--non-interactive --confirm`; review the proposed
examples and answer the confirmation prompt. Repeat `--role` in semantic order when
correcting a proposal, for example:

```bash
uv run latinitas-cards setup \
  --input source.csv \
  --profile .latinitas/profile.json \
  --role present_infinitive \
  --role present_1s \
  --role perfect_1s \
  --role perfect_passive_participle
```

Prepare the generated note type, fields, template, and HTML import setting before the
first import; the note type keeps a user-owned `Personal Notes` field that generated CSVs
never contain. For repeat imports, use the same note type with `LatinitasID` as the
first/matching field; `Personal Notes` stays unmapped because no CSV column exists for it.
Map a legacy CSV's `Personal Notes` column to **Ignore field** on every repeat import. The
[deterministic CSV export guide](docs/deterministic-csv-export.md) has the complete
checklist, including native Anki behavior already verified in a disposable collection;
retest it there before relying on it.

Treat setup JSON, terminal previews, generated CSVs, and identity manifests as
source-derived data: they can contain study text, stable IDs, and provenance. Redact
them before sharing an issue or other public report.

## Experimental and planned capabilities

These remain available for legacy workflows, not v0.1.0 release promises:

- `split --output-format apkg`: experimental APKG mutation and split-note cloning;
- `cloze` and `preview`/`generate --usfx`: experimental corpus-based generation; and
- `annotate`: experimental CLTK/Stanza analysis, with optional Ollama disambiguation.

Structural parser counts do not establish eligible-verb or universal morphological
coverage; full grammatical parsing and managed live-Anki updates are not implemented.
PPP and supine remain distinct semantic roles.

## Documentation

- [Changelog](CHANGELOG.md)
- [v0.1.0 release readiness](docs/release-readiness.md)
- [Stable generated-note identity](docs/stable-generated-note-identity.md)
- [Principal-part parser support matrix](docs/principal-part-parsing.md)
- [Sanitized representative-deck validation](docs/representative-deck-validation.md)
- [Deterministic CSV export and Anki import](docs/deterministic-csv-export.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [License compatibility audit](docs/license-compatibility-audit.md)

## Development and contribution

Set up the pinned development toolchain and local hooks with
[mise](https://mise.jdx.dev):

```bash
mise run setup
```

Read [AGENTS.md](AGENTS.md) for the project structure, coding conventions, test
commands, and contribution/commit policy. The required local checks are:

```bash
uv run ruff check
uv run mypy
uv run pytest -v
```

## Support

There is no support SLA or live Anki-collection assistance. Report reproducible issues
through the [GitHub issue tracker](https://github.com/fmueller/latinitas-cards/issues)
with sanitized commands and error output; do not upload private decks or credentials.

## License

[GPL-3.0-or-later](LICENSE)
