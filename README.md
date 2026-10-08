# Latinitas Cards

[![CI](https://github.com/fmueller/latinitas-cards/actions/workflows/build.yml/badge.svg)](https://github.com/fmueller/latinitas-cards/actions/workflows/build.yml)
[![Python 3.13 | 3.14](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue)](https://www.python.org/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-green)](LICENSE)

Latinitas Cards is an offline-first CLI for turning an existing Latin Anki deck or
CSV export into repeatable principal-part study cards. It inspects a source, saves a
confirmed profile, previews generated cards, and writes a deterministic UTF-8 Anki
text-import CSV without modifying the source.

> **Release:** [v0.1.1](https://github.com/fmueller/latinitas-cards/releases/tag/v0.1.1)
> is available on GitHub. PyPI publishing is deferred; install from the release tag below.

## Workflow

The supported workflow is **deck-first**:

1. Inspect a CSV, APKG, or COLPKG source and confirm its field mapping.
2. Save the human-readable, versioned profile and reuse it on later runs.
3. Preview principal-part completion and recognition cards.
4. Generate a deterministic CSV with `LatinitasID` first for repeat imports.
5. Import the CSV into a prepared Anki note type, then repeat the import with the
   same note type and first-field matching.

Also available: [`managed` updates](#updating-imported-notes-managed) for notes you
already imported, [explanations and themes](#explanations-and-themes) on principal-part
answers, and reviewed `form-parsing` exercises.

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
git clone --branch v0.1.1 --depth 1 https://github.com/fmueller/latinitas-cards.git
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

# Write deterministic UTF-8 Anki text-import CSV. The first export needs
# --approve-fresh-import; later exports reuse the checkpoint state that is
# saved next to the source as <source>.latinitas-cards.json.
uv run latinitas-cards generate \
  --input "$source" \
  --profile "$profile" \
  --output "$output" \
  --approve-fresh-import
```

For an interactive setup, omit `--non-interactive --confirm`; review the proposed
examples and answer the confirmation prompt. Field suggestions rank sampled content
(values matching principal-part forms, single-word shapes) alongside field names, and
the proposal shows sample values and the reason for each suggestion. When the evidence
ties, conflicts, or is too sparse, setup requires an explicit field choice — pass
`--lexical-entry-field`/`--principal-parts-field` or answer the interactive choice
prompt — before a profile can be saved. Repeat `--role` in semantic order when
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

For the [reviewed source layouts](docs/source-extraction-fixtures.md), the saved
profile's `principal_parts` section can set `pipe_alternatives: true` (keep `|`
alternatives within a role) and `trailing_poet_hint: true` (keep a trailing `poet.`
after the perfect form as a hint). Both are off by default; check representative values
before turning them on. Alternatives and hints are kept as source evidence, never
guessed into a single answer: unresolved forms are left off the cards and listed in the
preview. Zero-card entries are not exported.

Prepare the generated note type, fields, template, and HTML import setting before the
first import; the note type keeps a user-owned `Personal Notes` field that generated CSVs
never contain. For repeat imports, use the same note type with `LatinitasID` as the
first/matching field; `Personal Notes` stays unmapped because no CSV column exists for it.
Map a legacy CSV's `Personal Notes` column to **Ignore field** on every repeat import. The
[deterministic CSV export guide](docs/deterministic-csv-export.md) has the complete
checklist, including native Anki behavior already verified in a disposable collection;
retest it there before relying on it.

Collections from earlier pre-release Latinitas models are never silently converted: the
[legacy transition policy](docs/legacy-transition.md) keeps such data read-only, retains
collections with valuable history or personal annotations, and offers only an explicitly
approved, backed-up fresh start into a new dedicated note type for disposable data.

Treat setup JSON, terminal previews, generated CSVs, and identity manifests as
source-derived data: they can contain study text, stable IDs, and provenance. Redact
them before sharing an issue or other public report.

## Updating imported notes (`managed`)

`managed` prepares a reviewed update CSV for notes already in your collection, so
their scheduling can be kept under the offline checklist. It never opens Anki's
collection backend; `capture` reads only an explicitly attested closed backup copy.
You import the update file in Anki.

Try the complete sequence on a **sanitized disposable fixture**, without assembling
snapshot or baseline JSON by hand (run from this repository):

```bash
work=$(mktemp -d)
uv run --with anki==26.9.3 python scripts/check-managed-capture.py \
  --output-dir "$work/evidence"
```

This runs the installed CLI's `capture`, `adopt`, `plan`, `approve`, `emit`, a
fixture-only native import, another closed capture, and `observe`. It writes every
request/result plus `commands.json` and checks full card/history/model/deck tables,
personal text and manual tags. It does not verify the Desktop dialog or AnkiMobile.

For your destination, first follow the [closed-backup acquisition and adoption
guide](docs/managed-capture-and-adoption.md). Schema-18 protobuf definitions are
optional; no Anki runtime is added to the default installation. The selection file
records your collection/profile binding and the **whole immutable source/object
membership**; the ownership file records each note's explicitly reviewed tag origins
and keep overrides. Neither visible Latin text nor a deck name supplies identity.

```bash
uv run --with anki==26.9.3 latinitas-cards managed capture closed-backup.anki2 \
  --selection selection.json --closed-backup --interval-confirmed > snapshot.json
uv run latinitas-cards managed adopt snapshot.json --ownership ownership.json \
  --review 'reviewed each note and its ownership' --state baseline.json
uv run latinitas-cards managed plan request.json > plan.json      # what would change
uv run latinitas-cards managed approve plan.json \
  --operation '<id from plan>' --review 'why this change is OK' > approval.json
uv run latinitas-cards managed emit handoff.json --state baseline.json --output updates.csv
# Import updates.csv in Anki, then record what actually happened:
uv run latinitas-cards managed observe observation.json --state baseline.json --interval-confirmed
```

Back up first and don't review or sync on any device until the result is recorded.
Only content fields and tags can be updated; adding, retiring, or reactivating cards is
not supported. Capture/adoption are not apply approval. Proposals and the plan/handoff/
observation envelopes remain explicit JSON; the runnable fixture demonstrates their
construction, and the format and full checklist are in
[destination snapshots and file transport](docs/destination-snapshots-and-file-transport.md).

## Explanations and themes

Set the look of principal-part answers in the saved profile's `morphology` section;
`setup` has no flags for it.

```json
"morphology": {"version": 1, "theme": "muted", "appearance": "light", "comparison": "static"}
```

`theme` is `muted` or `monochrome`; `appearance` is `light` or `dark` (fixed, not
following device night mode); `comparison` is `static` (default) or `disclosure`.
Paste the reference CSS into your note type by hand; see
[reference note type](docs/reference-note-type.md#morphology-presentation-contract).

## Experimental and planned capabilities

These remain available for legacy workflows, not release promises:

- `split --output-format apkg`: experimental APKG mutation and split-note cloning;
- `cloze` and `preview`/`generate --usfx`: experimental corpus-based generation; and
- `annotate`: experimental CLTK/Stanza analysis, with optional Ollama disambiguation.

Structural parser counts do not establish eligible-verb or universal morphological
coverage; full grammatical parsing and live (in-Anki) updates are not implemented.
PPP and supine remain distinct semantic roles.

`form-parsing` offers offline, individually reviewed contextual feature preview and
fresh CSV export, not automatic analysis or managed application. Display checks
passed only for test notes on Anki Desktop 26.09.2 and AnkiMobile 25.09. See the
[review policy and separate manual reference setup](docs/contextual-form-parsing.md).

Authored JSONL study notes can be validated, previewed, and exported without corpus
resources or profile mappings:

```bash
uv run latinitas-cards authored validate notes.jsonl --namespace my-course
uv run latinitas-cards authored preview notes.jsonl --namespace my-course --kind vocab
mkdir -p authored-csv
uv run latinitas-cards authored export notes.jsonl --namespace my-course \
  --output-dir authored-csv --deck 'Latin::Authored'
```

Validation and preview write nothing; export writes one CSV per selected kind and
never changes the input. See the [authored import guide](docs/authored-note-import.md) for combined filters,
whole-file diagnostics, and first/repeat Anki import with Personal Notes unmapped.

For loosely structured Markdown, ask your repository agent to use
[`extracting-authored-notes`](.agents/skills/extracting-authored-notes/SKILL.md)
(also installed under `.claude/skills/`). Supply notes, language, a stable namespace,
and any existing JSONL. Review QA keys and the validated preview before export;
re-extraction preserves keys and skip decisions and reports new/changed/missing items.

## Documentation

- [Changelog](CHANGELOG.md)
- [v0.1.0 release readiness](docs/release-readiness.md)
- [v0.1.1 release readiness and limitations](docs/release-v0.1.1.md)
- [Authored note JSONL import format](docs/authored-note-import.md)
- [Stable generated-note identity](docs/stable-generated-note-identity.md)
- [Principal-part parser support matrix](docs/principal-part-parsing.md)
- [Sanitized representative-deck validation](docs/representative-deck-validation.md)
- [Deterministic CSV export and Anki import](docs/deterministic-csv-export.md)
- [Reference note type and safe import](docs/reference-note-type.md)
- [Legacy note-model transitions](docs/legacy-transition.md)
- [Destination snapshots and file transport](docs/destination-snapshots-and-file-transport.md)
- [Managed CSV native verification](docs/managed-csv-native-verification.md)
- [Managed CSV Desktop dialog verification](docs/managed-csv-desktop-verification.md)
- [Claim review policy](docs/claim-review-policy.md)
- [Contextual form parsing](docs/contextual-form-parsing.md)
- [Morphology native verification](docs/morphology-native-verification.md)
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
