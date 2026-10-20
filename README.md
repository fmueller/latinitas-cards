# Latinitas Cards

[![CI](https://github.com/fmueller/latinitas-cards/actions/workflows/build.yml/badge.svg)](https://github.com/fmueller/latinitas-cards/actions/workflows/build.yml)
[![Python 3.13 | 3.14](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue)](https://www.python.org/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-green)](https://github.com/fmueller/latinitas-cards/blob/main/LICENSE)

Latinitas Cards is an offline-first CLI for turning an existing Latin Anki deck or
CSV export into repeatable principal-part study cards. It inspects a source, saves a
confirmed profile, previews generated cards, and writes a deterministic UTF-8 Anki
text-import CSV without modifying the source.

> **Current release:** [v0.3.0](https://github.com/fmueller/latinitas-cards/releases/tag/v0.3.0),
> available on [PyPI](https://pypi.org/project/latinitas-cards/0.3.0/).
> See [release readiness and limits](https://github.com/fmueller/latinitas-cards/blob/main/docs/release-v0.3.0.md)
> for the verified publication evidence and compatibility boundaries.

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
answers, reviewed `form-parsing` exercises, and
[reviewed lexical/pattern selections and bounded phrase proposals](https://github.com/fmueller/latinitas-cards/blob/main/docs/reviewed-phrase-inputs.md)
via `phrases`, with separately reviewed translation/recognition exercises and offline
eligibility checks, deterministic approved CSV export and managed retained-slot updates.
Initial creation is manual CSV import plus explicit adoption; withdrawal remains a
blocked plan requiring reviewed manual resolution, not CSV omission or automatic retirement.

Profiles select recipes explicitly; the approved representative profile uses
`principal_part_completion` and `principal_part_recognition`. It uses the semantic role
`perfect_passive_participle` (PPP), not the legacy `supine` role. CLI controls and
diagnostics are English; generated study content is German with the profile's explicit
`language_tag` (default `de`).

The profile workflow never mutates the source CSV/APKG/COLPKG. ID-less CSV sources
require explicit identity reconciliation decisions on first use or when review items
appear; unchanged rows with a valid manifest are reused automatically. See the focused
[identity contract](https://github.com/fmueller/latinitas-cards/blob/main/docs/stable-generated-note-identity.md) and
[CSV/import workflow](https://github.com/fmueller/latinitas-cards/blob/main/docs/deterministic-csv-export.md) for ownership, recovery, and
Anki setup details.

## Installation

**Prerequisites:** Python 3.13 or 3.14. Linux and macOS are supported; Windows is not supported.
The default install needs no CLTK, Stanza or PyTorch.

### PyPI installation

[Latinitas Cards 0.3.0 is published on PyPI](https://pypi.org/project/latinitas-cards/0.3.0/).
Choose one isolated install:

```bash
uv tool install latinitas-cards
# Or, with pipx:
pipx install latinitas-cards
# Or, with pip inside an activated virtual environment:
python -m venv .venv
source .venv/bin/activate
pip install latinitas-cards
```

Upgrade using the same installer (activate the venv first for pip):

```bash
uv tool upgrade latinitas-cards
pipx upgrade latinitas-cards
pip install --upgrade latinitas-cards
```

The bundled Vulgate is not in the wheel or sdist. For corpus workflows, supply
your own USFX path and optionally `--descriptor FILE` for corpus metadata.

### Git development checkout

The development checkout remains available with [uv](https://docs.astral.sh/uv/).
The command below checks out the published v0.3.0 tag. To work on the latest
development version instead, clone without `--branch v0.3.0`.

```bash
git clone --branch v0.3.0 --depth 1 https://github.com/fmueller/latinitas-cards.git
cd latinitas-cards
uv sync --locked
uv run latinitas-cards --help
```

### Experimental annotation extras

Both `annotate` and `annotate-gpu` are experimental; the two extras must not be combined.
`annotate` downloads Stanza models on first run (network and disk space required).
For a checkout, the CPU extra installs CLTK 2, Stanza and CPU-only PyTorch:

```bash
uv sync --locked --extra annotate
```

uv's CPU torch index and extra conflict do not transfer to pip. Use this CPU-extra
command inside the activated venv on Linux:

```bash
pip install 'latinitas-cards[annotate]' --extra-index-url https://download.pytorch.org/whl/cpu
```

On macOS, PyPI's torch wheels are already CPU-only; use
`pip install 'latinitas-cards[annotate]'`. pip does not enforce uv's extra conflict:
choose one extra, never both. The additional index participates in dependency
resolution; use only trusted indexes and review the resolved packages.

> [!WARNING]
> **Highly experimental:** `uv sync --locked --extra annotate-gpu` selects CUDA
> PyTorch instead on compatible Linux/NVIDIA systems. The pip equivalent is
> `pip install 'latinitas-cards[annotate-gpu]'` in a separate venv,
> without the CPU index. It needs a matching NVIDIA driver and is not tested in CI.
> This is not a claim of verified macOS GPU or Windows support. Core deck-first
> workflows do not require CLTK, Stanza, PyTorch, or network access.
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

For the [reviewed source layouts](https://github.com/fmueller/latinitas-cards/blob/main/docs/source-extraction-fixtures.md), the saved
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
[deterministic CSV export guide](https://github.com/fmueller/latinitas-cards/blob/main/docs/deterministic-csv-export.md) has the complete
checklist, including native Anki behavior already verified in a disposable collection;
retest it there before relying on it.

Collections from earlier pre-release Latinitas models are never silently converted: the
[legacy transition policy](https://github.com/fmueller/latinitas-cards/blob/main/docs/legacy-transition.md) keeps such data read-only, retains
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
guide](https://github.com/fmueller/latinitas-cards/blob/main/docs/managed-capture-and-adoption.md). Schema-18 protobuf definitions are
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
Backup recoverability, snapshot freshness and the edit-free interval are operator
attestations: a nonempty backup/hash is not restoration proof, and the tool enforces
no snapshot age bound or collection lock. Emission is supported offline handoff,
not automated collection application or proof of a successful native import.
Only content fields and tags can be updated; adding, retiring, or reactivating cards is
not supported. Capture/adoption are not apply approval. Proposals and the plan/handoff/
observation envelopes remain explicit JSON; the runnable fixture demonstrates their
construction, and the format and full checklist are in
[destination snapshots and file transport](https://github.com/fmueller/latinitas-cards/blob/main/docs/destination-snapshots-and-file-transport.md).

## Explanations and themes

Set the look of principal-part answers in the saved profile's `morphology` section;
`setup` has no flags for it.

```json
"morphology": {
  "version": 1, "theme": "muted", "appearance": "light", "comparison": "static",
  "terminology": {"version": 1, "language": "latin"}
}
```

`theme` is `muted` or `monochrome`; `appearance` is `light` or `dark` (fixed, not
following device night mode); `comparison` is `static` (default) or `disclosure`.
`setup --reconfigure` preserves these saved settings while proposing source mappings
again. An explicit `morphology` object must include `"version": 1` and versioned
terminology. Explicitly upgrading a legacy profile uses muted/light/static defaults
when the whole section was absent.
Paste the reference CSS into your note type by hand; see
[reference note type](https://github.com/fmueller/latinitas-cards/blob/main/docs/reference-note-type.md#morphology-presentation-contract).

### Grammatical terminology

Principal-part answers and contextual form-parsing answers default to Latin terminology.
To use German, add `"terminology": {"version": 1, "language": "user"}` inside
`morphology`; `"language": "latin"` selects Latin. `user` uses the profile's
`language_tag` and currently supports only German (`de`, including regional tags).
Unknown settings, unsupported versions and unsupported user languages fail validation.

Legacy profiles without terminology are rejected on ordinary load. Explicitly upgrade
to Latin defaults with `latinitas-cards terminology upgrade old-profile.json new-profile.json`;
the old file is untouched and existing output files are never overwritten.
An explicitly supplied `terminology` object must include `"version": 1`.
Both `preview`/`generate --profile` and `form-parsing preview`/`export` accept
`--terminology latin|user` and report the effective setting and override.
Form-parsing reports go to stderr so stdout remains JSON.

Controlled feature values (such as `nominativus` and `pluralis`) remain canonical
in reviewed claims and display as `Kasus: Nominativ`, `Numerus: Plural` in German.
Values outside the vocabulary still need explicit individual review and display
verbatim, never machine-translated. Prompts and German explanatory prose are unchanged.
Changing terminology preserves claim fingerprints, linguistic reviews, identities and
frozen card slots. Changed answer payloads require renewed managed approval, not renewed
linguistic review. Separately approved retained-slot updates are available for current
principal-part, parsing and phrase layouts. CSV installs no templates or CSS; old layout
digests fail closed and need reviewed manual setup, never silent scheduled-template rebinding.

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
[review policy and separate manual reference setup](https://github.com/fmueller/latinitas-cards/blob/main/docs/contextual-form-parsing.md).

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
never changes the input. See the [authored import guide](https://github.com/fmueller/latinitas-cards/blob/main/docs/authored-note-import.md) for combined filters,
whole-file diagnostics, and first/repeat Anki import with Personal Notes unmapped.

For loosely structured Markdown, ask your repository agent to use
[`extracting-authored-notes`](https://github.com/fmueller/latinitas-cards/blob/main/.agents/skills/extracting-authored-notes/SKILL.md)
(also installed under `.claude/skills/`). Supply notes, language, a stable namespace,
and any existing JSONL. Review QA keys and the validated preview before export;
re-extraction preserves keys and skip decisions and reports new/changed/missing items.

## Documentation

- [Maintainer PyPI release runbook](https://github.com/fmueller/latinitas-cards/blob/main/docs/releasing.md)
- [Changelog](https://github.com/fmueller/latinitas-cards/blob/main/CHANGELOG.md)
- [v0.1.0 release readiness](https://github.com/fmueller/latinitas-cards/blob/main/docs/release-readiness.md)
- [v0.1.1 release readiness and limitations](https://github.com/fmueller/latinitas-cards/blob/main/docs/release-v0.1.1.md)
- [v0.2.0 release readiness and limitations](https://github.com/fmueller/latinitas-cards/blob/main/docs/release-v0.2.0.md)
- [v0.2.1 release readiness, accepted Linux native evidence and limits](https://github.com/fmueller/latinitas-cards/blob/main/docs/release-v0.2.1.md)
- [v0.3.0 publication evidence, offline walkthrough and limits](https://github.com/fmueller/latinitas-cards/blob/main/docs/release-v0.3.0.md)
- [Authored note JSONL import format](https://github.com/fmueller/latinitas-cards/blob/main/docs/authored-note-import.md)
- [Stable generated-note identity](https://github.com/fmueller/latinitas-cards/blob/main/docs/stable-generated-note-identity.md)
- [Principal-part parser support matrix](https://github.com/fmueller/latinitas-cards/blob/main/docs/principal-part-parsing.md)
- [Sanitized representative-deck validation](https://github.com/fmueller/latinitas-cards/blob/main/docs/representative-deck-validation.md)
- [Deterministic CSV export and Anki import](https://github.com/fmueller/latinitas-cards/blob/main/docs/deterministic-csv-export.md)
- [Reference note type and safe import](https://github.com/fmueller/latinitas-cards/blob/main/docs/reference-note-type.md)
- [Legacy note-model transitions](https://github.com/fmueller/latinitas-cards/blob/main/docs/legacy-transition.md)
- [Destination snapshots and file transport](https://github.com/fmueller/latinitas-cards/blob/main/docs/destination-snapshots-and-file-transport.md)
- [Managed CSV native verification](https://github.com/fmueller/latinitas-cards/blob/main/docs/managed-csv-native-verification.md)
- [Managed CSV Desktop dialog verification](https://github.com/fmueller/latinitas-cards/blob/main/docs/managed-csv-desktop-verification.md)
- [Claim review policy](https://github.com/fmueller/latinitas-cards/blob/main/docs/claim-review-policy.md)
- [Reviewed claim CLI and offline checks](https://github.com/fmueller/latinitas-cards/blob/main/docs/reviewed-claim-cli.md)
- [Contextual form parsing](https://github.com/fmueller/latinitas-cards/blob/main/docs/contextual-form-parsing.md)
- [Morphology native verification](https://github.com/fmueller/latinitas-cards/blob/main/docs/morphology-native-verification.md)
- [Third-party notices](https://github.com/fmueller/latinitas-cards/blob/main/THIRD_PARTY_NOTICES.md)
- [License compatibility audit](https://github.com/fmueller/latinitas-cards/blob/main/docs/license-compatibility-audit.md)

## Development and contribution

Set up the pinned development toolchain and local hooks with
[mise](https://mise.jdx.dev):

```bash
mise run setup
```

Read [AGENTS.md](https://github.com/fmueller/latinitas-cards/blob/main/AGENTS.md) for the project structure, coding conventions, test
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

[GPL-3.0-or-later](https://github.com/fmueller/latinitas-cards/blob/main/LICENSE)
