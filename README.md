# LatinitasCards

[![CI](https://github.com/fmueller/latinitas-cards/actions/workflows/build.yml/badge.svg)](https://github.com/fmueller/latinitas-cards/actions/workflows/build.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-green)](LICENSE)

A CLI toolkit for building Latin Anki flashcards. Inspect and restructure Anki exports, annotate grammar with [CLTK](https://cltk.org), and generate corpus-based cloze-deletion cards from USFX, plain-text, or parallel CSV corpora — designed for learners studying Latin through spaced repetition.

## Features

- **Inspect** Anki deck structure, field names, and sample notes
- **Split** multi-form cards into one-row-per-form records
- **Annotate** grammar (lemma, POS, morphology) via CLTK with optional LLM disambiguation
- **Generate cloze cards** from Latin corpora (USFX XML, plain text, CSV)
- **Parallel corpus support** — include EN/DE translations alongside Latin clozes
- **Difficulty filtering** — control cloze complexity (easy / medium / hard)
- **APKG rewrite** — update Anki packages in place while preserving originals

## Installation

**Prerequisites:** Python 3.10–3.12 and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/fmueller/latinitas-cards.git
cd latinitas-cards
poetry install
```

Verify the installation:

```bash
poetry run latinitas-cards --help
```

## Quick Start

```bash
# 1. Inspect your Anki deck
poetry run latinitas-cards inspect --input data/latin_university.apkg --head 5

# 2. Split multi-form entries into individual rows
poetry run latinitas-cards split \
  --input data/latin_university.apkg \
  --output split.csv \
  --source-field Konstruktion_Hinweise \
  --split-mode auto

# 3. Annotate grammar and generate cloze cards
poetry run latinitas-cards annotate --input split.csv --output annotated.csv --form-column form
poetry run latinitas-cards cloze \
  --input annotated.csv \
  --output cloze.csv \
  --corpus data/lat-clementine.usfx.xml \
  --corpus-format auto \
  --difficulty medium
```

## CLI Reference

| Command      | Description                                                      |
|--------------|------------------------------------------------------------------|
| `inspect`    | Inspect deck schema and show a head-like sample preview          |
| `split`      | Split multi-form cards into one-row-per-form records             |
| `annotate`   | Annotate CSV forms with CLTK lemma/POS/morphology metadata       |
| `cloze`      | Generate corpus-based cloze cards for each form in a CSV input   |
| `validate`   | Validate USFX parsing integrity and required input columns       |
| `preview`    | Show a sample of generated clozes without writing output         |
| `generate`   | Update an Anki CSV or APKG file with cloze examples from a Latin USFX corpus |

### inspect

```bash
poetry run latinitas-cards inspect --input data/latin_university.apkg --head 5
```

### split

```bash
poetry run latinitas-cards split \
  --input input.apkg \
  --output split.csv \
  --source-field Konstruktion_Hinweise \
  --split-mode auto
```

Optional APKG rewrite (keeps originals and adds split cards):

```bash
poetry run latinitas-cards split \
  --input input.apkg \
  --output output.apkg \
  --source-field Konstruktion_Hinweise \
  --split-mode auto \
  --output-format apkg
```

### annotate

```bash
poetry run latinitas-cards annotate \
  --input split.csv \
  --output annotated.csv \
  --form-column form
```

With optional Ollama LLM disambiguation:

```bash
poetry run latinitas-cards annotate \
  --input split.csv \
  --output annotated_llm.csv \
  --form-column form \
  --use-llm \
  --llm-provider ollama \
  --llm-model ministral-3:8b \
  --llm-endpoint http://localhost:11434
```

### cloze

```bash
poetry run latinitas-cards cloze \
  --input annotated.csv \
  --output cloze.csv \
  --corpus data/lat-clementine.usfx.xml \
  --corpus-format auto \
  --difficulty medium
```

With a parallel corpus (including EN/DE translations):

```bash
poetry run latinitas-cards cloze \
  --input annotated.csv \
  --output cloze_parallel.csv \
  --corpus opus_subset.csv \
  --corpus-format csv \
  --latin-column la \
  --translation-lang en \
  --translation-lang de \
  --parallel-mode include
```

When parallel columns are detected and behavior is unspecified:

- Interactive terminal: `latinitas-cards` prompts you
- Non-interactive execution: translations are ignored with a warning

### validate

```bash
poetry run latinitas-cards validate \
  --input data/latin_university.apkg \
  --usfx data/lat-clementine.usfx.xml
```

### preview

```bash
poetry run latinitas-cards preview \
  --input data/latin_university.apkg \
  --usfx data/lat-clementine.usfx.xml
```

### generate

```bash
poetry run latinitas-cards generate \
  --input data/latin_university.apkg \
  --output updated.csv \
  --usfx data/lat-clementine.usfx.xml
```

## Corpora

The bundled corpus `data/lat-clementine.usfx.xml` is a Latin Vulgate (Clementine) Bible in USFX format.

Good public sources for additional Latin corpora (including EN/DE parallel data):

- [OPUS](https://opus.nlpl.eu/) (recommended starting point)
  - `bible-uedin` (strong verse-aligned biblical corpus)
  - `Tatoeba` (sentence-level data)
  - `WikiMatrix` / `CCMatrix` (broader but noisier)
- For direct corpus pair discovery via API:
  - `https://opus.nlpl.eu/opusapi/?corpora=True&source=la&target=en`
  - `https://opus.nlpl.eu/opusapi/?corpora=True&source=la&target=de`

## Contributing

See [AGENTS.md](AGENTS.md) for coding style, project structure, and commit conventions.

Before submitting changes, run the validation chain:

```bash
poetry run ruff check
poetry run mypy
poetry run pytest -v
```

## License

[GPL-3.0-or-later](LICENSE)
