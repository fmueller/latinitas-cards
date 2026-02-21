# LatinitasCards

`latinitas-cards` is a CLI for Latin Anki workflows:

- inspect Anki decks and field structure,
- split multi-form cards into single-form rows,
- annotate grammar with CLTK,
- generate cloze cards from Latin corpora (USFX, text, or CSV corpora),
- optionally rewrite APKG decks while keeping originals.

## Workflow

1. Inspect your deck schema and sample notes.
2. Split multi-form entries into one-form rows (CSV-first default).
3. Annotate grammar columns with CLTK.
4. Generate corpus-based clozes with optional difficulty filtering.

## CLI Examples

Inspect deck structure and show the first rows:

```bash
poetry run latinitas-cards inspect --input data/latin_university.apkg --head 5
```

Split a field with multi-form entries into one row per form:

```bash
poetry run latinitas-cards split \
  --input input.apkg \
  --output split.csv \
  --source-field Konstruktion_Hinweise \
  --split-mode auto
```

Annotate forms with CLTK grammar metadata:

```bash
poetry run latinitas-cards annotate \
  --input split.csv \
  --output annotated.csv \
  --form-column form
```

Generate cloze cards from one or more corpora:

```bash
poetry run latinitas-cards cloze \
  --input annotated.csv \
  --output cloze.csv \
  --corpus data/lat-clementine.usfx.xml \
  --corpus-format auto \
  --difficulty medium
```

Generate clozes from a parallel corpus CSV and explicitly include EN/DE columns:

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

Optional APKG rewrite (keeps originals and adds split cards):

```bash
poetry run latinitas-cards split \
  --input input.apkg \
  --output output.apkg \
  --source-field Konstruktion_Hinweise \
  --split-mode auto \
  --output-format apkg
```

## CLTK + Optional LLM Disambiguation

Grammar analysis uses [CLTK](https://cltk.org). If you add LLM-based disambiguation in your own pipeline,
the project default is Ollama with model `ministral-3:8b`.

## Corpora Suggestions (Latin + English/German)

Good public sources for Latin corpora (including EN/DE parallel data) include:

- [OPUS](https://opus.nlpl.eu/) (recommended starting point)
  - `bible-uedin` (strong verse-aligned biblical corpus)
  - `Tatoeba` (sentence-level data)
  - `WikiMatrix` / `CCMatrix` (broader but noisier)
- For direct corpus pair discovery via API:
  - `https://opus.nlpl.eu/opusapi/?corpora=True&source=la&target=en`
  - `https://opus.nlpl.eu/opusapi/?corpora=True&source=la&target=de`

When parallel columns are detected and behavior is unspecified:

- interactive terminal: `latinitas-cards` prompts you
- non-interactive execution: translations are ignored with a warning

## Validation Rules

After code changes, always run:

1. `poetry run ruff check`
2. `poetry run mypy`
3. `poetry run pytest -v`

If any step fails, fix issues and rerun the full chain from step 1.

## Repository Metadata Checklist

After updating project scope, also update GitHub repository metadata:

- Description: `CLI for Latin Anki workflows: split multi-form cards, annotate grammar with CLTK, and generate corpus-based cloze cards.`
- Topics: `latin`, `anki`, `cloze`, `flashcards`, `nlp`, `cltk`, `linguistics`, `education`, `python`, `cli`.
