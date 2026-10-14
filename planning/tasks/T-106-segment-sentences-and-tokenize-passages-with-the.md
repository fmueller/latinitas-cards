---
id: T-106-segment-sentences-and-tokenize-passages-with-the
title: Segment sentences and tokenize passages with the fixed latin-seg and latin-tok rules
status: todo
priority: high
spec_ref: specs/v0.3.0.md#normalized-jsonl-corpus-model
dependencies:
    - T-104-define-versioned-corpus-records-with-a-canonical
updated_at: "2026-10-14T05:42:59Z"
---

# T-106-segment-sentences-and-tokenize-passages-with-the Segment sentences and tokenize passages with the fixed latin-seg and latin-tok rules

## Description

Implement the deterministic rule tokenizer `latin-tok/1` and sentence segmenter `latin-seg/1`
exactly as fixed in `specs/v0.3.0.md#fixed-contracts` (**Identity and hashing**) for the
segmentation described in `specs/v0.3.0.md#normalized-jsonl-corpus-model` ("Sentences and
tokens are derived segmentation ... without CLTK; a sentence never spans passages"). The code
today has only `cli.normalize_latin` and regex word splitting for cloze; neither is reused
because their output is not frozen. This task delivers pure functions over canonical passage
text and passage sequences. Wrapping results into `segmentation` annotation records, their
`annotation_id`, and the `annotations segment` writer belong to T-113.

## Acceptance

- `tokenize(text) -> tuple[Token, ...]` in `src/latinitas_cards/corpus_segmentation.py`:
  tokens are maximal runs of `[\w\u0300-\u036F]+` (Python `re`, Unicode) or a single
  `[^\w\s]` code point, with `kind` `word` or `punct` and code-point offsets `[start, end)`
  into `text`; whitespace is never a token; enclitics (`Dixitque`, `virumque`) stay one token;
  an apostrophe is its own `punct` token, both U+0027 and U+2019. Use the executable
  Python pattern `r"[\w\u0300-\u036F]+|[^\w\s]"`; no special joining or extra escaping.
  Literal regression vectors: `d'arma` -> `d`, `'`, `arma` with spans `[0,1)`, `[1,2)`,
  `[2,6)`, and `d’arma` with the same code-point spans; neither is one word token.
- `segment_passage(text, tokens)` ends a sentence after a `.`, `?` or `!` token plus any
  closer tokens (`"`, `'`, `)`, `»`, `”`) that start exactly at the previous token's end
  offset; `:` and `;` never end a sentence; there is no abbreviation list. Sentence spans run
  from the first to the last token's offsets and carry token index ranges.
- `segment_work(passages)` processes passages of one `work` in `order`: a passage whose last
  sentence has no terminator sets `continues: true` on it, and the first sentence of the next
  passage in the same work sets `continued: true`; a new work resets the flag; no sentence
  spans two passages.
- `tests/fixtures/corpus/segmentation/cases.json` holds hand-written expected tokens
  (text, kind, start, end) and sentences for at least: Vulgate GEN 1:1 and GEN 1:3
  (`Dixitque Deus : Fiat lux. Et facta est lux.`, French `" : "` spacing), `?` and `!`
  terminators, closers `."`, `.'`, `.)`, `.»`, `.”`, `?»` and a spaced closer that does
  not attach, `;` and `:` non-terminators, an apostrophe, digits and `_` inside `\w`,
  `cælum`/`œ` and a combining-macron word (non-ASCII code-point offsets checked by slicing
  `text[start:end]`), a passage that is only punctuation, and a three-passage work with
  GEN 1:14 style (ends with `:`) `continues`/`continued` plus a work boundary. Expected
  values are written from the rules before the implementation (RED first), never generated
  by the code under test.
- A property test over every passage of `tests/fixtures/corpus/valid/*.jsonl` asserts
  tokens are ordered, non-overlapping, cover every non-whitespace code point exactly once,
  and that sentence ranges partition the token list.
- Determinism: the module has no locale, clock, hash-seed or CLTK dependency; a test imports
  it with `sys.modules["cltk"] = None` and asserts identical output, and a static test greps
  the module for `cltk`, `stanza`, `random`, `time`, `locale` imports.
- Rule identifiers are exported as constants `LATIN_TOK = ("latin-tok", "1")` and
  `LATIN_SEG = ("latin-seg", "1")` for the producer `{kind: "rule", name: "latin-seg",
  version: "1"}` that T-113 records (`producer.version` is a non-empty string).
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- Input is canonical passage `text` from `corpus_model` (already NFC and whitespace-
  collapsed); do not re-normalize or fold, and never use `norm` for offsets.
- Offsets are Python `str` indices (code points), not UTF-8 bytes or UTF-16 units.
- Return frozen dataclasses (`Token(start, end, kind)`, `Sentence(start, end, token_start,
  token_end, continues, continued)`); the serialized annotation `value` shape is defined by
  T-113, which must not change these rules.
- Apostrophe handling is resolved by the prepared spec, not an open maintainer decision.
  Keep its literal punctuation rule; implementation workers may not amend the spec.
- Size guard: one module and its tests; no command. File a follow-up rather than widen.
