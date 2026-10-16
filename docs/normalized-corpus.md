# Normalized corpora

`corpus normalize` converts one source to canonical JSONL without changing
deck exports. For plain UTF-8 text, each non-empty canonical
text line becomes a passage (NFC, Unicode whitespace collapsed, trimmed); one
leading BOM is stripped and LF/CRLF inputs are accepted.

```sh
uv run latinitas-cards corpus normalize source.txt --adapter text \
  --corpus-id latin-reader --output reader.jsonl --json
uv run latinitas-cards corpus validate reader.jsonl --json
```

## Records and identity

The first record is a `latinitas-corpus` v1 `header`, followed by one `corpus`,
ordered `work` records, and ordered `passage` records. The header records the
corpus ID, language (`la` by default), adapter/rule versions, and raw source
basename, SHA-256 and byte count. No absolute paths, clocks or hostnames enter
the corpus or identity digests. Optional `--descriptor FILE` accepts the strict
`latinitas-corpus-descriptor` v1 model: edition, publisher, url and license go
in the header; title, period and register go in the corpus. They are null when
absent. `--language TAG` accepts a two/three-letter language followed by optional
hyphen-separated 2–8-character alphanumeric subtags.

Passages carry `passage_id` (`corpus_id:native_ref`), work, 1-based order,
citation (`scheme`, raw string `parts`, `display`), canonical text, text-only
`passage_hash`, and matching-only `norm` (`latin-fold/1`). Optional translations,
notes, headings, source markup and source extras are null for this adapter.
Committed JSON Schemas are in [schemas](schemas/).

## Ordinal manifest and review

Without a citation template, refs are `p000001`, `p000002`, … and citations use
scheme `ordinal`, with a single `passage` part. The first run allocates every
line, including duplicates, and writes `reader.manifest.json` (override with
`--manifest FILE`). The strict `latinitas-corpus-manifest` v1 stores active and
retired entries, their passage hashes and zero-based ordinals, the ordered hash
snapshot, next never-used ref, and the materialization receipt. Never edit it
manually or remove it to bypass review.

An unchanged ordered hash snapshot reuses refs positionally, including repeated
lines. Otherwise only unique exact hashes reuse refs automatically. Edited,
inserted, duplicate and removed passages require review. Exit 1 `review_required`
writes nothing; JSON `result.review_items` lists `{kind, ordinal, candidates}`,
with removed items last (null ordinal). Diagnostic `item` is the zero-based
ordinal and `ref` identifies a candidate when available.

```sh
uv run latinitas-cards corpus normalize source.txt --adapter text \
  --corpus-id latin-reader --output reader.jsonl \
  --approve-reuse 1=p000002 --approve-allocation 3 --approve-removal p000004
```

Use `--approve-reuse N=REF` to retain a reviewed identity, `--approve-allocation N`
for the next never-used ref, and `--approve-removal REF` to retire an unused ref.
Repeat flags for multiple items. Retired refs never return to allocation.
Unknown refs, out-of-range ordinals, duplicate/contradictory approvals and flags
on first creation fail with exit 3. Reuse of a retired ref is `retired_ref` (3).

## Recoverable writes

Writers hold both resolved-path `.lock` files in lexical order, from input and
old-state reads through replacements. Busy locks fail nonblocking with exit 4;
lock files are never removed. Outputs may not alias each other, source,
descriptor or lock paths, including symlink and hard-link aliases.

Both validated outputs are staged and fsynced before replacing the manifest,
then the corpus; each replacement fsyncs its parent. This is a recoverable
two-file commit, **not** atomic visibility across paths. Readers need only the
complete old/new corpus, never the manifest. Identical bytes retain their inode.

The manifest receipt retains the canonical request digest, target corpus byte
digest, and previous corpus byte digest (null on first creation). If interrupted
after manifest commit, **rerun the identical request, including approvals**.
The retry rebuilds the committed assignments without allocating refs or consuming
approvals again, verifies the target digest, and replaces only the corpus.
Reordered equivalent flags or relocated paths preserve request identity; changed
raw source, basename, corpus ID, language, descriptor content or approvals do not.
Foreign requests/bytes or an irreproducible target fail unchanged with exit 4
`binding_drift`. Expected write errors are exit 3 `unreadable`. Temporary files
are not committed state; handled failures clean them. Once the target corpus is
present, the identical request is a no-op and a new request may reconcile normally.

## Labeled citation templates

```sh
uv run latinitas-cards corpus normalize labeled.txt --adapter text \
  --corpus-id latin-reader --output labeled.jsonl \
  --citation-template wsl:work.section.line
```

Input lines are `LABEL<TAB>text`, for example `Aeneis.1.1<TAB>Arma virumque cano.`
Split labels on `.` with one non-empty component per unit. Raw NFC citation
values are preserved; refs encode each component's UTF-8 bytes once (literal `%`
becomes `%25`, `%2E` becomes `%252E`, `é` becomes `%C3%A9`). A `work` unit creates
works in first-appearance order; otherwise the only work is the corpus ID.
Duplicate labels and malformed lines fail with exit 3. Templates never read or
write a manifest, and combining them with `--manifest` or approval flags is a
usage error (2), as is a malformed template. `--json` emits the same versioned
result envelope as validation; default text output summarizes diagnostics.

## Structural USFX

```sh
uv run latinitas-cards corpus normalize tests/fixtures/corpus/usfx/gen-1-1-5.usfx.xml \
  --adapter usfx --corpus-id vulgata-clementina --output genesis.jsonl \
  --descriptor data/lat-clementine.descriptor.json --json
```

The supported subset is `usfx`, `book@id` (container or empty milestone), `h`
(work title), `c@id|n`, `v@id|n` (milestone or container), `ve`, transparent `p`/`q`,
and `wj`/`w` inline markup. Root attributes
are ignored; `id` wins over `n`. Milestone verses must end with `ve`; container
verses may have a trailing `ve`. Works retain first-appearance order and passages
retain document order, not book-ID sort. Character data joins without inserted
separators before canonicalization. Raw NFC book/chapter/verse citation parts
produce `GEN 1:3a`; their individually encoded components form `GEN.1.3a`.
Literal `.` and `%2E` encode distinctly as `%2E` and `%252E`.

The `usfx/1` header hashes and counts raw source bytes; the source is never
rewritten. No descriptor is selected implicitly, even for the bundled source.
With no `--descriptor`, provenance fields are null. See the
[provenance audit](license-compatibility-audit.md#bundled-corpus-provenance)
for the declared Clementine provenance and pending maintainer confirmation.

This adapter writes only the corpus, using the shared locked atomic writer and
alias checks. `--manifest`, approvals and citation templates are usage errors
(2). Duplicate references, empty passages, invalid structure, malformed XML and
unsafe XML fail with exit 3 and leave the output unchanged. Unknown elements
produce one diagnostic with every name and count, also with exit 3; explicit
`--allow-unknown` retains their text (and therefore its hash), attributes and spans
in `source_extra`, with the same diagnostic as a warning on stderr in text mode.
`source_markup` retains `wj`/`w` attributes and code-point spans into canonical
text. Spans exclude surrounding whitespace; a boundary inside an NFC-composed
code point includes that indivisible code point.

Notes (`f`, `x`, `fe`) and headings (`s`) flatten their descendants in document
order, canonicalize their text, and do not enter passage text or hashes. Notes
inside a verse attach there; headings and notes outside a verse attach to the next
passage in the same work, or fail `invalid_structure` if none exists.
The strict expat reader rejects DTD/entity declarations and external references,
disables parameter-entity parsing, checks 64 MiB at stat and while streaming, and
rejects element depth above 64, including note descendants. No XML dependency is added.
Legacy Vulgate commands retain their existing parsing and outputs; their hardening
is pending and they are not yet resource-bounded XML ingestors.

The GEN 1:1–5 fixture and expected JSONL are committed. Regenerate deliberately
with `uv run python scripts/regen-corpus-fixtures.py --target usfx --force`;
tests never regenerate snapshots. The fixture is the bundled source's exact
prefix through GEN 1:5's `ve`, followed by `</book></usfx>` and a newline.
Synthetic markup snapshots and rejection expectations are regenerated separately
with `uv run python scripts/regen-corpus-fixtures.py --target markup --force`.
