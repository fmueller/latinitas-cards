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

Descriptors also accept optional `revision`, passed unchanged to the header for
all adapters. It records the supplied upstream revision, not proof that the
download matches it; the raw-input SHA256 independently identifies the bytes.
The new revision field is omitted when absent, preserving existing serialization
and header digests. Older strict readers reject enriched records/descriptors.

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
for the declared Clementine provenance, dated maintainer confirmation and
remaining verification limits.

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
The hardened expat reader shared by the XML adapters rejects DTD/entity declarations
and external references, disables parameter-entity parsing, checks 64 MiB at stat and
while streaming, and rejects element depth above 64, including note descendants. No
XML dependency is added.
For accepted inputs, legacy Vulgate commands (`preview --usfx`, `generate --usfx`,
and `cloze` with USFX) retain their existing projection and outputs. They reject
DTD/entity declarations and external entities, inputs over 64 MiB, and element depth over
64; these limits do not bound total memory.

The GEN 1:1–5 fixture and expected JSONL are committed. Regenerate deliberately
with `uv run python scripts/regen-corpus-fixtures.py --target usfx --force`;
tests never regenerate snapshots. The fixture is the bundled source's exact
prefix through GEN 1:5's `ve`, followed by `</book></usfx>` and a newline.
Synthetic markup snapshots and rejection expectations are regenerated separately
with `uv run python scripts/regen-corpus-fixtures.py --target markup --force`.

## Structural TEI

```sh
uv run latinitas-cards corpus normalize tests/fixtures/corpus/tei/prose.xml \
  --adapter tei --corpus-id caesar-bg --output caesar.jsonl --json
```

`--adapter tei` (`tei/1`, profile `epidoc-cts/1`) imports one TEI P5 file in the
`http://www.tei-c.org/ns/1.0` namespace in the Perseus/EpiDoc CTS style as one work.
It uses the shared hardened reader with namespace processing; outside `teiHeader`
and skipped divisions, unsupported elements, including any in another namespace,
are unknown, and processing instructions are ignored. The root `TEI`
holds `teiHeader` and `text/body`, and the body holds exactly one
`div[@type="edition"]` with `xml:lang` `lat` or `la`. A missing edition fails
`invalid_structure` and another language `unsupported_language` (both exit 3).
Other body divisions, such as a translation, are skipped with one
`skipped_division` warning naming their types.

A passage is an `l[@n]` outside `note`, `rdg`, `sic`, `abbr`, `orig` and `del`, or a
deepest `div[@type="textpart"]` with no such line. `lg` groups lines transparently;
an `l` without `@n`, or inside those elements, is flattened text. An `l[@n]` inside
`head`, or with an empty `@n`, fails `invalid_structure`. In a verse textpart a
`note`, `del`, `rdg`, `lb`, `pb`, `cb`, `milestone` or `gap` between lines, including
inside an unnumbered `l` or between `lg` groups, attaches to the preceding kept line
of the same textpart, else to the next passage; `head`, and these items (plus `bibl`
and `desc`) outside verse textparts, attach to the next passage, else to the last
one. Other text outside lines fails `invalid_structure`. A passage whose canonical
text is empty, such as a line consisting only of `del`, is omitted with a
`skipped_empty_passage` warning naming its ref; its own notes, headings and empty
known-element spans attach as if they stood between passages, and its other
known-element spans are dropped. Unknown elements in an omitted passage instead
fail `invalid_structure` even with `--allow-unknown`, since no retained canonical
passage-text position exists for them.
An edition whose passages are all omitted fails `invalid_structure`.

Text and markup:

- `note`, `bibl` and `desc` become `note`-kind notes and `head` a heading, each with
  its flattened text. `del` and each `rdg` become `variant`-kind notes.
- `app` keeps the text of its `lem`. Inside `choice`, the `corr`, `expan` or `reg`
  text is kept and `sic`, `abbr` or `orig` become `variant`-kind notes; with no kept
  child nothing stays in `text`. `app` and `choice` record no span and allow only
  these children, at most one of them kept, and whitespace. `lem` outside `app`
  fails `invalid_structure`.
- Flattened text has no position in canonical text, so `app`, `lem`, `rdg`, `choice`,
  `gap` or an unknown element inside a note, heading or variant fails
  `invalid_structure`.
- Outside `choice`, `reg`, `corr`, `expan`, `sic`, `abbr` and `orig` keep their text
  and record spans, as do `p`, `ab`, `add`, `supplied`, `unclear`, `hi`, `emph`, `ex`,
  `q`, `quote`, `foreign`, `name`, `persName`, `placeName`, `geogName`, `num`,
  `date`, `w`, `seg`, `label`, `title` and `cit`. Each `source_markup` entry holds the
  element's attributes and its span in canonical text.
- `lb`, `pb`, `cb`, `milestone` and `gap` must be empty and record empty spans; a
  kept passage holding gaps warns `gap_in_text` once with its ref and gap count.
- Any other element, including drama (`sp`, `speaker`, `stage`), fails
  `unknown_element` (exit 3) with names and counts. With `--allow-unknown` an unknown
  element in passage text keeps its text and records its attributes and span in
  `source_extra`, and the same diagnostic is a warning. One outside passage text
  (between textparts or lines, inside flattened text, `app` or `choice`) still fails
  `invalid_structure`, as does one in an omitted passage. An unknown element with
  an empty span in a retained passage is supported; it is not moved to another
  passage or assigned an invented attachment span.

Citations use scheme `tei-cts`: one part per textpart `@subtype` (plus `line` for a
line) with the raw NFC `@n` values, displayed joined with `.`; `native_ref` joins
the individually encoded components (components `1`, `2` and `1.5 é` give
`1.2.1%2E5%20%C3%A9`).
Every passage must share the same unit path and every ancestor textpart needs
`@subtype` and `@n`, otherwise `invalid_structure`; a repeated ref fails
`duplicate_source_ref`. `refsDecl[@n="CTS"]/cRefPattern/@n` labels that, read
deepest first, differ case-insensitively from the unit path only warn
`citation_mismatch`.

The work ID is `--work ID` (`^[A-Za-z0-9._-]{1,64}$`, TEI only), else the textgroup
and work of a CTS URN in the edition `@n` when they match that pattern, else the
first two components of `publicationStmt/idno[@type="filename"]` when they match
`^[a-z]+[0-9]+\.[a-z]+[0-9]+$`, else the corpus ID. The work and corpus titles are
the first `titleStmt/title`. The header supplies `edition`
(`editionStmt/edition`, else the first `sourceDesc//title`), `publisher`,
`url` (`idno[@type="URI"]`) and `license` (`licence/@target`, else its text), each
null when absent. A `--descriptor` value overrides these fields and the corpus
title (the work title keeps the header value) and warns `descriptor_override` for
each header value it replaces with a different one.

TEI work records additionally retain the sole `titleStmt/author` and an actual
edition-level CTS URN from edition `@n` and/or `body/@xml:base`. Nonempty URN
candidates must all match `urn:cts:namespace:textgroup.work.edition` (ASCII
letters, digits, underscore and hyphen in each component) and agree. Malformed
or conflicting values warn `citation_mismatch` and suppress the edition URN;
work-only and passage URNs are not edition identities. Multiple authors warn
and omit the single-author field. Neither metadata field is inferred from the
filename or corpus ID, and the existing work-ID fallback is unchanged.

Absent new fields are omitted. Existing documents round-trip unchanged; TEI
renormalization adds metadata where available, and older strict readers reject
these enriched records even though corpus version remains 1. Header changes
(including supplied revision) invalidate pending header-bound annotation plans:
replan and obtain any required approval again. Work-only metadata changes do not.
Unchanged passage/annotation content keeps its claims, exercise IDs, context
digests and selections. No annotation or selection migration is performed.

Like USFX, this adapter writes only the corpus and rejects `--manifest`, approval
flags and `--citation-template` (exit 2). Regenerate the
synthetic goldens deliberately with
`uv run python scripts/regen-corpus-fixtures.py --target tei --force`.

### Recognized TEI elements

This is the complete case-sensitive element vocabulary outside the header and
skipped divisions. Recognition does not make an element valid in every position;
the placement rules above still apply.

| Elements | Treatment |
| --- | --- |
| `TEI`, `teiHeader`, `text`, `body`, `div` | Document, header and edition/textpart structure. |
| `l`, `lg` | Numbered line passages and transparent line groups. |
| `head` | Heading, outside canonical passage text. |
| `note`, `bibl`, `desc` | Flattened notes, outside canonical passage text. |
| `app`, `lem`, `rdg` | Apparatus: keep lemma text, retain readings as variants. |
| `choice`, `corr`, `expan`, `reg`, `sic`, `abbr`, `orig` | Editorial alternatives; outside choice, retain text and spans. |
| `del` | Deleted text retained as a variant, not canonical text. |
| `p`, `ab`, `add`, `supplied`, `unclear`, `hi`, `emph`, `ex` | Retain text and markup spans. |
| `q`, `quote`, `foreign`, `name`, `persName`, `placeName`, `geogName` | Retain text and markup spans. |
| `num`, `date`, `w`, `seg`, `label`, `title`, `cit` | Retain text and markup spans. |
| `lb`, `pb`, `cb`, `milestone`, `gap` | Empty spans; gaps also warn. |

Header descendants are read as metadata, not passage markup. The consumed paths
are `titleStmt/title`, `editionStmt/edition`, `sourceDesc//title`,
`publicationStmt/publisher`, `publicationStmt/idno`,
`publicationStmt/availability/licence` and
`encodingDesc/refsDecl[@n="CTS"]/cRefPattern`. Other header content is not
guaranteed to be retained.

### TEI scope and compatibility

No network fetching, drama conversion, multi-file editions, translation-division
import or arbitrary TEI customization is supported. Normalize a parallel
translation separately and use `annotations import-translations`; skipped TEI
translation divisions do not become annotations. `--allow-unknown` is not a
general preservation mode: unknown elements need a position in retained canonical text.
The out-of-text retention question remains a blocked maintainer decision; do not
move or flatten scholarly content to evade a rejection without reviewing that
source transformation.

CTS pattern labels are compared deepest-first and case-insensitively. A reversed
human-readable example is not evidence of a warning; the actual unit path decides.
The example-versus-rule question remains with the maintainer.
Older v0.3.0 readers reject `tei/1` corpora even though the corpus wire version
remains 1. USFX and plain-text outputs keep their existing formats.
