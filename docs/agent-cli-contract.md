# Agent CLI contract

`latinitas-cards --version` eagerly prints `latinitas-cards <version>` from
installed package metadata and exits 0 without requiring a subcommand or input.
The existing root help description and legacy commands are unchanged.

The new `corpus`, `annotations` and `policy` commands use the version 1
`latinitas-cli-result` envelope. Text is the default. `--json` emits exactly one
canonical JSON document on stdout, followed by a newline. Diagnostics are carried
inside that document; text mode sends diagnostics to stderr with unsafe controls
escaped. An unexpected failure emits `internal`, exit 70, without a traceback.

The strict envelope contains `format`, `version`, `command`, `ok`, `exit_code`,
`diagnostics` and `result`. `ok` is true exactly when `exit_code` is zero.
Each diagnostic contains `code`, `severity` (`error` or `warning`), `message`,
`item`, `path` and `ref`. The last three are nullable. `item` is the patch
operation index, or for `corpus normalize` the zero-based passage ordinal N.
`path` is a file path and `ref` a fingerprint or ID. Extra keys are forbidden.
The generated schema lives at the repository-root-relative path
`docs/schemas/latinitas-cli-result.v1.schema.json`.

## Exit classes

| exit | meaning |
| --- | --- |
| 0 | Success, including warnings, no-op and pending review |
| 1 | Check or validation findings |
| 2 | Usage, prompt or timestamp needed |
| 3 | Input or output rejected |
| 4 | Stale or conflicting state |
| 5 | Policy or review gate refusal |
| 70 | Unexpected internal failure |

Computed error precedence is **3, 4, 5, then 1**, not numeric maximum.
Warnings do not affect the exit code. Explicit usage exits (2) and internal
failures (70) override computed classes. Click parse errors exit 2 before the
callback and produce no envelope.

## Closed diagnostic registry

Codes used at both severities retain their error exit class here; warnings
always compute to zero. New codes must be added to the enum, registry and table.

| code | exit | meaning |
| --- | --- | --- |
| `exists` | 3 | Output exists with different bytes |
| `binding_drift` | 4 | Bound state differs from the plan base |
| `busy` | 4 | Writer lock is held |
| `denied` | 5 | Operation refused by policy |
| `actor_refused` | 5 | Receipt actor is not permitted |
| `prompt_required` | 2 | Required prompt cannot be shown |
| `timestamp_required` | 2 | Explicit timestamp is required |
| `unsupported_version` | 3 | Document version is unsupported |
| `stale` | 1 | Passage hash differs from the anchor |
| `orphaned` | 1 | Anchor segmentation was superseded |
| `unmatched` | 1 | Anchored passage is missing |
| `internal` | 70 | Unexpected internal failure |
| `unreadable` | 3 | Filesystem input or output is inaccessible |
| `invalid_json` | 3 | Input is not valid JSON |
| `missing_header` | 3 | Stream header is missing |
| `unsupported_format` | 3 | Document format is unsupported |
| `alias` | 3 | Output aliases a protected path |
| `usage_error` | 2 | Callback arguments are incompatible |
| `unknown_element` | 3 | Source contains unknown elements |
| `input_too_large` | 3 | XML input exceeds 64 MiB |
| `depth_exceeded` | 3 | XML element depth exceeds 64 |
| `already_applied` | 0 | Operation was already applied |
| `invalid_corpus` | 3 | Consumer rejected an invalid corpus |
| `invalid_corpus_id` | 1 | Corpus ID is malformed |
| `duplicate_native_ref` | 1 | Native reference is repeated |
| `non_nfc_text` | 1 | Passage text is not NFC |
| `non_canonical_text` | 1 | Passage whitespace is not canonical |
| `hash_mismatch` | 1 | Passage hash differs from canonical text |
| `norm_mismatch` | 1 | Passage norm differs from latin-fold/1 |
| `passage_id_mismatch` | 1 | Passage ID does not match corpus and native reference |
| `invalid_citation` | 1 | Citation is invalid |
| `float_value` | 1 | JSON floats are forbidden |
| `invalid_record` | 1 | Corpus record schema or order is invalid |
| `invalid_manifest` | 3 | Corpus manifest is invalid |
| `invalid_sidecar` | 3 | Annotation sidecar is invalid |
| `segmentation_differs` | 0 | Current segmentation differs from recomputed value |
| `invalid_approval` | 3 | Approval is invalid or contradictory |
| `retired_ref` | 3 | Approval names a retired reference |
| `review_required` | 1 | Identity reconciliation requires review |
| `invalid_line` | 3 | Text line does not match citation template |
| `duplicate_source_ref` | 3 | Source label is repeated |
| `invalid_structure` | 3 | USFX structure is invalid |
| `empty_passage` | 3 | Source passage has no canonical text |
| `invalid_xml` | 3 | Source is not well-formed XML |
| `unsafe_xml` | 3 | Source XML contains a forbidden declaration |

## Contract command matrix

Only exits produced inside callbacks (with envelopes) belong in this matrix.
Click errors and unexpected internal failures are tested once on the contract
test app rather than repeated per command. Implementation tasks add rows and
executable cases together.

| command | exit codes |
| --- | --- |
| corpus validate | 0, 1, 3 |
| corpus normalize | 0, 1, 2, 3, 4 |
| annotations segment | 0, 2, 3, 4 |

`corpus validate CORPUS [--json]` streams the input without modifying it. Its
result is `{"counts":{"header":1,"corpus":1,"work":N,"passage":M}}`;
counts include recognized record types even when their contents have findings.
Text mode writes `<n> errors, <m> warnings` to stdout and one
`<severity> <code> <ref>: <message>` line per diagnostic to stderr (`-` for
an absent ref). JSON mode writes only the envelope to stdout.

The header/JSON gate reports `unreadable`, `invalid_json`, `missing_header`,
`unsupported_format` or `unsupported_version` (3). After a valid header,
schema/order/unknown-work errors are `invalid_record`; other findings are
`invalid_corpus_id`, `duplicate_native_ref`, `non_nfc_text`,
`non_canonical_text`, `hash_mismatch`, `norm_mismatch`,
`passage_id_mismatch`, `invalid_citation` and `float_value` (1).
Findings are retained in file order, including before a later JSON gate failure;
exit 3 takes precedence over 1. Invalid prerequisites do not generate cascading
derived-value findings. Diagnostics use the CORPUS argument as given for `path`,
`passage_id` (or `native_ref` when the ID is unusable) for `ref`, and null `item`.
Encoded references are compared verbatim, never decoded. Sidecar validation,
`corpus show` and `corpus config` are not implemented by this command yet.

`corpus normalize` accepts plain text or structural USFX and returns `{"review_items":[]}` on success.
Unresolved ordinal assignments return `review_required` (1) with items containing
`kind`, zero-based `ordinal` (null for removals), and `candidates` refs. Findings
are ordered by ordinal, then removed refs. No output bytes change until all are
approved. See [normalized corpora](normalized-corpus.md) for the flags and recovery
protocol. Template/manifest or template/approval combinations, malformed templates
or languages, and USFX combined with templates, manifests or approvals return `usage_error` (2).
Rejected input/approvals/aliases and expected filesystem failures return 3;
lock contention and recovery binding drift return 4. Existing corpus files are
atomically replaced, not treated as immutable artifacts (`exists` does not apply).

### Legacy exceptions (not envelope commands)

`claims list`, `claims record` and `claims inspect` keep their v0.2.1 JSON
and codes for deck and corpus inputs. Failures exit 1, except rejection of
corpus `--value`, `--evidence` or `--selected-sense` exits 3. Invalid batch
items still exit 1 and Click usage errors exit 2. All other legacy commands
retain their APIs, output and exit behavior; there is no envelope retrofit.

## Writers and schemas

New writers reject direct, symlink and hard-link aliases to protected paths.
Locks are exclusive, nonblocking `<path>.lock` files, never unlinked; multi-file
locks are acquired in lexical resolved-path order and held through replacement.
Stage and fsync every output before beginning a multi-file replacement, then
fsync each parent directory after replacing its file. Multi-file replacement
is recoverable, not atomically visible across paths. Expected filesystem errors
are `unreadable` (3), contention is `busy` (4), and a different existing artifact
is `exists` (3); identical bytes are a no-op. Legacy writers are unchanged.

Structured hashes use UTF-8 canonical JSON (sorted keys, Unicode preserved,
compact separators, no floats) and a `sha256:` prefix. Regenerate registered
schemas with `uv run python scripts/export-json-schemas.py`; tests only check
drift and never regenerate files.
