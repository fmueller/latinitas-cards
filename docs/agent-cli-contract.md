# Agent CLI contract

`latinitas-cards --version` eagerly prints `latinitas-cards <version>` from
installed package metadata and exits 0 without requiring a subcommand or input.
The existing root help description and legacy commands are unchanged.

The new `corpus`, `annotations`, `policy` and `corpus-cards` commands use the version 1
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
| `invalid_input` | 3 | Declared approval or claim decision input is invalid |
| `invalid_patch` | 3 | Patch document is invalid |
| `invalid_plan` | 3 | Plan identity or structure is invalid |
| `invalid_receipt` | 3 | Receipt structure or plan binding is invalid |
| `unknown_operation` | 3 | Selected operation is unknown |
| `invalid_operation` | 3 | Selected operation is invalid |
| `input_hash` | 1 | Input hash differs from the anchor |
| `invalid_value` | 1 | Proposed annotation value is invalid |
| `span_out_of_range` | 1 | Proposed span exceeds its binding |
| `unknown_segmentation` | 1 | Referenced segmentation is unknown |
| `unknown_target` | 1 | Operation target is unknown |
| `target_inactive` | 1 | Operation target is inactive |
| `duplicate_operation` | 1 | Patch contains a duplicate operation |
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
| `invalid_policy` | 3 | Policy document is invalid |
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
| `invalid_structure` | 3 | Source structure is invalid |
| `empty_passage` | 3 | Source passage has no canonical text |
| `invalid_xml` | 3 | Source is not well-formed XML |
| `unsafe_xml` | 3 | Source XML contains a forbidden declaration |
| `unknown_passage` | 3 | Requested passage is absent from the corpus |
| `invalid_profile` | 3 | Corpus profile is invalid |
| `profile_mismatch` | 3 | Corpora differ from the profile's corpus list |
| `invalid_reading` | 3 | Reading range has unknown or reversed bounds |
| `unknown_exercise` | 3 | Requested exercise is not available in the selection |
| `unsupported_language` | 3 | Source language is unsupported |
| `citation_mismatch` | 0 | Declared citation labels differ from the source structure |
| `skipped_division` | 0 | Non-edition source division was skipped |
| `skipped_empty_passage` | 0 | Source passage without canonical text was omitted |
| `descriptor_override` | 0 | Descriptor value replaced a source header value |
| `gap_in_text` | 0 | Source text has an editorial gap |
| `unmatched_translation` | 0 | Latin passage has no parallel translation |
| `unmatched_parallel` | 0 | Parallel passage has no Latin passage |

## Contract command matrix

Only exits produced inside callbacks (with envelopes) belong in this matrix.
Click errors and unexpected internal failures are tested once on the contract
test app rather than repeated per command. Implementation tasks add rows and
executable cases together.

| command | exit codes |
| --- | --- |
| corpus validate | 0, 1, 2, 3 |
| corpus normalize | 0, 1, 2, 3, 4 |
| annotations segment | 0, 2, 3, 4 |
| annotations inspect | 0, 1, 3 |
| annotations plan | 0, 1, 3, 4 |
| annotations approve | 0, 3, 4, 5 |
| annotations apply | 0, 2, 3, 4, 5 |
| annotations import-translations | 0, 2, 3, 4 |
| corpus show | 0, 2, 3 |
| corpus config | 0, 3 |
| policy validate | 0, 3 |
| corpus-cards profile validate | 0, 3 |
| corpus-cards select | 0, 1, 2, 3, 4 |

`annotations approve` requires explicit `--operation` IDs (no approve-all).
An auto ID is accepted as a no-op selection. Empty/blank `--review` or a malformed
declared `--actor` exits 3 `invalid_input`, without a receipt. Actors are not authenticated.

`annotations apply PLAN --corpus CORPUS --sidecar SIDECAR [--policy FILE]
[--receipt RECEIPT] [--timestamp T]` appends automatic and receipted review records
in plan order. Denied operations warn; unreceipted review IDs remain `pending`.
The result reports `appended`, `pending`, `denied` and `already_applied`.
A receipt naming another plan is rejected input (`invalid_receipt`, exit 3),
not binding drift. Input errors precede drift (4), then actor/policy refusal (5).
The corpus, sidecar and exact corpus claims locks are held from snapshot reads
through replacement; contention is `busy` (4), and lock files persist.
The sidecar must match its base or a line-boundary base plus this plan's records.
Only new records need a timestamp: patch `created`, then `--timestamp`, then
`SOURCE_DATE_EPOCH`; reruns neither append nor require a timestamp.
Apply never records claim decisions. Pending-operation route and complete safety
facts are recomputed under those locks using the entire embedded patch and current
version 2 decisions. Changed decision status, dependent IDs, conflict IDs or route,
or newly invalid anchors/targets, reject the entire batch with `binding_drift` (4)
before any append, even if both routes are `review`. Re-plan/re-approve; unrelated
decisions and expected deduplicated partial appends do not invalidate the plan.
Completed operations skip pending checks. Malformed/unreadable/v1 claims exit 3,
not empty decisions. Default human-only actor enforcement also applies to hand-built
receipts (`actor_refused`, 5); `receipt_actor: any` permits agent declarations.

| apply safety case | exit / write effect |
| --- | --- |
| changed or removed decision/dependent/conflict; new invalid pending operation | 4 `binding_drift`, no append |
| expected partial append, later receipt, completed rerun, unrelated decision | 0, append only new eligible records |
| claims unreadable / malformed / v1 | 3 `unreadable` / `invalid_sidecar` / `unsupported_version`, no append |
| claims writer lock held | 4 `busy`, no append |
| invalid plan operation, any producer/actor kind | 3 `invalid_operation`, no append |
| agent receipt under human-only policy | 5 `actor_refused`, no append |
| deny operation without receipt selection | 0 with `denied` warning, denied records never appended |

`annotations import-translations CORPUS --parallel PARALLEL --language TAG
--producer-kind human|agent|llm --producer-name N --producer-version V --output
PATCH [--timestamp T]` writes only a `latinitas-annotation-patch` and reports
`{"operations","patch","written"}`; `written` is false for an identical rerun.
Unmatched refs warn on both sides; usage, language, artifact and lock exits are
described in [annotation sidecars](annotation-sidecars.md#import-aligned-parallel-translations).

The routing matrix lives in `tests/unit/annotation_routing_test.py`; lock contention
and synchronized competing claims writers also run in `annotation_apply_test.py`.

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
Encoded references are compared verbatim, never decoded.

With `--annotations SIDECAR`, validation also reports anchor findings in sidecar
order: `stale`, `orphaned`, `unmatched` (1). `ref` is the annotation ID, `path`
is SIDECAR, and the message names its passage. Unsupported sidecar formats or
versions and invalid sidecar records exit 3; no lock is taken or created.

`annotations inspect SIDECAR --corpus CORPUS [--claims SIDECAR] [--check] [--json]` lists provenance
and lifecycle flags in sidecar order. Anchor findings are warnings (0) unless
`--check` promotes them to errors (1). Both modes report identical states.

`corpus show CORPUS [--passage ID]... [--limit N] [--annotations SIDECAR] [--claims SIDECAR] [--json]`
returns `{"passages":[{"passage_id","citation","text","annotations"}],"truncated":bool}`.
Each annotation has `annotation_id`, `layer`, `value`, `producer`; only current,
non-superseded, non-retracted proposals are visible. Retraction events are not
proposals. IDs select in corpus order, with duplicates collapsed. The default
limit is 20, maximum 1000; 0 and 1001 are Click usage errors (2, no envelope).
`truncated` means more matching passages exist, not merely more corpus passages.
Unknown requested IDs exit 3 (`unknown_passage`), even beyond the selected page.
All three readers leave bytes and mtimes unchanged and create no locks.
All three accept an explicit `--claims SIDECAR` version 2 decision file.
`corpus validate CORPUS --annotations SIDECAR --claims SIDECAR` reports stale
decisions as errors (1), with code `stale`, `ref` = claim fingerprint and `path`
= claims file. `annotations inspect` reports decision findings as warnings unless
`--check` is used. Show omits stale annotations, not current withheld proposals.
Duplicate live decisions, whether identical or opposite, are ignored and reported
as `invalid_input` with the claims path and fingerprint. `corpus show` and inspect
without `--check` warn (exit 0); validate and inspect `--check` error (exit 3).
Show continues to omit stale-decision diagnostics. Mixed findings retain the
standard exit precedence (3 before 4, 5, 1).
Show and inspect add `review_status`: `open`, `accepted` or `withheld`; no matching
live decision means `open`. Neither grants acceptance nor changes decisions.
For corpus show/validate, `--claims` without `--annotations` exits 2 `usage_error`
with an envelope. Inspect takes the annotation sidecar positionally.
An explicit missing/unreadable claims file exits 3 `unreadable`; v1 or higher
versions exit 3 `unsupported_version`, each with one envelope under `--json`.

`corpus config [--policy FILE] [--corpus FILE] [--json]` reports `tool_version`,
`supported_formats` (format name to maximum version), `rule_versions`, resolved
`policy_path`/`corpus_path`, `policy_digest`, `corpus_digest`,
`corpus_header_digest`, and `receipt_actor`. Every entry is `{value, origin}`:
`default` for built-in or absent values, `flag` for paths supplied on the command
line, and `file` for values/digests read from those files. There is no project
configuration file. Omitted digests/paths are null; omitted `receipt_actor` is
`human`, even when a policy file is supplied. File digests hash exact bytes;
the header digest hashes the canonical validated corpus header, matching plans.
Invalid corpus findings produce `invalid_corpus` naming the first finding,
followed by the findings; loader diagnostics retain their codes (exit 3).
The report is read-only: no writes, locks or mtime changes.

Prompts require both stdin and stdout to be TTYs. `setup` needing field choices
without a terminal exits 2 with `prompt_required` naming
`--lexical-entry-field`, `--principal-parts-field`, and `--confirm`; otherwise a
needed confirmation names `--confirm` and `--non-interactive`. Supplying all
needed field flags and `--confirm` saves without prompting. Explicit
`--non-interactive` preserves legacy setup outcomes, including cancellation
without `--confirm`. Setup keeps its legacy output, not a result envelope.
`cloze --parallel-mode prompt` without a terminal warns on stderr and ignores
translations. Contract commands (`corpus`, `annotations`, `policy`, `corpus-cards`) never prompt.

`corpus-cards profile validate PROFILE [--json]` reads only the
`latinitas-corpus-profile` version 1 file (see
[the example](examples/corpus-profile.json)) and returns
`{"profile_id","profile_digest","profile"}`. `profile` is the effective profile
with defaults applied, and `profile_digest` is its structured hash, so key order,
whitespace and explicitly written defaults do not change it. The profile is strict:
unknown keys, floats, a `target_language` whose primary subtag is not `de` or
`en`, unknown or repeated recipes, out-of-range budgets, limits or weights,
unsorted or repeated `corpora`, unsorted or repeated `targets` (compared after
`latin-fold/1`), blank names, names with surrounding whitespace or line breaks, and a reading
`corpus_id` not in `corpora` exit 3 (`invalid_profile`). Validation does not
read corpora: reading works and refs are checked by the later `corpus-cards`
commands that use them, and those commands exit 3 (`profile_mismatch`) when given
corpora that differ from the profile's `corpora` list.

`corpus normalize` accepts plain text, structural USFX or structural TEI and returns `{"review_items":[]}` on success.
Unresolved ordinal assignments return `review_required` (1) with items containing
`kind`, zero-based `ordinal` (null for removals), and `candidates` refs. Findings
are ordered by ordinal, then removed refs. No output bytes change until all are
approved. See [normalized corpora](normalized-corpus.md) for the flags and recovery
protocol. Template/manifest or template/approval combinations, malformed templates
or languages, USFX or TEI combined with templates, manifests or approvals, `--allow-unknown`
with plain text, and `--work` that is malformed or used without TEI return `usage_error` (2).
Rejected input/approvals/aliases and expected filesystem failures return 3;
lock contention and recovery binding drift return 4. Existing corpus files are
atomically replaced, not treated as immutable artifacts (`exists` does not apply).

### Ranked corpus selection

`corpus-cards select PROFILE --corpus FILE... --selection FILE --report FILE`
fills free recipe and work budgets with `latin-rank/1`; `per_run` counts only newly
selected exercises. Existing selections, including ineligible ones, always stay
and occupy recipe/work slots. The report lists all candidates with eligibility,
reasons, factors, weighted contributions, scores and outcomes. Eligible candidates
sort by descending score, corpus ID, passage order, sentence index (passage last),
target word position (no target last), then LatinitasID. Ineligible identities sort
last by LatinitasID, followed by invalid spans by annotation ID.

Factors use accepted morphology, not surface-text guesses or annotation confidence.
Known lemmas require both `--known-lemmas PROFILE --known-lemmas-source SOURCE`;
they are read without writes. Without them every word is unknown and every lexical
target is new. Single-word targets do not count toward unknown context words.
Matching known/target lemmas strips a trailing ` (n)` homograph suffix; per-lemma
exclusion keeps that suffix and folds only case/macrons, per `latin-fold/1`.
Near duplicates use folded word-token **multisets** of units (phrase spans for
phrase translation), with intersection/union at least 4/5.

The closed outcome/withholding vocabulary in the selection/candidate schemas is:
`kept`, `selected`; filters in precedence order `filtered:fragment`, `filtered:length`,
`filtered:unknown`, `filtered:reading`; exclusions in precedence order `deselected`,
`budget:recipe`, `budget:work`, `budget:run`, `near_duplicate`, `per_lemma`;
fact reasons `missing_fact`, `stale_fact`, `open_fact`, `withheld_fact`,
`conflicting_facts`, `unsupported_morphology`, `wrong_language`, `no_features`,
`invalid_span`; selection reasons `target_missing`, `context_changed`.
These are report values, not diagnostics. Deselection's human reason is free text.

`--confirm-context ID` (repeatable) records a reviewed selected exercise's current
digest; a missing/nonselected ID exits 3 (`unknown_exercise`). A changed context
stays ineligible until confirmed. Confirmation never transfers identity to a new key.
Selection binding compares the exact profile ID and target-language tag, not just
its primary subtag (`binding_drift`, 4); exercise identity still uses only the
lowercased primary subtag. Reading bounds are validated at use, inclusive by passage
order within the specified work (`invalid_reading`, 3). Profile validation alone
does not check those references. Stale claim decisions are reported with exit 1;
the candidate report still records the safe current eligibility and selection.

Selection inputs are captured in private temporary files before fact/known-lemma
consumption. Report digests identify those captured bytes, not later edits to live
inputs. This is snapshot provenance, not a promise that inputs remain unchanged
until output commit. Temporary paths never enter the selection or report.
Present selection documents require every envelope/state field and reject duplicate
JSON keys (`invalid_input`, 3), rather than treating ambiguous state as empty.

Both outputs are locked and staged with the writer rules below; selection replaces
last. The report is always replaced, but unchanged selection bytes are not replaced
and do not increment `runs`. Initial creation, even empty, counts as run 1.
Report bytes are never read as state. A report left by
a failed selection replacement may describe an uncommitted run: rerun `select` to
recompute it from selection state. Deselect/restore and preview/export/inspect/plan
commands are separate work; this command does not render or export a deck.

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
schemas with `uv run python scripts/export-json-schemas.py`; tests check drift
and exercise the exporter in a temporary directory, never rewriting committed files.
