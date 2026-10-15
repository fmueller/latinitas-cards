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
| `already_applied` | 0 | Operation was already applied |

## Contract command matrix

Only exits produced inside callbacks (with envelopes) belong in this matrix.
Click errors and unexpected internal failures are tested once on the contract
test app rather than repeated per command. No contract groups are registered yet;
their implementation tasks add rows and executable cases together.

| command | exit codes |
| --- | --- |

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
