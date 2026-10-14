---
id: T-103-define-the-cli-result-envelope-exit-codes
title: Define the CLI result envelope, exit codes, diagnostic registry and --version
status: todo
priority: high
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies:
    - T-102-capture-characterization-goldens-for-legacy
updated_at: "2026-10-14T05:42:59Z"
---

# T-103-define-the-cli-result-envelope-exit-codes Define the CLI result envelope, exit codes, diagnostic registry and --version

## Description

Deliver the shared foundation of `specs/v0.3.0.md#agent-ready-cli-contract`: the
`latinitas-cli-result` version 1 envelope, the exit-code classes and precedence 3, 4, 5,
then 1 (**Exit codes**), the closed diagnostic registry, the per-command exit matrix in
`docs/agent-cli-contract.md`, shared writer primitives (**Files**, sidecar writer rules)
and `latinitas-cards --version` (**Commands**). Today `cli.py` builds the root `app` with
no callback, `claims` prints its own `{"claims"|"recorded", "diagnostics"}` JSON and every
writer hand-rolls temp-file replacement (`claim_sidecar.record_batch`,
`phrase_inputs.save_selections`). `claims` output and exit behaviour stay unchanged; it is
not a contract command. Not in this task: any `corpus`, `annotations` or `policy` command
(T-105, T-113 and T-116 register into this foundation and add their own matrix rows and
codes), `corpus config` and the no-prompt guarantee (T-120).

## Acceptance

- `src/latinitas_cards/cli_contract.py` defines a strict Pydantic `CliResult` (`format:
  "latinitas-cli-result"`, `version: 1`, `command`, `ok`, `exit_code`, `diagnostics:
  [{code, severity: "error"|"warning", message, item, path, ref}]`, `result`; `item:
  int|null`, `path`/`ref: str|null`; extra keys forbidden; `ok` true iff `exit_code ==
  0`). Tests reject `"version": 1.0`, severity `"info"`, an extra key and an unregistered
  code; `CliResult` is registered in `json_schemas.SCHEMAS` and
  `docs/schemas/latinitas-cli-result.v1.schema.json` passes the shared drift test.
- `exit_code_for(diagnostics)` returns 0 for warnings only, otherwise the
  highest-precedence error class in the order 3, 4, 5, 1; parametrized cases: {1,5}->5,
  {5,4}->4, {4,3}->3, {1}->1, {warning 5}->0. Exit 2 (`usage_error`, `prompt_required`,
  `timestamp_required`) and 70 (`internal`) are raised explicitly and override the
  computed class.
- Codes are members of `class Code(StrEnum)` in `cli_contract.py`, and `DIAGNOSTIC_CODES`
  maps each code to its exit class and meaning, seeded with every code Fixed Contracts
  names (`exists`, `binding_drift`, `busy`, `denied`, `actor_refused`, `prompt_required`,
  `timestamp_required`, `unsupported_version`, `stale`, `orphaned`, `unmatched`,
  `internal`) plus `unreadable`, `invalid_json`, `missing_header`, `unsupported_format`,
  `alias` and `usage_error`; later tasks add their own rows and reuse a seeded code for the
  same condition (a test asserts no two registry entries share a meaning string). Each code
  has exactly one exit class; warning-only codes have class `0`. Codes used with both
  severities (`unknown_element`, `denied`, `stale`, `orphaned`, `unmatched`) take their
  class from the error severity. `Diagnostic.code` and every contract exception take
  `Code`, so mypy proves every raised code is registered. A test parses the `| code | exit
  | meaning |` table in `docs/agent-cli-contract.md` and asserts `set(Code) ==
  set(DIAGNOSTIC_CODES) ==` the docs table; a static scan over `src/latinitas_cards/**`
  rejects any `Diagnostic(`/`ContractError(` call whose code argument is a `str` literal
  (negative control: a synthetic source using `"busy"`).
- A `run_contract(command, json_output, body)` helper (used by every later contract
  command): in `--json` mode stdout holds exactly one canonical `CliResult` and nothing
  else; text mode prints the result to stdout and diagnostics to stderr with controls
  escaped by `profile_setup.encode_unsafe_controls`; an unexpected exception yields one
  `internal` envelope, exit 70, no traceback on stdout. A test-only Typer app in the test
  module triggers exits 0, 1, 2, 3, 4, 5 and 70: each `--json` stdout parses with
  `CliResult.model_validate_json` and its `exit_code` equals the process exit; a Click
  parse error (missing required option) exits 2 with empty stdout and no envelope.
- `src/latinitas_cards/atomic_io.py`: `reject_aliases(output, protected)` (direct path,
  symlink, hard link; generalizes `save_selections`), `atomic_replace(path, data)`
  (same-dir temp, `fsync`, `os.replace`, temp removed on failure), `writer_lock(path)`
  taking `<path>.lock` with `fcntl.flock(LOCK_EX | LOCK_NB)` and raising a `busy` contract
  error, and `write_artifact(path, data, protected)` (identical bytes: no-op; different
  bytes: `exists`, exit 3; it also takes `writer_lock(path)`). Tests: a lock held by the
  test process gives `busy` with bytes unchanged; a monkeypatched failing `os.replace`
  keeps the old file and leaves no temp file; each alias form is rejected with nothing
  written.
- Provide reusable multi-file locking in lexical resolved-path order, exclusive and
  nonblocking, holding every acquired lock until the protected operation ends. Never
  unlink lock files; release acquired locks on contention or failure. Tests acquire the
  same paths in reversed caller order and contend on a later lock without leaking an
  earlier lock. T-107 and T-118 use this primitive, not nested independent writer calls.
  Stage/fsync and replacement primitives must permit T-107 to stage both outputs before
  replacing either and fsync each parent directory after replacement. Expected filesystem
  failures map to exit 3 `unreadable`, not `internal` (70).
- `src/latinitas_cards/canonical.py`: `canonical_json(value) -> str` (`sort_keys=True,
  ensure_ascii=False, separators=(",", ":"), allow_nan=False`, raising `ValueError` on any
  `float`), `digest(value) -> "sha256:<hex>"` over its UTF-8 bytes and `digest_bytes(data)
  -> "sha256:<hex>"`. `src/latinitas_cards/json_schemas.py` holds the registry `SCHEMAS:
  dict[str, type[BaseModel]]` (format name -> model) and `render_schema(model) -> str`
  (`json.dumps(model.model_json_schema(), indent=2, sort_keys=True, ensure_ascii=False) +
  "\n"`); `scripts/export-json-schemas.py` writes
  `docs/schemas/<format>.v<version>.schema.json` for every entry (never called by tests);
  `tests/unit/json_schema_drift_test.py` is parametrized over `SCHEMAS` and also fails on an
  unregistered file in `docs/schemas/`. Later tasks only add registry entries.
- `latinitas-cards --version` prints `latinitas-cards <version>` and exits 0; the test
  compares with `importlib.metadata.version("latinitas-cards")` and the `pyproject.toml`
  version read with `tomllib`; `--help` still lists every existing command and the T-102
  characterization tests stay green.
- `docs/agent-cli-contract.md` documents the envelope, exit table, precedence, registry
  table and a `| command | exit codes |` matrix. A test asserts that every registered
  command under the `corpus`, `annotations` and `policy` groups has a matrix row and every
  documented (command, exit) pair has an entry in `CONTRACT_EXIT_CASES` in
  `tests/unit/cli_contract_matrix_test.py`, which runs each case through CliRunner. Matrix
  rows list only exits produced inside the callback (with an envelope); Click usage errors
  (exit 2, no envelope) and `internal` (70) are tested once on the test-only app and are not
  repeated per command. The docs state that a diagnostic's `item` is the patch operation
  index, or for `corpus normalize` the zero-based passage ordinal N.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Capture the registry/docs equality test, the matrix-coverage test and one `--json`
  envelope from the test-only app per exit class.

## Implementation Notes

- Keep `cli.py` unchanged except one line in `_register_commands()` attaching the root
  callback defined in `commands/version.py` (eager `--version` option, `typer.Exit`). The
  callback has no docstring and no `help=`; a test asserts the root `--help` description
  text is unchanged.
- Validate envelopes with the Pydantic model; do not add a `jsonschema` dependency.
- Contract commands never declare `exists=`, `readable=`, `file_okay=`/`dir_okay=` on path
  arguments; missing or unreadable inputs are `unreadable` (exit 3) envelopes.
- `atomic_io.py` is the single writer primitive for T-104 (`corpus_io.write_corpus`),
  T-107, T-109, T-113, T-117 and T-118; do not migrate `claims record` (it keeps its
  blocking lock) or `save_selections` in this task.
- Document the `claims` exception in the exit matrix: deck and corpus commands retain
  v0.2.1 JSON/codes and failures exit 1, except corpus `--value`, `--evidence` and
  `--selected-sense` rejection (exit 3); invalid batch items remain exit 1 and Click
  usage errors exit 2. No envelope retrofit or claims implementation belongs here.
- TDD order: registry/docs test, envelope model, exit precedence, `run_contract`,
  `atomic_io`, `canonical`/`json_schemas`, `--version`. Size guard: do not add any helper
  beyond `cli_contract.py`, `atomic_io.py`, `canonical.py`, `json_schemas.py`,
  `commands/version.py` and `scripts/export-json-schemas.py`; file a follow-up rather than
  retrofit any legacy command.
