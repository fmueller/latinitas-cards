---
id: T-116-define-the-versioned-automation-policy-file-and
title: Define the versioned automation policy file and routing truth table
status: todo
priority: high
spec_ref: specs/v0.3.0.md#policy-gated-agent-automation
dependencies:
    - T-103-define-the-cli-result-envelope-exit-codes
    - T-113-define-annotation-sidecar-records-and-the
updated_at: "2026-10-14T05:42:59Z"
---

# T-116-define-the-versioned-automation-policy-file-and Define the versioned automation policy file and routing truth table

## Description

Define the `latinitas-policy` version 1 file, its routing function and `policy validate
FILE [--json]`, per `specs/v0.3.0.md#policy-gated-agent-automation` and Fixed Contracts
**Policy**, **Versioned documents** and **Identity and hashing** (`producer.kind`,
producer globs match `name`). The policy gates sidecar writes, not linguistic acceptance:
it is not part of the deck profile (`profile.py`), never enters claim fingerprints and
never routes to a claim decision. `docs/claim-review-policy.md` (automatic acceptance
disabled) stays in force. This task delivers a pure routing function and its truth table;
computing the destructive and conflicting facts belongs to T-117 and T-119, and
`annotations plan` consumes the router in T-117.

## Acceptance

- `src/latinitas_cards/policy.py` defines strict Pydantic models: `format:
  "latinitas-policy"`, `version: 1` (strict int), ordered `rules: [{operation:
  add|supersede|retract|*, layer: segmentation|morphology|translation|correction|*,
  producer_kind: rule|analyzer|agent|llm|human|*, producer: <glob>, action:
  auto|review|deny}]`, `receipt_actor: human|any` (default `human`); unknown keys at any
  level are invalid. `producer` is required (use `"*"` for any).
- `policy validate FILE --json` exits 0 for valid fixtures and 3 for each invalid one
  under `tests/fixtures/policy/`: `threshold.json`, `min-confidence.json`,
  `calibration.json` (one such field each), `unknown-field.json`, `auto-supersede.json`,
  `auto-retract.json`, `auto-any-operation.json` (`operation: "*"` with `auto`),
  `version-float.json` (`1.0`), `version-2.json`, `other-format.json`, `not-json.json` and
  a missing path; codes are `invalid_policy`, `unsupported_version`, `unsupported_format`,
  `invalid_json`, `unreadable`. A missing file reaches the callback (exit 3 with an
  envelope), not Click's `exists=True` check (exit 2).
- `route(operation, layer, producer_kind, producer_name, destructive, conflicting,
  policy)` returns `Route(action, reasons)` with `action` in `auto|review|deny` only.
  Without a policy `add` is `auto` and every other operation `review`; with a policy the
  first matching rule wins (`fnmatch.fnmatchcase` on `producer_name`), else `review`;
  destructive or conflicting operations are never `auto`: a matching `deny` still denies,
  otherwise they become `review`. Reasons are ordered tokens from a closed set documented
  in `docs/automation-policy.md` (`no_policy`, `rule:<index>`, `default_review`,
  `destructive`, `conflicting`); `rule:<index>` uses the zero-based rule index.
- Truth table: a parametrized test over operation x layer x producer kind x destructive x
  conflicting (`add` is never destructive; a `retract` can be conflicting, against a sidecar
  supersede of the same anchor and layer by the same producer: 200 combinations per policy,
  40 `add`, 80 `supersede`, 80 `retract`, asserted as `len(rows) == 3 * 200`) under no
  policy, under
  `first-match.json` (a narrower `auto` rule before a broader `review` rule, a producer
  glob such as `fixture-*`) and under `deny-precedence.json`, against expected rows in
  `tests/fixtures/policy/truth-table.json`. The expected file is produced by a generator
  in the test directory that does not import `policy.py`, and committed; at least ten
  hand-written spot rows (first match, glob miss, default `review`, `deny` over
  destructive, `deny` over conflicting) are asserted literally.
- `policy_digest(path)` is `sha256:` plus the hex digest of the file bytes, or `None`
  without `--policy`. Import tests show `policy.py` imports nothing from `claim_review`,
  `claim_sidecar` or `profile`, and `DeckProfile` gains no field.
- The policy model registered in `json_schemas.SCHEMAS`
  (`docs/schemas/latinitas-policy.v1.schema.json`, shared drift test), a new
  `docs/automation-policy.md` (rules, first match, default, never-`auto` rules, actors are
  declared not authenticated, calibrated acceptance is v0.5.0), one added paragraph in
  `docs/claim-review-policy.md` stating that the v0.3.0 policy gates sidecar writes only,
  and the matrix row `policy validate: 0, 3`.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the truth-table row count and the SHA-256 of `truth-table.json`.

## Implementation Notes

- New `policy.py` (models, loader raising contract diagnostics, `route`, `policy_digest`)
  and `commands/policy.py`, a Typer sub-app registered as `policy` in
  `_register_commands()`, using `run_contract` from T-103.
- The `auto` restriction is a model validator: any rule whose `operation` is `supersede`,
  `retract` or `*` with `action: auto` is invalid. Loader errors carry the same codes in
  `policy validate` and in `annotations plan`/`apply` (T-117, T-118).
- `receipt_actor` is parsed and exposed; its enforcement belongs to T-117, T-118 and
  T-119.
- TDD order: model rejections, validate command, router spot rows, generated truth table.
  Size guard: one module plus one command module.
