# Sidecar-write automation policy (v1)

`latinitas-cards policy validate FILE [--json]` validates without writing files.
JSON mode emits the shared CLI envelope (exit 0 or 3); a missing file is an
`unreadable` diagnostic, not a Click argument error. Other codes are `invalid_json`,
`unsupported_format`, `unsupported_version`, and `invalid_policy`.

```json
{
  "format": "latinitas-policy",
  "version": 1,
  "receipt_actor": "human",
  "rules": [
    {"operation": "add", "layer": "morphology", "producer_kind": "agent",
     "producer": "fixture-*", "action": "auto"},
    {"operation": "*", "layer": "*", "producer_kind": "*",
     "producer": "*", "action": "review"}
  ]
}
```

All rule fields are required. Operations are `add`, `supersede`, `retract` or `*`;
layers are `segmentation`, `morphology`, `translation`, `correction` or `*`;
producer kinds are `rule`, `analyzer`, `agent`, `llm`, `human` or `*`.
`producer` is a case-sensitive shell glob matched against producer **name**, not
kind or version (`fnmatch.fnmatchcase`; `*`, `?`, `[abc]` are supported).
`action` is `auto`, `review` or `deny`. Unknown keys at every level, including
threshold, `min_confidence` and calibration, are invalid. Duplicate JSON keys at
any level are also rejected as `invalid_policy` so permissions cannot be hidden
by ambiguous object members. Version must be the
integer 1, not `1.0`, a boolean or a string. See the
[generated schema](schemas/latinitas-policy.v1.schema.json); model validation also
enforces the cross-field restriction that `auto` requires explicit `operation: add`.

## Routing contract

Rules are ordered: the **first** matching rule wins, even if a later one denies.
With a policy, no match means `review` (including an empty rules list).
Without a policy, `add` is `auto` and other operations are `review`.
An `auto` rule for `supersede`, `retract` or `*` is invalid.
Destructive or conflicting operations are never `auto`: demote to `review`,
but retain a matching `deny`. The caller supplies those risk facts; this router
does not compute them or perform writes.

`Route(action, reasons)` contains ordered tokens from this closed vocabulary:

1. Exactly one base token: `no_policy`, `rule:<index>` (zero-based winning rule),
   or `default_review` (policy present with no matching rule).
2. `destructive`, if that fact is true, even when the action is already review/deny.
3. `conflicting`, if that fact is true, even when the action is already review/deny.

`policy_digest(path)` returns `sha256:` plus SHA-256 of the **raw file bytes**,
so whitespace changes the digest; no path returns `None`.

## Safety boundary

`receipt_actor` is `human` (default) or `any`. Actors are declared, **not
authenticated**. This task parses that setting; future plan/apply/approval
commands enforce it. Validation and routing do not approve or apply operations.

The policy gates sidecar writes only, not linguistic acceptance. It is not part
of the deck profile or claim fingerprint and cannot create a claim decision.
[Automatic linguistic acceptance remains disabled](claim-review-policy.md);
calibrated acceptance is v0.5.0 scope. No confidence threshold is accepted here.

The committed `tests/fixtures/policy/truth-table.json` contains 600 expected rows:
200 each for no policy, `first-match.json` and `deny-precedence.json`. Regenerate
with `uv run python tests/fixtures/policy/generate_truth_table.py`; this independent
generator imports no application code. Handwritten tests additionally distinguish
glob misses/case, first-match ordering, default review and deny under both risks.
