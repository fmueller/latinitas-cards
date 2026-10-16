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

Plans compute destructiveness for `supersede`/`retract` from accepted or withheld
decisions bound to the target annotation ID, even if the decision binding is stale,
and from annotations depending on the target segmentation. Conflicts are different
values at the same layer and anchor from the same producer. Current live records
and the entire patch contribute to these facts, deduplicated by annotation ID;
different creation timestamps alone do not create a conflict.

`Route(action, reasons)` contains ordered tokens from this closed vocabulary:

1. Exactly one base token: `no_policy`, `rule:<index>` (zero-based winning rule),
   or `default_review` (policy present with no matching rule).
2. `destructive`, if that fact is true, even when the action is already review/deny.
3. `conflicting`, if that fact is true, even when the action is already review/deny.

Plans also classify operation-level invalidity using the closed reasons `unmatched`,
`stale`, `orphaned`, `input_hash`, `invalid_value`, `span_out_of_range`,
`unknown_segmentation`, `unknown_target`, `target_inactive`, `duplicate_operation`.
Invalid operations are persisted for inspection but cannot receive approval.

`policy_digest(path)` returns `sha256:` plus SHA-256 of the **raw file bytes**,
so whitespace changes the digest; no path returns `None`.

## Safety boundary

`receipt_actor` is `human` (default) or `any`. Actors are declared, **not
authenticated**. `annotations approve` enforces it; an agent declaration is refused
when the plan requires a human. Apply enforces the same rule for hand-built receipts.
`receipt_actor: any` permits either declaration; it does not authenticate the actor.
Invalid operations cannot be approved or applied regardless of producer or actor kind.
Denied operations cannot be approved and are reported as warnings without appending
at apply. Validation and routing do not approve or apply operations.

## Apply-time safety

Apply holds the corpus, annotation sidecar and exact `<CORPUS>.claims.json.lock`
writer locks from snapshot reads through replacement. Contention exits 4 `busy`;
`claims record` keeps its blocking protocol on that same claims lock. Lock files may
be created even when claims are absent, but claims are never written by apply.

Every pending operation is checked again against the current corpus, effective
sidecar, whole embedded patch and version 2 decisions in `<CORPUS>.claims.json`.
Its validity, route and complete safety facts (decision status and sorted dependent
and conflicting annotation IDs) must still match. Any changed fact or route, or a
newly invalid pending operation, rejects the **whole batch before any append** with
4 `binding_drift`, even if the route remains `review`. Re-plan and re-approve;
an old receipt cannot authorize changed context. Unrelated decisions do not cause
drift and there is no claims digest in the plan. Expected partial appends are
deduplicated; completed records are not reapplied or subjected to pending checks.

Missing claims means no decisions. Unreadable, malformed and v1 claims fail closed
with exit 3 `unreadable`, `invalid_sidecar` and `unsupported_version`, respectively.

The policy gates sidecar writes only, not linguistic acceptance. It is not part
of the deck profile or claim fingerprint and cannot create a claim decision.
[Automatic linguistic acceptance remains disabled](claim-review-policy.md);
calibrated acceptance is v0.5.0 scope. No confidence threshold is accepted here.

The committed `tests/fixtures/policy/truth-table.json` contains 600 expected rows:
200 each for no policy, `first-match.json` and `deny-precedence.json`. Regenerate
with `uv run python tests/fixtures/policy/generate_truth_table.py`; this independent
generator imports no application code. Handwritten tests additionally distinguish
glob misses/case, first-match ordering, default review and deny under both risks.
