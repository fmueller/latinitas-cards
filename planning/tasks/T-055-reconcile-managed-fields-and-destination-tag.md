---
id: T-055-reconcile-managed-fields-and-destination-tag
title: Reconcile managed fields and destination tag ownership
status: completed
priority: high
spec_ref: specs/v0.2.0.md#destination-tag-ownership
dependencies:
    - T-054-implement-bound-destination-snapshots-and-observed
updated_at: "2026-10-05T02:56:47Z"
---

# T-055-reconcile-managed-fields-and-destination-tag Reconcile managed fields and destination tag ownership

## Description

Build the deterministic three-way field/tag reconciliation used by managed plans, preserving destination-only user data.

## Acceptance

- Managed fields change safely only when destination equals baseline; destination equal to proposal is convergent/no-write. Divergence from both is a conflict, including destination-only edits regeneration would undo.
- Resolve explicitly by keep destination, accept proposal, or reviewed replacement; record decisions and resulting managed values. Personal Notes and other user-owned fields are never writable or offered as overridable conflicts.
- Preserve destination-only tags, separate source/configured/lifecycle contributions, and retain tags with overlapping origins when only one contribution disappears.
- Show managed-tag removal decisions; a deleted but still-required managed tag is a conflict. Explicit keep-as-user-owned overrides persist across later plans. Unknown initial ownership requires review.
- Managed-looking destination additions are not automatically tool-owned. Reserved lifecycle collisions require resolution and never authorize suspension. Do not infer invisible user intent.
- Test exact final tag sets for additions/removals, overlap, user deletion/addition, ownership overrides, collisions, unknown baseline, and no-op regeneration; sibling notes share the same reconciled tags.

## Verification

Use asymmetric three-way fixtures and red/green tests. Run the mandatory ruff, mypy, pytest -v chain.

## Implementation and review evidence

- Added pure managed field/tag reconciliation and a bound shared-note entry point.
  Results expose writes, exact tags, removal candidates, conflicts and reviewed
  decisions/results; only conflict-free results expose a journal target.
- Reused the T-054 snapshot and journal validators. Optional lifecycle origins,
  suppressed required tags and reviewed decisions survive observed anchors;
  existing version-1 ownership remains compatible. No CLI, native transport,
  scheduling authorization or calibration claims were added.
- RED: new comparison tests failed with the missing reconciliation module;
  the bound observation test then failed with missing `reconcile_note`.
  GREEN: initial full validation passed 736 tests.
- A re-added suppressed tag test failed because suppression remained in the
  returned ownership; intersecting suppression with required absent tags fixed it.
- Dedicated code-simplifier Task cached baseline tag sets/destination additions.
  Its checks and the integrating focused run passed 45 tests.
- Separate independent General, Security, Database/persistence and Python
  code-reviewer Tasks reviewed the actual changes. The first three returned:
  "No concrete task-relevant findings."
- Python candidate PY-1: "Fields recorded in keep_fields can still appear as
  proposed writes and in a conflict-free reconciliation target when the proposal
  differs from the destination." Fresh candidate validation rejected PY-1:
  the existing if/elif guard retains destination and direct reproduction had
  no writes. This was not a production defect.
- Candidate validation identified a test gap: "The reconciliation tests do not
  cover that keep-field behavior." Added direct regression coverage and proved
  it by temporarily disabling the guard: `Meaning=P` failed the expected
  `Meaning=B` assertion. Restoring the guard passed; no mutation remains.
- Self-review found explicit accept-proposal after a keep-tag override removed
  the proposed tag. The new test failed with `[]` versus `[reserved]`;
  processing the reviewed choice before the override fixed it.
- Exact shared-note tags and asymmetric sibling-card rows round-trip through
  observed state. An independent fresh disposition-verification Task verified
  the rejected candidate, fixed test gap and tag-choice fix and returned:
  "No concrete task-relevant findings."
- Final exact chain: `uv run ruff check` passed; `uv run mypy` passed (76 source
  files); `uv run pytest -v` passed (738 tests). Fresh reviewer independently
  repeated the same passing chain and focused tests (47 passed).
- No deferred findings or new follow-ups. Native apply/proof, card lifecycle
  planning and CLI plan/apply remain the existing later tasks.

## Implementation Notes

- 2026-10-05T02:56:47Z: verification pass
- 2026-10-05T02:56:47Z: Implemented offline reconciliation; independent lane review and fresh disposition verification complete; mandatory ruff/mypy/738-test chain passed.
