---
id: T-078-close-managed-lifecycle-classification-test-gaps
title: Close managed lifecycle and classification test gaps
status: completed
priority: low
spec_ref: specs/v0.2.0.md#note-and-card-lifecycle-contract
dependencies: []
updated_at: "2026-10-08T00:45:12Z"
---

# T-078-close-managed-lifecycle-classification-test-gaps Close managed lifecycle and classification test gaps

## Description

Found by the T-069 review. Missing tests: conflicting suspension ownership, retiring an
already user-suspended card, the changed tool-suspension branch, and a card-level `reactivate`
operation refused at approve. A managed note deleted in the destination is classified `create`
rather than a conflict, and a `retire` classification hides an underlying conflict.
Not release-blocking (lifecycle operations are unsupported at apply).

## Acceptance

- Tests cover the four listed lifecycle branches.
- A previously anchored note missing from a complete destination is a reviewed conflict, not `create`; retire entries keep their conflict reasons visible.

## Verification Notes

- Strict RED: anchored complete absence was `create` instead of `conflict`.
  GREEN: the minimal baseline-anchor guard yields a blocked conflict without
  proposed operations; genuinely never-anchored absence remains create.
- Five deliberate regressions failed the new tests: conflicting ownership,
  changed tool suspension, user-suspended retirement, masked retire reasons,
  and accepted card reactivation. Restored production behavior: 45 focused tests pass.
- Final `uv run ruff check`, `uv run mypy`, `uv run pytest -v` pass:
  89 source files checked, 900 tests passed. Installed CLI smoke confirms
  anchored absence conflict/no operations and reactivate approval refusal
  (exit 1, unsupported operation, no receipt). Initial smoke expected exit 2;
  corrected to the command's documented exit 1 after inspecting its error handler.
- Dedicated simplifier moved the adopt import only. Independent General,
  persistence/state integrity, Python, and Security/data-loss lanes each returned
  "No concrete task-relevant findings." Fresh candidate validation and
  disposition verification agreed; no fixes, deferrals, or follow-ups required.

## Implementation Notes

- Pin: specs/v0.2.0.md, note-and-card-lifecycle-contract. Accepted remote base
  contains T077; no transferred or uncommitted source work was needed.
- Retire reasons were already preserved in the current implementation; regression
  coverage now protects field and tag reasons without unnecessary production edits.
- Asymmetric synthetic fixtures check exact card IDs, ownership, suspension,
  siblings, anchors and history. They establish offline planning/refusal only,
  not native lifecycle execution or structural preservation guarantees.
- Lifecycle and creation remain unsupported at approval/application. No transport,
  schema, cli.py, deletion, or unsuspension implementation was added.
- 2026-10-08T00:45:12Z: verification pass
- 2026-10-08T00:45:12Z: Anchored absence reviewed conflict; four lifecycle branches and retire reasons protected; offline lifecycle remains unsupported
