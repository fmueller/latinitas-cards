---
id: T-222-avoid-repeated-local-validation-during-agent
title: Avoid repeated local validation during agent workflows
status: completed
priority: low
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies: []
updated_at: "2026-11-07T23:02:15Z"
---

# T-222-avoid-repeated-local-validation-during-agent Avoid repeated local validation during agent workflows

## Description

Complete the local-validation part of the CI improvement requested by the
maintainer. Remove unconditional duplicate pytest at pre-push and make agent
iteration checks focused, while preserving one final local code gate required
by the active spec's whole-spec acceptance section.

## Acceptance

- Pre-push no longer invokes pytest; real attribution/author guards still run.
- Pre-commit's staged lint/format, Python-triggered mypy, planning validation
  and policy guards remain unchanged.
- AGENTS.md distinguishes focused development checks from one final ordered
  ruff/mypy/pytest gate before code delivery or task completion. Later executable
  changes invalidate it; unchanged executable inputs do not need another run.
- Documentation/planning-only work uses relevant structural validation, with
  affected documentation/spec contract tests and explicit exceptions for
  executable examples, packaging and runtime inputs.
- Preserve T-221's CI behavior and all maintainer-only approval gates.

## Verification Notes

- A disposable repository and the real pinned Lefthook runner reproduced the
  duplicate before the change: the pre-push hook invoked `uv run pytest`.
- The corrected hook probe accepts a clean file-changing push, invokes no uv,
  and rejects an agent-authored push. An independent specialist also checked
  forbidden attribution/session URLs and existing-branch updates. Baseline
  config accepted the same agent-authored push and reran pytest.
- Parsed config and byte comparison confirm pre-commit and commit-msg remain
  unchanged. Lefthook's parsed pre-push contains only the stdin-enabled guard.
- Commit/push message and author guard suites, Taskrail and whitespace passed.
- Final full chain passed in order: ruff; mypy (199 source files); pytest
  (3262 passed, 7 skipped, one warning, 440.27 seconds).
- Simplifier made no changes. Independent policy and hook-safety reviews were
  reconciled with no unresolved required findings. Corrected the policy review's
  documentation-contract gap: docs consumed by tests require affected contract
  tests before reusing other passing evidence, not just structural validation.

## Implementation Notes

- No test-result cache, bypass environment variable or hook installation change
  is introduced. Existing installed Lefthook wrappers read the updated config.
- `use_stdin: true` fixes Git ref forwarding to the retained safety guard;
  the real-runner negative control showed this was necessary for enforcement.
- Pre-existing limitation: empty-commit-only pushes are skipped by Lefthook in
  both baseline and revised config. Filed a separate suggested fix with the
  verified `no matching push files` evidence; not widened or fixed by this task.
- T-221 CI changes remain intact. Work remains local and uncommitted on main.
- 2026-11-07T23:02:15Z: verification pass
- 2026-11-07T23:02:15Z: Local validation policy and hook changes verified; not committed or pushed.
