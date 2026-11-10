---
id: T-223-enforce-push-policy-for-empty-commits
title: Enforce push policy for empty commits
status: completed
priority: medium
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies: []
updated_at: "2026-11-07T23:23:29Z"
---

# T-223-enforce-push-policy-for-empty-commits Enforce push policy for empty commits

## Description

Ensure Lefthook invokes the existing push-policy scanner for empty-commit-only
pushes, preserving identity and attribution checks without restoring pytest.

## Acceptance

- Real pinned Lefthook rejects forbidden authors and attribution on empty-only
  new-branch pushes and existing-branch updates.
- Clean empty roots/updates, no-op updates and branch deletion remain accepted.
- Regression tests exercise the actual hook configuration and runner, not just
  the policy script; no uv invocation is allowed. CI installs the pinned runner.
- Missing scanner fails closed; a normal file-changing push also proves pytest
  stays disabled. Git's remote-name/URL arguments must not become scan ranges.
- Existing pre-commit/commit-msg checks and prior CI/local-gate work remain;
  editing the new hook entry point triggers the existing push-policy suite.

## Verification Notes

- RED: the extended push-message suite failed with `an empty agent-authored
  push was accepted` and Lefthook's `no matching push files` skip diagnostic.
- GREEN: the extended real-Lefthook suite passes on the final script-hook
  implementation, including remote-name/URL arguments and missing-scanner failure.
- Actual installed hooks in disposable local repositories accepted a clean empty
  root via `git push` and rejected empty commits with forbidden author/attribution;
  the destination refs remained unchanged after both rejected pushes.
- Independent general/claims and security reviewers plus the coordinator found
  no remaining required findings after two review fixes. The simplifier made
  no edits to the final revision.
- Mutation probes reject restored pytest on a nonempty clean push, unconditional
  wrapper success and accidental forwarding of remote/URL args. Missing scanner
  and wrapper fail closed; stdin and space-containing paths were independently
  exercised. No deliberate mutations remain in the working tree.
- Shell syntax, Taskrail and whitespace checks passed. Final full chain passed
  in order: ruff; mypy (199 source files); pytest (3262 passed, 7 skipped,
  one warning, 467.34 seconds). No executable changes followed that final run.

## Implementation Notes

- Use Lefthook's script hook, which does not implicitly filter pushed files,
  retaining stdin forwarding. Its four-line entry point execs the existing
  scanner without forwarding Git's remote/URL arguments as revision ranges.
- Rejected a custom `files` selection after review proved a missing scanner
  would silently skip. Added its RED regression before switching to script mode.
  The pinned Lefthook ignores `always`; no unsupported option is retained.
- Install the existing mise-pinned tools in CI's guard job so hook integration
  coverage also runs on clean GitHub runners. No optional skip hides this test.
- 2026-11-07T23:23:29Z: verification pass
- 2026-11-07T23:23:29Z: Empty-commit bypass fixed and verified; not committed or pushed to project remote.
