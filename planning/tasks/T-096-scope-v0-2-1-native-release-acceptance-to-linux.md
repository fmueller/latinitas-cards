---
id: T-096-scope-v0-2-1-native-release-acceptance-to-linux
title: Scope v0.2.1 native release acceptance to Linux Desktop
status: completed
priority: medium
spec_ref: specs/v0.2.1.md#automatic-lightdark-appearance
dependencies: []
updated_at: "2026-10-13T20:32:28Z"
---

# T-096-scope-v0-2-1-native-release-acceptance-to-linux Scope v0.2.1 native release acceptance to Linux Desktop

## Description

On 2026-10-13 the maintainer explicitly authorized removing AnkiMobile testing
from the v0.2.1 release gate and continuing through the v0.2.1 release. Amend
the active spec and current native/readiness documentation narrowly: native
Linux Anki Desktop acceptance remains required; AnkiMobile remains unverified
for the new appearance behavior. Preserve dated historical client evidence and
the earlier implementation-only deferral without presenting either as new proof.

## Acceptance

- Record the dated decision and update the active spec's native acceptance gate
  to require actual Linux Anki Desktop, not AnkiMobile. Keep compatibility CSS
  and fallback behavior unchanged; no runtime or capability changes.
- Current native checklist and v0.2.1 readiness guidance consistently distinguish
  required Desktop testing from unverified Mobile. No browser or backend fixture
  may substitute for actual Desktop reviewer/import interaction evidence.
- Preserve the Desktop matrix: both palettes, default/light/dark/auto, actual
  reviewer front/reveal/answer and disclosure, and native light-night-light
  toggling without regeneration. Cover the affected phrase/principal/parsing
  recipes with synthetic isolated profiles; no live collection, account or sync.
- The outer overseer subsequently runs native synthetic acceptance, validates
  findings, and uses the existing sequential worker repair loop if necessary.
  This task does not run or certify that acceptance, publish, tag, or deploy.
- Use workflow-v3 sourced via autonomous-backlog, independent review and final
  validation before Taskrail completion; deliver focused commit to origin/main.

## Verification Notes

- Initial accepted remote revision: 0be494a55aa377de397cab2371d208e19514a1d9.
- Initial spec SHA256: 0b2150fc9778a3d7c0b058a5d0961a819ee283f148569323c32eee07ac3776b4.
- This is an explicitly authorized spec amendment, not permission to change
  unrelated scope, priorities, dependencies, acceptance gates or runtime behavior.

## Implementation Notes

- Fetched origin/main and confirmed the clean accepted planning base
  `e1a229f41aedb2394801c5dbbdc42964d885e89e`; validate, next and independent
  status inspection confirmed T096 before start. No product transfer was needed.
- Amended only the active spec and current native/reference/readiness guidance.
  Preserved the dated historical evidence and the superseded two-client checklist;
  recorded the 2026-10-13 decision separately from the implementation-only deferral.
- Exactly one Task implementation worker handled the documentation amendment.
  Docs-only validation used textual/diff inspection, not artificial runtime TDD.
  Dedicated code-simplifier Task clarified comparison/disclosure wording only.
- Separate read-only General and Security code-reviewer Tasks both returned
  "No concrete task-relevant findings." Security covered collection safety and
  evidence boundaries; language/framework/database lanes were omitted because
  no runtime, CSS, schema, or data operations changed. Fresh candidate validation
  confirmed the empty candidate set; no findings required fixes or deferral.
- Initial and final exact Ruff, mypy, pytest chains passed. Final results:
  Ruff all checks passed; mypy no issues in 107 source files; 1045 tests passed.
  No native test was run or certified. Linux Desktop acceptance remains required
  and pending; AnkiMobile auto remains unverified and outside this release gate.
- New spec SHA256:
  `aabfdf060b1d90074a672b8cd29875677857b3d2ccb1f4455e09ade0e9de5ca4`.
  The overseer must intentionally repin before its native synthetic acceptance.
- 2026-10-13T20:32:28Z: verification pass
