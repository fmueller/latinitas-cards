---
id: T-096-scope-v0-2-1-native-release-acceptance-to-linux
title: Scope v0.2.1 native release acceptance to Linux Desktop
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#automatic-lightdark-appearance
dependencies: []
updated_at: "2026-10-13T20:19:12Z"
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
