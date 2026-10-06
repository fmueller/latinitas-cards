---
id: T-069-check-v0-2-0-gaps-and-drift
title: Check v0.2.0 gaps, inconsistencies, and spec drift
status: todo
priority: high
spec_ref: specs/v0.2.0.md#goals
dependencies: []
updated_at: "2026-10-06T20:02:53Z"
---

# T-069-check-v0-2-0-gaps-and-drift Check v0.2.0 gaps, inconsistencies, and spec drift

## Description

Before publishing v0.2.0 (T-068), review the shipped implementation against
specs/v0.2.0.md for gaps, internal inconsistencies, and drift. Cover every
Potential Features area (managed update plans, note/card lifecycle, destination
tag ownership, safe update application, pre-release migration boundary,
evidence-led extraction, calibrated form parsing, morphology themes) plus the
Caution, Delivery Order And Open Evidence, and Explicitly Excluded sections.

## Acceptance

- Run `taskrail coverage` and the taskrail-gap review; record structural and
  semantic gap findings per spec area.
- Compare each spec area's promises with code, CLI help, tests, and completed
  task notes; flag unimplemented, partially implemented, or over-claimed items.
- Check drift: behavior or CLI surface that contradicts or exceeds the spec,
  and spec text that no longer matches what shipped.
- Check consistency across README, CHANGELOG `[Unreleased]`, docs/, specs/README.md,
  AGENTS.md, and command help (names, options, defaults, limitations).
- Confirm explicitly excluded items are not shipped or claimed.
- Fix small doc/spec inconsistencies in place; create follow-up tasks via
  `taskrail task new` for anything larger, marking which ones block the release.
- Record a summary of findings and their disposition in Verification Notes.

## Verification Notes

- Pending.

## Implementation Notes

- Review only; release metadata and publication belong to T-068.
